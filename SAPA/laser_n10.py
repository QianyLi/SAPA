"""
N-sample BGE-M3 + FAISS expansion with RRF fusion for the recommend task.

Mirrors test_llama_flow_laser.py but handles N>=2 sampled outputs per task:
  for each sample (a candidate ASIN), expand with BGE-M3+FAISS to top_k items,
  then RRF-fuse the N ranked lists into a single top-10 list.

The output JSON has the same shape as the laser script, so
test_compute.py can be used downstream unchanged (its recommend branch
uses get_recommendations_by_history's len==10 short-circuit).

Usage:
  python SAPA/laser_n10.py \
      --input_file        SAPA/data/param_data.json \
      --index_file        PersonalWAB/envs/pwab/functions/search/faiss_dense_bge_m3.index \
      --all_products_jsonl PersonalWAB/envs/pwab/functions/data/Products/all_products.jsonl \
      --output_file       SAPA/data/param_data_laser_n10.json \
      --model_path        BAAI/bge-m3 \
      --top_k_per_seed    10 \
      --rrf_k             60 \
      --final_top_k       10
"""

import argparse
import json
import re
from collections import defaultdict

import faiss
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm


ASIN_RE = re.compile(r'^B[0-9A-Z]{9}')


def parse_args():
    ap = argparse.ArgumentParser(description="N-sample LASER expansion + RRF")
    ap.add_argument('--input_file', type=str, required=True,
                    help='Param JSON where each value is a list of N sampled outputs')
    ap.add_argument('--index_file', type=str, required=True,
                    help='Path to faiss_dense_bge_m3.index')
    ap.add_argument('--all_products_jsonl', type=str, required=True,
                    help='Full product corpus (id+contents) used to encode seeds')
    ap.add_argument('--output_file', type=str, required=True)
    ap.add_argument('--model_path', type=str, default='BAAI/bge-m3')
    ap.add_argument('--top_k_per_seed', type=int, default=10,
                    help='Per-seed FAISS retrieval depth before RRF')
    ap.add_argument('--rrf_k', type=int, default=60, help='RRF constant')
    ap.add_argument('--final_top_k', type=int, default=10,
                    help='Final list length after RRF')
    return ap.parse_args()


class BatchedLaserExpander:
    """BGE-M3 + FAISS, with batched seed encoding for the N-sample case.

    We bypass FlagEmbedding (its import chain breaks under newer transformers)
    and load BGE-M3 directly via transformers AutoModel. The dense embedding
    is the L2-normalised CLS hidden state, which matches BGEM3FlagModel's
    `dense_vecs` output and therefore the existing FAISS index.
    """

    def __init__(self, index_path, all_products_path, model_path, max_length=512,
                 batch_size=32):
        print(f"[laser_n10] loading FAISS index from {index_path}")
        self.index = faiss.read_index(index_path)
        self.batch_size = batch_size
        self.max_length = max_length

        print(f"[laser_n10] loading BGE-M3 from {model_path}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(
            model_path, torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device).eval()

        self.asin_to_contents = {}
        self.id_to_asin = []
        print(f"[laser_n10] loading product corpus from {all_products_path}")
        with open(all_products_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="corpus"):
                d = json.loads(line)
                asin = d['id']
                self.asin_to_contents[asin] = d['contents']
                self.id_to_asin.append(asin)

        if self.index.ntotal != len(self.id_to_asin):
            print(f"[laser_n10] WARN: index size ({self.index.ntotal}) != "
                  f"corpus size ({len(self.id_to_asin)})")

    @torch.no_grad()
    def _encode(self, texts):
        """BGE-M3 dense embedding: L2-normalised CLS hidden state."""
        all_embs = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            enc = self.tokenizer(batch, padding=True, truncation=True,
                                 max_length=self.max_length, return_tensors="pt")
            enc = {k: v.to(self.device) for k, v in enc.items()}
            out = self.model(**enc)
            # CLS token = position 0 of last_hidden_state
            cls = out.last_hidden_state[:, 0]
            cls = F.normalize(cls, p=2, dim=-1)
            all_embs.append(cls.float().cpu().numpy())
        return np.concatenate(all_embs, axis=0).astype('float32')

    def expand_batch(self, seed_asins, top_k):
        """Run FAISS on all seeds in one batched call.

        Returns: list of list[asin], same length as seed_asins. For seeds not
        in the corpus, returns [seed] as a 1-element fallback list.
        """
        valid_idxs = [i for i, a in enumerate(seed_asins) if a in self.asin_to_contents]
        results = [None] * len(seed_asins)

        if not valid_idxs:
            return [[a] for a in seed_asins]

        texts = [self.asin_to_contents[seed_asins[i]] for i in valid_idxs]
        embs = self._encode(texts)
        D, I = self.index.search(embs, top_k)

        for k, original_pos in enumerate(valid_idxs):
            row = []
            for idx in I[k]:
                if idx >= 0 and idx < len(self.id_to_asin):
                    row.append(self.id_to_asin[idx])
            results[original_pos] = row

        for i, r in enumerate(results):
            if r is None:
                results[i] = [seed_asins[i]]
        return results


def extract_seed(sample):
    """Pull the first ASIN out of a single sampled output string."""
    s = str(sample).strip()
    if not s:
        return None
    # Sometimes the model emits "B07XXXXXX" or "B07XXXXXX, B08...".
    head = s.split(',')[0].strip()
    if ASIN_RE.match(head):
        return head
    # fallback: scan whole string for first ASIN-shaped token
    for tok in re.split(r'[\s,]+', s):
        if ASIN_RE.match(tok):
            return tok
    return None


def rrf_merge(ranked_lists, k_const, top_k):
    """Reciprocal Rank Fusion over multiple ranked lists of ASINs."""
    scores = defaultdict(float)
    for lst in ranked_lists:
        for rank, asin in enumerate(lst):
            scores[asin] += 1.0 / (k_const + rank + 1)
    return [a for a, _ in sorted(scores.items(), key=lambda x: -x[1])[:top_k]]


def main():
    args = parse_args()
    expander = BatchedLaserExpander(args.index_file, args.all_products_jsonl,
                                    args.model_path)

    print(f"[laser_n10] loading task results from {args.input_file}")
    tasks = json.load(open(args.input_file, encoding='utf-8'))

    out = {}
    rec_count = 0
    seed_diversity = []

    for instruction, samples in tqdm(tasks.items(), desc="tasks"):
        # Non-recommend tasks pass through untouched
        if not samples or not isinstance(samples, list):
            out[instruction] = samples
            continue
        first = str(samples[0]).strip() if samples else ""
        if not ASIN_RE.match(first):
            out[instruction] = samples
            continue

        rec_count += 1
        # Extract one seed per sample; drop ones we can't parse
        seeds = []
        for s in samples:
            sd = extract_seed(s)
            if sd:
                seeds.append(sd)
        if not seeds:
            out[instruction] = samples
            continue
        seed_diversity.append(len(set(seeds)))

        # Batched FAISS expansion: one ranked list per seed
        ranked_lists = expander.expand_batch(seeds, args.top_k_per_seed)
        fused = rrf_merge(ranked_lists, args.rrf_k, args.final_top_k)

        # Pad in case fusion gave fewer than final_top_k (rare, only if all
        # seeds were OOV)
        if len(fused) < args.final_top_k:
            fused = fused + [s for s in seeds if s not in fused]
            fused = fused[:args.final_top_k]

        out[instruction] = fused

    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"[laser_n10] expanded {rec_count} recommend tasks")
    if seed_diversity:
        print(f"[laser_n10] mean unique seeds per task: "
              f"{sum(seed_diversity)/len(seed_diversity):.2f}")
    print(f"[laser_n10] wrote {args.output_file}")


if __name__ == "__main__":
    main()
