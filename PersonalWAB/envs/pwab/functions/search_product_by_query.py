from typing import Any, Dict, List, Union
from sentence_transformers import CrossEncoder
import json
import os
import pickle
import re
import torch

FOLDER_PATH = os.path.dirname(__file__)
INDEX_PATH = os.path.join(FOLDER_PATH, 'search', 'indexes')

RERANKER_PATH = os.getenv("PWAB_RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")

_BM25_PICKLE = os.environ.get("PWAB_BM25_INDEX", "").strip() or None

bm25_searcher = None
_PY_BM25 = None
_PY_BM25_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _py_tokenize(text):
    return [t.lower() for t in _PY_BM25_TOKEN_RE.findall(text or "")]


if _BM25_PICKLE:
    print(f"[search] loading Python BM25 index from: {_BM25_PICKLE}")
    with open(_BM25_PICKLE, "rb") as _f:
        _PY_BM25 = pickle.load(_f)
    print(f"[search] Python BM25 ready | docs={len(_PY_BM25['asins'])}")
else:
    from pyserini.search.lucene import LuceneSearcher
    print(f'Loading BM25 searcher from: {INDEX_PATH}')
    bm25_searcher = LuceneSearcher(INDEX_PATH)

print(f'Loading Reranker from: {RERANKER_PATH} ...')
try:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    reranker = CrossEncoder(RERANKER_PATH, device=device, max_length=512)
    print(f"✅ Reranker loaded on {device}")
except Exception as e:
    print(f"❌ Reranker load failed: {e}")
    reranker = None


def search_product_by_query(
    data: Dict[str, Any],
    query: Union[str, List[str]],
    target_asin: str = ""
) -> List[str]:

    original_instruction = data.get('instruction', "")
    sample_id = original_instruction

    if isinstance(query, str):
        queries = [query]
    else:
        queries = query

    candidate_map = {}

    if _PY_BM25 is not None:
        import numpy as np
        asins = _PY_BM25["asins"]
        contents_all = _PY_BM25["contents"]
        bm25 = _PY_BM25["bm25"]
        for q in queries:
            if not q or not isinstance(q, str) or not q.strip():
                continue
            try:
                toks = _py_tokenize(q)
                if not toks:
                    continue
                scores = bm25.get_scores(toks)
                top_idx = np.argpartition(-scores, min(20, len(scores) - 1))[:20]
                top_idx = top_idx[np.argsort(-scores[top_idx])]
                for idx in top_idx:
                    docid = asins[idx]
                    if docid not in candidate_map:
                        candidate_map[docid] = contents_all[idx]
            except Exception as e_search:
                print(f"❌ Python BM25 error on query '{q}': {e_search}")
                continue
    else:
        for q in queries:
            if not q or not isinstance(q, str) or not q.strip():
                continue

            try:
                hits = bm25_searcher.search(q, k=20)

                for hit in hits:
                    if hit.docid not in candidate_map:
                        try:
                            doc_object = bm25_searcher.doc(hit.docid)
                            if doc_object is not None:
                                raw_json = doc_object.raw()
                                item = json.loads(raw_json)
                                content = item.get('contents', item.get('text', str(item)))
                                candidate_map[hit.docid] = content
                        except Exception as e_parse:
                            print(f"❌ Error fetching doc {hit.docid}: {e_parse}")
                            continue
            except Exception as e_search:
                print(f"❌ Search Error on query '{q}': {e_search}")
                continue

    docids_list = list(candidate_map.keys())
    contents_list = list(candidate_map.values())

    actual_N = len(candidate_map)

    hit_before_rerank = 1 if target_asin and target_asin in docids_list else 0

    if not contents_list:
        recall_log = os.getenv("PWAB_RECALL_LOG", "SAPA/data/recall_exp4_multi.jsonl")
        os.makedirs(os.path.dirname(recall_log) or ".", exist_ok=True)
        with open(recall_log, "a", encoding="utf-8") as f:
            log_data = {
                "instruction": sample_id,
                "target_asin": target_asin,
                "pool_size_N": 0,
                "hit_before_rerank": 0,
                "hit_after_rerank": 0
            }
            f.write(json.dumps(log_data, ensure_ascii=False) + "\n")
        return[]

    if reranker and original_instruction:
        try:
            pairs = [[original_instruction, doc] for doc in contents_list]
            scores = reranker.predict(pairs, batch_size=32, show_progress_bar=False)

            ranked_results = sorted(zip(docids_list, contents_list, scores), key=lambda x: x[2], reverse=True)

            final_top10_ids =[docid for docid, content, score in ranked_results[:10]]
            final_top10_contents = [content for docid, content, score in ranked_results[:10]]

        except Exception as e_rank:
            print(f"❌ Rerank Error: {e_rank}")
            final_top10_ids = docids_list[:10]
            final_top10_contents = contents_list[:10]
    else:
        final_top10_ids = docids_list[:10]
        final_top10_contents = contents_list[:10]

    hit_after_rerank = 1 if target_asin and target_asin in final_top10_ids else 0

    with open("recall_exp4_multi.jsonl", "a", encoding="utf-8") as f:
        log_data = {
            "instruction": sample_id,
            "target_asin": target_asin,
            "pool_size_N": actual_N,
            "hit_before_rerank": hit_before_rerank,
            "hit_after_rerank": hit_after_rerank
        }
        f.write(json.dumps(log_data, ensure_ascii=False) + "\n")

    results = []
    for i, (asin, content) in enumerate(zip(final_top10_ids, final_top10_contents)):
        results.append('Product ' + str(i) + ' [ASIN: ' + str(asin) + ']: \n' + content + '\n')

    return results


search_product_by_query.__info__ = {
    "type": "function",
    "function": {
        "name": "search_product_by_query",
        "description": "Multi-query Recall + BGE Reranking",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"}
            },
            "required": ["query"],
        },
    },
}
