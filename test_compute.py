import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import argparse
import os
import numpy as np
from PersonalWAB.envs.pwab.functions.get_recommendations_by_history import get_recommendations_by_history
from PersonalWAB.envs.pwab.functions.search_product_by_query import search_product_by_query
from tabulate import tabulate


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate PWA tasks")
    parser.add_argument('--evaluate_dpo', type=str, default='False', help='Whether to evaluate DPO')
    parser.add_argument('--task_file', type=str, default='SAPA/data/user_instructions.json', help='Path to task file')
    parser.add_argument('--param_file', type=str, default='SAPA/output/res/', help='Path to tool input file')
    parser.add_argument('--function_file', type=str, default='SAPA/output/', help='Path to tool selected file')
    parser.add_argument('--all_products', type=str, default='SAPA/data/all_products.json', help='Path to all products file')
    parser.add_argument('--dpo_output', type=str, default='SAPA/data/dpo_data.json', help='Path to DPO output file')


    parser.add_argument('--vectors_file', type=str, default=None, help='Optional path to product_vectors.npy')
    parser.add_argument('--asins_file', type=str, default=None, help='Optional path to product_asins.json')
    parser.add_argument('--rrf_top_k', type=int, default=9, help='Per-seed retrieval depth (seed itself counts as one, so 9 neighbours -> 10 items per list)')
    parser.add_argument('--rrf_constant', type=int, default=60, help='RRF k constant')
    return parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(os.getenv('PWAB_SIM_MODEL', 'sentence-transformers/all-MiniLM-L6-v2'))
model = AutoModel.from_pretrained(os.getenv('PWAB_SIM_MODEL', 'sentence-transformers/all-MiniLM-L6-v2'))
if torch.cuda.is_available():
    model.to('cuda')

def get_consensus_best_index(candidates, tokenizer, model):
    'Documentation.'

    if not candidates or len(candidates) <= 1:
        return 0


    cand_list = [str(c) for c in candidates]


    inputs = tokenizer(cand_list, padding=True, truncation=True, return_tensors='pt').to(model.device)
    with torch.no_grad():
        model_output = model(**inputs)


    token_embeddings = model_output[0]
    input_mask_expanded = inputs['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
    embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    embeddings = F.normalize(embeddings, p=2, dim=1)


    cosine_sim_matrix = torch.mm(embeddings, embeddings.t())
    consistency_scores = cosine_sim_matrix.sum(dim=1).cpu().numpy()


    return int(np.argmax(consistency_scores))

def compute_similarity_origin(target_review, agent_review):
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    sentences = [target_review, agent_review]
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
    if torch.cuda.is_available():
        encoded_input.to('cuda')
    with torch.no_grad():
        model_output = model(**encoded_input)
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    similarity = F.cosine_similarity(sentence_embeddings[0], sentence_embeddings[1], dim=0).item()
    del model_output
    del sentence_embeddings
    torch.cuda.empty_cache()
    return similarity

def compute_similarity(target_review, agent_reviews_list):
    'Documentation.'
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


    sentences = [target_review] + list(agent_reviews_list)


    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
    if torch.cuda.is_available():
        encoded_input = {k: v.to('cuda') for k, v in encoded_input.items()}


    with torch.no_grad():
        model_output = model(**encoded_input)

    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)


    target_vec = sentence_embeddings[0:1]
    agent_vecs = sentence_embeddings[1:]


    similarities = F.cosine_similarity(target_vec, agent_vecs, dim=1)


    max_score = torch.max(similarities).item()


    del model_output
    del sentence_embeddings
    return max_score

def trucate_text(text, max_length):
    tokenized_memory = llama_tokenizer(text, return_tensors=None, truncation=True, max_length=max_length)
    truncated_memory_ids = tokenized_memory["input_ids"]
    memory_text_truncated = llama_tokenizer.decode(truncated_memory_ids, skip_special_tokens=True)
    return memory_text_truncated

args = parse_args()


PRODUCT_VECS = None
PRODUCT_VECS_T = None
PRODUCT_ASINS = None
ASIN_TO_IDX = None
if args.vectors_file and args.asins_file:
    print(f"[recommend-rrf] loading {args.vectors_file}")
    PRODUCT_VECS = np.load(args.vectors_file)
    if torch.cuda.is_available():
        PRODUCT_VECS_T = torch.from_numpy(PRODUCT_VECS).to('cuda')
    else:
        PRODUCT_VECS_T = torch.from_numpy(PRODUCT_VECS)
    with open(args.asins_file) as f:
        PRODUCT_ASINS = json.load(f)
    ASIN_TO_IDX = {a: i for i, a in enumerate(PRODUCT_ASINS)}
    print(f"[recommend-rrf] index ready | shape={PRODUCT_VECS.shape}")


def parse_seed_asins(raw_output):
    """Extract candidate ASINs from one model sample (string or list).

    Recommend outputs vary: sometimes a single ASIN, sometimes a comma-separated
    list. We take whatever the model emitted and split on commas.
    """
    if isinstance(raw_output, list):

        out = []
        for s in raw_output:
            out.extend([t.strip() for t in str(s).split(',') if t.strip()])
        return out
    return [t.strip() for t in str(raw_output).split(',') if t.strip()]


def retrieve_topk_for_seed(seed_asin, k):
    """Cosine retrieval against the product index. Returns list[asin] of length<=k.

    Excludes the seed itself (the seed ASIN already represents the prediction;
    we want neighbours to round out the top-10).
    """
    idx = ASIN_TO_IDX.get(seed_asin)
    if idx is None:
        return []
    q = PRODUCT_VECS_T[idx:idx + 1]
    sims = torch.mm(q, PRODUCT_VECS_T.t())[0]
    sims[idx] = -1e9
    top = torch.topk(sims, k=min(k, len(PRODUCT_ASINS)))
    return [PRODUCT_ASINS[i.item()] for i in top.indices]


def recommend_rrf(samples, target_asin, k_per_seed, k_rrf):
    """N samples -> per-seed retrieval -> RRF fusion -> rank score.

    Each sampled string can already contain >1 ASIN; we treat its first ASIN
    as the seed and ALSO inject the seed itself into the merged ranking at
    rank 0 (so a correct seed scores even if its neighbours don't surface).
    """
    rrf_scores = {}
    for sample in samples:
        seeds = parse_seed_asins(sample)
        if not seeds:
            continue
        seed = seeds[0]

        ranked = [seed] + [a for a in retrieve_topk_for_seed(seed, k_per_seed) if a != seed]
        for rank, asin in enumerate(ranked):
            rrf_scores[asin] = rrf_scores.get(asin, 0.0) + 1.0 / (k_rrf + rank + 1)

    if not rrf_scores:
        return 0.0
    fused = sorted(rrf_scores.items(), key=lambda x: -x[1])[:10]
    fused_asins = [a for a, _ in fused]
    if target_asin in fused_asins:
        rank = fused_asins.index(target_asin) + 1
        return 1 - (rank - 1) / 10
    return 0.0


tasks = json.load(open(args.task_file))
tool_input = json.load(open(args.param_file))
tool_selected = json.load(open(args.function_file))
all_products = json.load(open(args.all_products))

final_results = {'search':[], 'recommend':[], 'review':[]}
tool_accuracy = {'search':[], 'recommend':[], 'review':[]}

llama_tokenizer = AutoTokenizer.from_pretrained(os.getenv('PWAB_BASE_MODEL', 'meta-llama/Llama-2-7b-chat-hf'))

if args.evaluate_dpo == 'True':
    all_res = {}

    for task in tqdm(tasks['test']):
        task_type = task['type']
        instructions = task['task']
        target_asin = task['target']['product_info']['parent_asin']
        cur_res = {}
        if task_type == 'search':
            query = tool_input[instructions]
            for q in query:
                res = search_product_by_query(data={}, query=q)
                score = 0
                for i in range(len(res)):
                    if target_asin in res[i]:
                        score = 1 - i/len(res)
                        break
                cur_res[q] = score
        elif task_type == 'recommend':
            history = tool_input[instructions]
            for h in history:
                h_ = [item.strip() for item in h.split(',')]
                h_ = list(set(h_))
                res = get_recommendations_by_history(data={'all_products':all_products}, product_sequence=h_)
                score = 0
                for i in range(len(res)):
                    if target_asin in res[i]:
                        score = 1 - i/len(res)
                        break
                cur_res[h] = score
        else:
            review = tool_input[instructions]
            for r in review:
                target_review = task['target']['review']['text']
                agent_review = r
                similarity = compute_similarity(target_review, agent_review)
                cur_res[r] = similarity
        all_res[instructions] = cur_res
    with open(args.dpo_output, 'w') as f:
        json.dump(all_res, f, indent=2)
else:

    missing_count = 0
    total_lookup = 0
    missing_keys = []
    for task in tqdm(tasks['test']):

        tool = tool_selected[task['task']][0]
        if tool == 'search_product_by_query':
            task_type = 'search'
        elif tool == 'get_recommendations_by_history':
            task_type = 'recommend'
        else:
            task_type = 'review'
        gt_task_type = task['type']
        if task_type == gt_task_type:
            tool_accuracy[gt_task_type].append(1)
        else:
            tool_accuracy[gt_task_type].append(0)
            final_results[gt_task_type].append(0)
            continue
        instructions = task['task']
        target_asin = task['target']['product_info']['parent_asin']
        score = 0


        if task_type == 'search':
            total_lookup += 1


            if instructions not in tool_input:
                missing_count += 1
                missing_keys.append(instructions)
                final_results[gt_task_type].append(0)
                continue


            query_candidates = tool_input[instructions]


            if len(query_candidates) > 0:
                try:


                    res = search_product_by_query(
                        data={'instruction': instructions},
                        query=query_candidates
                    )
                except Exception as e:
                    print(f"Search failed: {e}")
                    res = []
            else:
                res = []


            score = 0

            for i in range(len(res)):
                if target_asin in res[i]:
                    score = 1 - i / len(res)
                    break


        elif task_type == 'recommend':
            total_lookup += 1


            if instructions not in tool_input:
                missing_count += 1
                missing_keys.append(instructions)
                final_results[gt_task_type].append(0)
                continue

            raw_output = tool_input[instructions]


            if PRODUCT_VECS_T is not None:

                if isinstance(raw_output, list):
                    samples = raw_output
                else:
                    samples = [raw_output]
                score = recommend_rrf(
                    samples,
                    target_asin,
                    k_per_seed=args.rrf_top_k,
                    k_rrf=args.rrf_constant,
                )
                final_results[gt_task_type].append(score)
                continue


            recommended_items = []
            if raw_output and isinstance(raw_output, list):

                if len(raw_output) == 1 and isinstance(raw_output[0], str) and ',' in raw_output[0]:
                    raw_string = raw_output[0]
                    recommended_items = [item.strip() for item in raw_string.split(',') if item.strip()]


                else:

                    recommended_items = [str(item).strip() for item in raw_output if item]


            score = 0
            if target_asin in recommended_items:

                rank_index = recommended_items.index(target_asin)
                rank = rank_index + 1


                if rank <= 10:
                    score = 1 - (rank - 1) / 10
                else:
                    score = 0


        else:
            total_lookup += 1
            if instructions not in tool_input:
                missing_count += 1
                missing_keys.append(instructions)
                final_results[gt_task_type].append(0)
                continue


            agent_reviews_list = tool_input[instructions]

            selected_review = ""

            if agent_reviews_list and len(agent_reviews_list) > 0:


                best_idx = get_consensus_best_index(agent_reviews_list, tokenizer, model)


                selected_review = agent_reviews_list[best_idx]


            else:

                final_results[gt_task_type].append(0)
                continue
            target_review = task['target']['review']['text']


            score = compute_similarity_origin(target_review, selected_review)


        final_results[gt_task_type].append(score)


    print(f"[STATS] 缺失的 tool_input 键: {missing_count} / {total_lookup}")
    if missing_count:
        print(f"[STATS] 缺失键示例: {missing_keys[:5]}")

    combined_data = [
        ['Search', len(final_results['search']), sum(tool_accuracy['search']) / len(tool_accuracy['search']), sum(final_results['search']) / len(final_results['search'])],
        ['Recommend', len(final_results['recommend']), sum(tool_accuracy['recommend']) / len(tool_accuracy['recommend']), sum(final_results['recommend']) / len(final_results['recommend'])],
        ['Review', len(final_results['review']), sum(tool_accuracy['review']) / len(tool_accuracy['review']), sum(final_results['review']) / len(final_results['review'])],
        ['Overall', len(final_results['search'] + final_results['recommend'] + final_results['review']),
        sum(tool_accuracy['search'] + tool_accuracy['recommend'] + tool_accuracy['review']) / len(tool_accuracy['search'] + tool_accuracy['recommend'] + tool_accuracy['review']),
        sum(final_results['search'] + final_results['recommend'] + final_results['review']) / len(final_results['search'] + final_results['recommend'] + final_results['review'])]
    ]

    headers = ['Task Type', 'Total', 'Tool Accuracy Avg', 'Result Avg']
    print(tabulate(combined_data, headers=headers, tablefmt='grid'))
