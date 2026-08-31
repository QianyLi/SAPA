import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import argparse
import numpy as np
import os
import re
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
    parser.add_argument('--reflection_file', type=str, default='SAPA/output/full_reflection_log.json', help='详细反思日志')
    return parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(os.getenv('PWAB_SIM_MODEL', 'sentence-transformers/all-MiniLM-L6-v2'))
model = AutoModel.from_pretrained(os.getenv('PWAB_SIM_MODEL', 'sentence-transformers/all-MiniLM-L6-v2'))
if torch.cuda.is_available():
    model.to('cuda')

def compute_similarity_batch(target_review, agent_reviews_list):
    """Compute review similarities in batches."""
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
    return similarities.cpu().tolist()

args = parse_args()

tasks = json.load(open(args.task_file))
tool_input = json.load(open(args.param_file))
tool_selected = json.load(open(args.function_file))
all_products = json.load(open(args.all_products))

final_results = {'search':[], 'recommend':[], 'review':[]}
tool_accuracy = {'search':[], 'recommend':[], 'review':[]}
full_reflection_logs = []

for task in tqdm(tasks['test']):
    instruction = task['task']
    if instruction not in tool_selected or instruction not in tool_input:
        continue

    tool_pred = tool_selected[instruction][0]
    if 'search_product_by_query' in tool_pred:
        task_type = 'search'
    elif 'get_recommendations_by_history' in tool_pred:
        task_type = 'recommend'
    else:
        task_type = 'review'

    gt_task_type = task['type']

    is_tool_correct = 1 if task_type == gt_task_type else 0
    tool_accuracy[gt_task_type].append(is_tool_correct)

    if not is_tool_correct:
        final_results[gt_task_type].append(0)
        continue

    candidates = tool_input[instruction]
    target_asin = task['target'].get('product_info', {}).get('parent_asin', "")

    task_log = {
        "instruction": instruction,
        "task_type": gt_task_type,
        "target": "",
        "candidates_analysis": []
    }
    scores = []

    if gt_task_type == 'search':
        task_log["target"] = target_asin
        for idx, q in enumerate(candidates):
            res = search_product_by_query(data={}, query=q)
            s = 0
            for i in range(len(res)):
                if target_asin in res[i]:
                    s = 1 - i/len(res)
                    break
            scores.append(s)
            task_log["candidates_analysis"].append({"rank": idx, "text": q, "score": s})

    elif gt_task_type == 'recommend':
        task_log["target"] = target_asin
        for idx, h in enumerate(candidates):
            h_ = [item.strip() for item in h.split(',')]
            h_ = list(set(h_))
            res = get_recommendations_by_history(data={'all_products':all_products}, product_sequence=h_)
            s = 0
            for i in range(len(res)):
                if target_asin in res[i]:
                    s = 1 - i/len(res)
                    break
            scores.append(s)
            task_log["candidates_analysis"].append({"rank": idx, "text": h, "score": s})

    else:
        target_review = task['target']['review']['text']
        task_log["target"] = target_review
        scores = compute_similarity_batch(target_review, candidates)
        for idx, s in enumerate(scores):
            task_log["candidates_analysis"].append({"rank": idx, "text": candidates[idx], "score": s})

    final_results[gt_task_type].append(max(scores) if scores else 0)
    full_reflection_logs.append(task_log)


os.makedirs(os.path.dirname(args.reflection_file), exist_ok=True)
with open(args.reflection_file, 'w', encoding='utf-8') as f:
    json.dump(full_reflection_logs, f, indent=4, ensure_ascii=False)

rank_scores = np.zeros(10)
rank_wins = np.zeros(10)
for log in full_reflection_logs:
    s_list = [c['score'] for c in log['candidates_analysis']]
    if not s_list: continue
    s_arr = np.array(s_list[:10] + [0]*(10-len(s_list)))
    rank_scores += s_arr
    rank_wins[np.argmax(s_arr)] += 1

total = len(full_reflection_logs)

print("\n" + "="*60)
print(f"📊 10条候选结果 Rank 质量统计 (样本数: {total})")
print("="*60)
dist_table = []
for i in range(10):
    dist_table.append([f"Rank {i}", f"{rank_scores[i]/total:.4f}", f"{(rank_wins[i]/total)*100:.2f}%"])
print(tabulate(dist_table, headers=['排名', '平均分', '最优命中率'], tablefmt='grid'))

combined_data = [
    ['Search', len(final_results['search']), np.mean(tool_accuracy['search']), np.mean(final_results['search'])],
    ['Recommend', len(final_results['recommend']), np.mean(tool_accuracy['recommend']), np.mean(final_results['recommend'])],
    ['Review', len(final_results['review']), np.mean(tool_accuracy['review']), np.mean(final_results['review'])]
]
print("\n" + tabulate(combined_data, headers=['任务', '数量', '工具准确率', '结果平均分 (Best-of-10)'], tablefmt='grid'))

print(f"\n✅ 详细反思日志已保存至: {args.reflection_file}")
