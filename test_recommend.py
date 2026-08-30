import json
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import argparse
import os
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
    return parser.parse_args()

# 初始化模型
tokenizer = AutoTokenizer.from_pretrained(os.getenv('PWAB_SIM_MODEL', 'sentence-transformers/all-MiniLM-L6-v2'))
model = AutoModel.from_pretrained(os.getenv('PWAB_SIM_MODEL', 'sentence-transformers/all-MiniLM-L6-v2'))
if torch.cuda.is_available():
    model.to('cuda')

def compute_similarity(target_review, agent_review):
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
    sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
    similarity = torch.nn.functional.cosine_similarity(sentence_embeddings[0], sentence_embeddings[1], dim=0).item()
    return similarity

args = parse_args()

tasks = json.load(open(args.task_file))
tool_input = json.load(open(args.param_file))
tool_selected = json.load(open(args.function_file))
# all_products = json.load(open(args.all_products)) # 如果不用详情可注释

final_results = {'search':[], 'recommend':[], 'review':[]}
tool_accuracy = {'search':[], 'recommend':[], 'review':[]}

# === 核心诊断计数器 ===
rec_diagnostics = {
    'total': 0,
    'recall_hits': 0,      # 召回命中：Target 在 10 个候选里
    'llm_top1_hits': 0,    # LLM 命中：Target 在第 1 位
    'rank_scores': []      # 你的指标：1 - i/N
}

data_source = tasks['test'] if 'test' in tasks else tasks['train']

for task in tqdm(data_source):
    if task['task'] not in tool_selected:
        continue
        
    tool = tool_selected[task['task']][0]
    
    if tool == 'search_product_by_query':
        task_type = 'search'
    elif tool == 'get_recommendations_by_history':
        task_type = 'recommend'
    else:
        task_type = 'review'
        
    gt_task_type = task['type']
    
    # 1. 工具选择准确率
    if task_type == gt_task_type:
        tool_accuracy[gt_task_type].append(1)
    else:
        tool_accuracy[gt_task_type].append(0)
        final_results[gt_task_type].append(0)
        continue
        
    instructions = task['task']
    target_asin = task['target']['product_info']['parent_asin']
    score = 0
    
    # --- Search Task ---
    if task_type == 'search':
        query = tool_input.get(instructions, [])
        for q in query:
            res = search_product_by_query(data={}, query=q)
            for i in range(len(res)):
                if target_asin in res[i]:
                    curr = 1 - i/len(res)
                    if curr > score: score = curr
                    break      
    
    # --- Recommend Task (核心诊断部分) ---
    elif task_type == 'recommend':
        # 获取 LLM 重排后的列表 (通常是 10 个 ASIN)
        # 因为是你提供的 similarity top 10，所以只要 target 在这里面，就算召回成功
        res = tool_input.get(instructions, [])
        if not isinstance(res, list): res = [res]
        
        rec_diagnostics['total'] += 1
        is_retrieved = False
        
        for i in range(len(res)):
            # 检查 target 是否在列表中
            if target_asin.strip() == str(res[i]).strip():
                is_retrieved = True
                
                # 计算你的 Rank 分数
                score = 1 - i/len(res)
                rec_diagnostics['rank_scores'].append(score)
                
                # 检查是否排在第一位 (LLM 选择成功)
                if i == 0:
                    rec_diagnostics['llm_top1_hits'] += 1
                break
        
        if is_retrieved:
            rec_diagnostics['recall_hits'] += 1
        else:
            # 没召回，分数为0
            rec_diagnostics['rank_scores'].append(0)
            score = 0

    # --- Review Task ---
    else:
        review = tool_input.get(instructions, [])
        for r in review:
            target_review = task['target']['review']['text']
            agent_review = r
            similarity = compute_similarity(target_review, agent_review)
            if similarity > score: score = similarity
                
    final_results[gt_task_type].append(score)

# --- 打印总体结果 ---
print("\n" + "="*30 + " Overall Performance " + "="*30)
combined_data = [
    ['Search', len(final_results['search']), 
     sum(tool_accuracy['search']) / len(tool_accuracy['search']) if tool_accuracy['search'] else 0, 
     sum(final_results['search']) / len(final_results['search']) if final_results['search'] else 0],
    
    ['Recommend', len(final_results['recommend']), 
     sum(tool_accuracy['recommend']) / len(tool_accuracy['recommend']) if tool_accuracy['recommend'] else 0, 
     sum(final_results['recommend']) / len(final_results['recommend']) if final_results['recommend'] else 0],
    
    ['Review', len(final_results['review']), 
     sum(tool_accuracy['review']) / len(tool_accuracy['review']) if tool_accuracy['review'] else 0, 
     sum(final_results['review']) / len(final_results['review']) if final_results['review'] else 0],
    
    ['Overall', len(final_results['search'] + final_results['recommend'] + final_results['review']),
     sum(tool_accuracy['search'] + tool_accuracy['recommend'] + tool_accuracy['review']) / len(tool_accuracy['search'] + tool_accuracy['recommend'] + tool_accuracy['review']) if (tool_accuracy['search'] + tool_accuracy['recommend'] + tool_accuracy['review']) else 0,
     sum(final_results['search'] + final_results['recommend'] + final_results['review']) / len(final_results['search'] + final_results['recommend'] + final_results['review']) if (final_results['search'] + final_results['recommend'] + final_results['review']) else 0]
]
print(tabulate(combined_data, headers=['Task Type', 'Total', 'Tool Accuracy', 'Result Score'], tablefmt='grid'))

# --- 打印 Recommend 详细诊断 ---
print("\n" + "="*30 + " Recommend Diagnostics (Test Set) " + "="*30)

total_rec = rec_diagnostics['total']
recall_hits = rec_diagnostics['recall_hits']
llm_hits = rec_diagnostics['llm_top1_hits']

# 1. 真实召回率 (Test Recall)
test_recall = recall_hits / total_rec if total_rec > 0 else 0

# 2. LLM 选择准确率 (只有在召回成功的基础上看)
# 含义：如果正确答案在列表里，LLM 把它排在第一位的概率
llm_conditional_acc = llm_hits / recall_hits if recall_hits > 0 else 0

# 3. 总体 Top-1 准确率 (端到端)
overall_top1 = llm_hits / total_rec if total_rec > 0 else 0

diag_data = [
    ['Test Set Recall', f"{test_recall:.2%}", "Similarity Top-10 包含 Target 的比例 (召回层能力)"],
    ['LLM Top-1 Selection', f"{llm_conditional_acc:.2%}", "在包含 Target 的列表中，LLM 把它排在第一位的比例 (LLM 能力)"],
    ['End-to-End Top-1', f"{overall_top1:.2%}", "Target 最终被排在第一位的总比例"]
]
print(tabulate(diag_data, headers=['Metric', 'Value', 'Meaning'], tablefmt='fancy_grid'))
