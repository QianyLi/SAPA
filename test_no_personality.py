import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import argparse
import os
# 确保这些函数的导入路径是正确的
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
    return parser.parse_args()

# --- 模型和 Tokenizer 的加载 (保持不变) ---
tokenizer = AutoTokenizer.from_pretrained(os.getenv('PWAB_SIM_MODEL', 'sentence-transformers/all-MiniLM-L6-v2'))
model = AutoModel.from_pretrained(os.getenv('PWAB_SIM_MODEL', 'sentence-transformers/all-MiniLM-L6-v2'))
if torch.cuda.is_available():
    model.to('cuda')

def compute_similarity(target_review, agent_review):
    # (这个函数保持不变)
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    sentences = [target_review, agent_review]
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
    if torch.cuda.is_available():
        encoded_input = encoded_input.to('cuda') # 确保输入数据在GPU上
    with torch.no_grad():
        model_output = model(**encoded_input)
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    similarity = F.cosine_similarity(sentence_embeddings[0], sentence_embeddings[1], dim=0).item()
    return similarity

# --- 加载数据文件 (保持不变) ---
args = parse_args()
tasks = json.load(open(args.task_file))
tool_input = json.load(open(args.param_file))
tool_selected = json.load(open(args.function_file))
all_products = json.load(open(args.all_products))

# --- 初始化结果字典 (保持不变) ---
final_results = {'search':[], 'recommend':[], 'review':[]}
tool_accuracy = {'search':[], 'recommend':[], 'review':[]}

# DPO评估部分也需要同样的修改
if args.evaluate_dpo == 'True':
    all_res = {}
    for task in tqdm(tasks['test']):
        # ==================== MODIFICATION START 1 ====================
        try:
            instructions = task['task']
            task_type = task['type']
            target_asin = task['target']['product_info']['parent_asin']
            cur_res = {}
            
            # 检查tool_input中是否存在该指令
            if instructions not in tool_input:
                continue # 如果不存在，直接跳过

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
            else: # review
                review = tool_input[instructions]
                for r in review:
                    target_review = task['target']['review']['text']
                    agent_review = r
                    similarity = compute_similarity(target_review, agent_review)
                    cur_res[r] = similarity
            all_res[instructions] = cur_res
        except KeyError:
            # 如果在处理过程中（例如查找target_asin时）出现KeyError，也跳过
            continue
        # ==================== MODIFICATION END 1 ======================
    with open(args.dpo_output, 'w') as f:
        json.dump(all_res, f, indent=2)

# 主要评估逻辑
else:
    for task in tqdm(tasks['test']):
        # ==================== MODIFICATION START 2 ====================
        # 使用 try...except 块来包裹对单个任务的所有处理
        try:
            instructions = task['task'] # 先获取指令/查询
            
            # 尝试从模型输出中获取工具和参数
            # 如果 instructions 这个 key 不存在，代码会直接跳转到 except 块
            tool = tool_selected[instructions][0]
            
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
                continue # 工具选择错误，这里逻辑不变

            target_asin = task['target']['product_info']['parent_asin']
            score = 0
            
            # 获取工具参数，这里是第二个可能出错的地方
            model_output_params = tool_input[instructions]

            if task_type == 'search':
                for q in model_output_params:
                    res = search_product_by_query(data={}, query=q)
                    for i in range(len(res)):
                        if target_asin in res[i]:
                            score = 1 - i/len(res)
                            break      
            elif task_type == 'recommend':
                for h in model_output_params:
                    h_ = [item.strip() for item in h.split(',')]
                    h_ = list(set(h_))
                    res = get_recommendations_by_history(data={'all_products':all_products}, product_sequence=h_)
                    for i in range(len(res)):
                        if target_asin in res[i]:
                            score = 1 - i/len(res)
                            break
                # for pred_list in model_output_params:
                #     # pred_list 就是你的 [LLM_Top1] + [Baseline_Top9]
                #     # 它应该是一个 ASIN 的列表，例如 ['B001', 'B002', ...]
                    
                #     # 容错处理：万一 pred_list 是字符串 (没解析好)，跳过
                #     if isinstance(pred_list, str):
                #         continue
                        
                #     # 直接在列表中查找 GT
                #     if target_asin in pred_list:
                #         rank = pred_list.index(target_asin)
                #         # PersonalWAB 的标准算分公式
                #         current_score = 1 - (rank / len(pred_list))
                        
                #         if current_score > score:
                #             score = current_score
                # pred_list = model_output_params
                
                # # 容错：确保它是列表
                # if isinstance(pred_list, list) and len(pred_list) > 0:
                #     # 如果列表里包含字符串，说明是正常的 ASIN 列表
                #     # (原代码的错误在于它想把 pred_list 当作 attempts 的列表去遍历)
                    
                #     if target_asin in pred_list:
                #         rank = pred_list.index(target_asin)
                #         # PersonalWAB 的标准算分公式
                #         current_score = 1 - (rank / len(pred_list))
                        
                #         if current_score > score:
                #             score = current_score
            else: # review
                for r in model_output_params:
                    target_review = task['target']['review']['text']
                    agent_review = r
                    similarity = compute_similarity(target_review, agent_review)
                    score = similarity
            
            final_results[gt_task_type].append(score)

        except KeyError:
            # 如果在 try 块中的任何地方（比如 tool = tool_selected[instructions][0] 或
            # model_output_params = tool_input[instructions]）发生了 KeyError，
            # 意味着模型没有为这个任务生成输出。
            # 我们就简单地跳过这个任务，进入下一次循环。
            continue
        # ==================== MODIFICATION END 2 ======================

    # --- 结果汇总和打印 (保持不变) ---
    # 为了避免除以零的错误，检查列表是否为空
    search_tool_acc = sum(tool_accuracy['search']) / len(tool_accuracy['search']) if tool_accuracy['search'] else 0
    search_res_avg = sum(final_results['search']) / len(final_results['search']) if final_results['search'] else 0
    
    rec_tool_acc = sum(tool_accuracy['recommend']) / len(tool_accuracy['recommend']) if tool_accuracy['recommend'] else 0
    rec_res_avg = sum(final_results['recommend']) / len(final_results['recommend']) if final_results['recommend'] else 0

    rev_tool_acc = sum(tool_accuracy['review']) / len(tool_accuracy['review']) if tool_accuracy['review'] else 0
    rev_res_avg = sum(final_results['review']) / len(final_results['review']) if final_results['review'] else 0
    
    overall_tool_acc_list = tool_accuracy['search'] + tool_accuracy['recommend'] + tool_accuracy['review']
    overall_res_list = final_results['search'] + final_results['recommend'] + final_results['review']
    
    overall_tool_acc = sum(overall_tool_acc_list) / len(overall_tool_acc_list) if overall_tool_acc_list else 0
    overall_res_avg = sum(overall_res_list) / len(overall_res_list) if overall_res_list else 0

    combined_data = [
        ['Search', len(final_results['search']), search_tool_acc, search_res_avg],
        ['Recommend', len(final_results['recommend']), rec_tool_acc, rec_res_avg],
        ['Review', len(final_results['review']), rev_tool_acc, rev_res_avg],
        ['Overall', len(overall_res_list), overall_tool_acc, overall_res_avg]
    ]

    headers = ['Task Type', 'Total Evaluated', 'Tool Accuracy Avg', 'Result Avg']
    print(tabulate(combined_data, headers=headers, tablefmt='grid'))
