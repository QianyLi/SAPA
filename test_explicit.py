import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import argparse
import os
from tabulate import tabulate

# 请确保以下路径指向你项目中实际的文件位置
from PersonalWAB.envs.pwab.functions.get_recommendations_by_history import get_recommendations_by_history
from PersonalWAB.envs.pwab.functions.search_product_by_query import search_product_by_query

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate PWA tasks")
    parser.add_argument('--evaluate_dpo', type=str, default='False', help='Whether to evaluate DPO')
    parser.add_argument('--task_file', type=str, default='SAPA/data/user_instructions.json', help='Path to task file')
    parser.add_argument('--param_file', type=str, default='SAPA/output/res/tool_input.json', help='Path to tool input file')
    parser.add_argument('--function_file', type=str, default='SAPA/output/tool_selected.json', help='Path to tool selected file')
    parser.add_argument('--all_products', type=str, default='SAPA/data/all_products.json', help='Path to all products file')
    parser.add_argument('--dpo_output', type=str, default='SAPA/data/dpo_data.json', help='Path to DPO output file')
    
    # [新增] 日志保存路径与详细模式开关
    parser.add_argument('--log_output', type=str, default='evaluation_analysis.json', help='Path to save detailed evaluation logs')
    parser.add_argument('--verbose', type=bool, default=True, help='Print detailed logs for each task')
    return parser.parse_args()

# ================= Global Setup =================
args = parse_args()

print("Loading Models...")
# 加载用于计算相似度的 Sentence-BERT
tokenizer = AutoTokenizer.from_pretrained(os.getenv('PWAB_SIM_MODEL', 'sentence-transformers/all-MiniLM-L6-v2'))
model = AutoModel.from_pretrained(os.getenv('PWAB_SIM_MODEL', 'sentence-transformers/all-MiniLM-L6-v2'))
if torch.cuda.is_available():
    model.to('cuda')

# 加载用于截断或其他用途的 Llama Tokenizer (保留原逻辑)
llama_tokenizer = AutoTokenizer.from_pretrained(os.getenv('PWAB_BASE_MODEL', 'meta-llama/Llama-2-7b-chat-hf'))

print("Loading Data...")
tasks = json.load(open(args.task_file))
tool_input = json.load(open(args.param_file))
tool_selected = json.load(open(args.function_file))
all_products = json.load(open(args.all_products))

# 结果容器
final_results = {'search':[], 'recommend':[], 'review':[]}
tool_accuracy = {'search':[], 'recommend':[], 'review':[]}
analysis_records = [] # 用于保存详细 JSON 日志

# ================= Helper Functions =================

def compute_similarity(target_review, agent_review):
    """计算两个文本的余弦相似度"""
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
    
    # 清理显存
    del model_output
    del sentence_embeddings
    # torch.cuda.empty_cache() # 频繁清空可能会变慢，可视情况开启
    return similarity

def safe_avg(lst):
    """安全计算平均值，防止除以零"""
    return sum(lst) / len(lst) if lst else 0.0

# ================= Main Evaluation Loop =================

if args.evaluate_dpo == 'True':
    # DPO 评测逻辑保持不变
    all_res = {}
    print("Starting DPO Evaluation...")
    for task in tqdm(tasks['test']):
        task_type = task['type']
        instructions = task['task']
        target_asin = task['target']['product_info']['parent_asin']
        cur_res = {}
        if task_type == 'search':
            query = tool_input.get(instructions, [])
            for q in query:
                res = search_product_by_query(data={}, query=q)
                score = 0
                for i in range(len(res)):
                    if target_asin in res[i]:
                        score = 1 - i/len(res)
                        break      
                cur_res[q] = score
        elif task_type == 'recommend':
            history = tool_input.get(instructions, [])
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
            review = tool_input.get(instructions, [])
            for r in review:
                target_review = task['target']['review']['text']
                agent_review = r
                similarity = compute_similarity(target_review, agent_review)
                cur_res[r] = similarity
        all_res[instructions] = cur_res
    
    with open(args.dpo_output, 'w') as f:
        json.dump(all_res, f, indent=2)
    print(f"DPO evaluation saved to {args.dpo_output}")

else:
    # 标准评测逻辑 (Standard Evaluation)
    print(f"\nStarting Evaluation... Logs will be saved to: {args.log_output}")
    
    for idx, task in enumerate(tqdm(tasks['test'])):
        instructions = task['task']
        target_info = task.get('target', {})
        
        # 1. 基础信息提取与校验
        # 容错：如果指令不在 tool_selected 中，说明该样本生成失败或数据不一致
        if instructions not in tool_selected:
            continue
            
        tool = tool_selected[instructions][0]
        
        # 映射工具名称到任务类型
        if tool == 'search_product_by_query':
            task_type = 'search'
        elif tool == 'get_recommendations_by_history':
            task_type = 'recommend'
        else:
            task_type = 'review'
            
        gt_task_type = task['type']
        target_asin = target_info.get('product_info', {}).get('parent_asin', 'N/A')
        target_review_text = target_info.get('review', {}).get('text', 'N/A')
        
        # 初始化当前 Case 的记录
        case_record = {
            "id": idx,
            "instruction": instructions,
            "gt_type": gt_task_type,
            "pred_type": task_type,
            "target_asin": target_asin,
            "target_review": target_review_text if gt_task_type == 'review' else "N/A",
            "score": 0.0,
            "generated_content": None,   # 模型生成的参数 (query, history string, review text)
            "retrieved_results": [],     # [新增] 工具实际召回的商品列表
            "status": "Unknown"
        }

        # 2. 检查工具选择准确性 (Tool Accuracy)
        is_tool_correct = (task_type == gt_task_type)
        if is_tool_correct:
            tool_accuracy[gt_task_type].append(1)
        else:
            tool_accuracy[gt_task_type].append(0)
            final_results[gt_task_type].append(0)
            
            # 记录错误
            case_record["status"] = "Tool Mismatch"
            case_record["score"] = 0.0
            analysis_records.append(case_record)
            
            if args.verbose:
                print(f"\n[Case {idx}] ❌ Tool Mismatch: GT={gt_task_type} vs Pred={task_type}")
            continue

        # 3. 执行任务并计算得分
        score = 0
        generated_content = ""
        tool_retrieved_results = [] # 暂存召回列表
        
        if task_type == 'search':
            query_list = tool_input.get(instructions, [])
            generated_content = query_list
            
            if query_list:
                for q in query_list:
                    res = search_product_by_query(data={}, query=q)
                    
                    # [关键] 记录这一轮搜索的结果
                    tool_retrieved_results = res 
                    
                    for i in range(len(res)):
                        # 核心打分逻辑：Rank Decay
                        if target_asin in res[i]:
                            score = 1 - i/len(res)
                            break
                    if score > 0: break

        elif task_type == 'recommend':
            history_list = tool_input.get(instructions, [])
            generated_content = history_list
            
            if history_list:
                for h in history_list:
                    h_ = [item.strip() for item in h.split(',')]
                    h_ = list(set(h_))
                    
                    res = get_recommendations_by_history(data={'all_products':all_products}, product_sequence=h_)
                    
                    # [关键] 记录这一轮推荐的结果
                    tool_retrieved_results = res
                    
                    for i in range(len(res)):
                        if target_asin in res[i]:
                            score = 1 - i/len(res)
                            break
                    if score > 0: break

        else: # Review
            review_list = tool_input.get(instructions, [])
            generated_content = review_list
            tool_retrieved_results = "N/A (Review Task)"
            
            if review_list:
                for r in review_list:
                    similarity = compute_similarity(target_review_text, r)
                    score = similarity
                    # Review 只要有一个生成通常就算分，或者取最大值，这里保持覆盖逻辑
        
        final_results[gt_task_type].append(score)
        
        # 4. 完善并保存 Log
        case_record["score"] = score
        case_record["generated_content"] = generated_content
        case_record["retrieved_results"] = tool_retrieved_results # 将具体商品列表写入 JSON
        case_record["status"] = "Hit" if score > 0 else "Miss"
        
        analysis_records.append(case_record)

        # 5. 控制台打印 (Verbose Mode)
        if args.verbose:
            status_icon = "✅" if score > 0 else "❌"
            if score == 0:
                # 只打印 Miss 的情况以减少刷屏，或者全部打印
                # print(f"[Case {idx}] {status_icon} Score: 0.0 | Type: {task_type}")
                pass
            else:
                # print(f"[Case {idx}] {status_icon} Score: {score:.2f} | Type: {task_type}")
                pass

    # ================= Reporting =================
    
    combined_data = [
        ['Search', len(final_results['search']), safe_avg(tool_accuracy['search']), safe_avg(final_results['search'])],
        ['Recommend', len(final_results['recommend']), safe_avg(tool_accuracy['recommend']), safe_avg(final_results['recommend'])],
        ['Review', len(final_results['review']), safe_avg(tool_accuracy['review']), safe_avg(final_results['review'])],
        ['Overall', 
         len(final_results['search'] + final_results['recommend'] + final_results['review']),
         safe_avg(tool_accuracy['search'] + tool_accuracy['recommend'] + tool_accuracy['review']),
         safe_avg(final_results['search'] + final_results['recommend'] + final_results['review'])]
    ]

    headers = ['Task Type', 'Total', 'Tool Acc', 'Result Score Avg']
    
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(tabulate(combined_data, headers=headers, tablefmt='grid'))
    print("="*60)
    
    # 保存 JSON 文件
    print(f"Saving detailed logs to {args.log_output} ...")
    output_dir = os.path.dirname(args.log_output)
    
    # 2. 如果路径包含文件夹（不是当前目录），且文件夹不存在，则创建
    if output_dir and not os.path.exists(output_dir):
        print(f"Warning: Directory '{output_dir}' does not exist. Creating it now...")
        os.makedirs(output_dir, exist_ok=True) # exist_ok=True 防止并发创建时报错
        
    # 3. 写入文件 ('w'模式会自动创建文件，如果文件已存在则覆盖)
    try:
        with open(args.log_output, 'w', encoding='utf-8') as f:
            json.dump(analysis_records, f, indent=2, ensure_ascii=False)
        print("Done! Evaluation log saved successfully.")
    except Exception as e:
        print(f"Error saving file: {e}")
