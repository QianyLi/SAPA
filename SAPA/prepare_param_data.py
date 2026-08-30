import json
import os
import random
import argparse
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

# ==============================================================================
# 假设 utils.py 在同一目录下
# ==============================================================================
from utils import (
    retrieve_top_k_memories,        # Search/Review
    retrieve_top_k_memories_score,  # Recommend History
    build_taskspe_memory,           # 【关键】格式化函数
    generate_search_query,          # Search Target Gen
    prettify_product_info,          # Review Info Prettify
    PARAM_PROMPT_SEARCH,            
    PARAM_PROMPT_RECOMMEND,   
    PARAM_PROMPT_REVIEW,
    # LLAMA3_PROMPT_GENERATE,
    # LLAMA3_PROMPT_RECOMMEND             
)

# ==============================================================================
# 1. 辅助工具函数
# ==============================================================================
def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def save_checkpoint(data, ids, data_path, ids_path):
    try:
        # 原子化保存：先写临时文件再重命名，防止中断导致文件损坏
        with open(data_path + '.tmp', 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
        os.replace(data_path + '.tmp', data_path)
        
        with open(ids_path + '.tmp', 'w', encoding='utf-8') as f:
            json.dump(list(ids), f)
        os.replace(ids_path + '.tmp', ids_path)
    except Exception as e:
        print(f"Error saving checkpoint: {e}")

def estimate_tokens(text):
    """粗略估算 Token 数"""
    if not text: return 0
    return len(text) / 3.5

def format_memory_code4_style(history_items, task_type):
    """Search/Review 专用格式化"""
    mem_strs = []
    if not history_items: return []
    for item in history_items:
        if task_type == 'search':
            info = item.get('product_info', {})
            title = info.get('title', 'N/A')
            category = info.get('main_category', 'N/A')
            price = info.get('price', 'N/A')
            store = info.get('store', 'N/A')
            mem_strs.append(f"Title:{title}\nMain Category:{category}\nPrice:{price}\nStore:{store}")
        elif task_type == 'review':
            review = item.get('review', {})
            rating = review.get('rating', 'N/A')
            text = review.get('text', '').replace('\n', ' ')
            mem_strs.append(f"Rating:{rating}\nText:{text}")
    return mem_strs

# ==============================================================================
# 2. 参数解析
# ==============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description='SFT Data Gen')
    parser.add_argument('--instruction_file', type=str, required=True)
    parser.add_argument('--retrieval_results_file', type=str, required=True)
    parser.add_argument('--history_file', type=str, required=True)
    parser.add_argument('--product_data_file', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    
    parser.add_argument('--llama_tokenizer_path', type=str, default='meta-llama/Llama-2-7b-chat-hf')
    parser.add_argument('--sim_model_path', type=str, default='sentence-transformers/all-MiniLM-L6-v2')
    
    parser.add_argument('--mem_token_length', type=int, default=768)
    parser.add_argument('--mem_length', type=int, default=100)
    
    # 方案二关键参数
    parser.add_argument('--candidate_pool_size', type=int, default=10)
    parser.add_argument('--max_candidate_token_budget', type=int, default=3000)
    parser.add_argument('--similarity_threshold', type=float, default=0.5)
    
    parser.add_argument('--save_interval', type=int, default=50)
    return parser.parse_args()

# ==============================================================================
# 3. 主逻辑
# ==============================================================================
def main():
    set_seed(42)
    args = parse_args()

    # 初始化 Checkpoint
    checkpoint_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    base_name = os.path.basename(args.output_file).replace('.json', '')
    data_checkpoint_path = os.path.join(checkpoint_dir, f'{base_name}_checkpoint.json')
    ids_checkpoint_path = os.path.join(checkpoint_dir, f'{base_name}_ids.json')

    llama_data = {'train': [], 'test': []}
    processed_task_ids = set()

    # 容错加载
    if os.path.exists(data_checkpoint_path):
        print(f"Loading checkpoint: {data_checkpoint_path}")
        try:
            with open(data_checkpoint_path, 'r', encoding='utf-8') as f:
                llama_data = json.load(f)
        except json.JSONDecodeError:
            print("Checkpoint corrupted, starting fresh.")
            
    if os.path.exists(ids_checkpoint_path):
        try:
            with open(ids_checkpoint_path, 'r', encoding='utf-8') as f:
                processed_task_ids = set(json.load(f))
        except json.JSONDecodeError:
            processed_task_ids = set()

    # 加载资源
    print("Loading resources...")
    llama_tokenizer = AutoTokenizer.from_pretrained(args.llama_tokenizer_path)
    sim_tokenizer = AutoTokenizer.from_pretrained(args.sim_model_path)
    sim_model = AutoModel.from_pretrained(args.sim_model_path)
    if torch.cuda.is_available():
        sim_model = sim_model.to('cuda')
    sim_model.eval()

    instructions = json.load(open(args.instruction_file, 'r'))
    product_database = json.load(open(args.product_data_file, 'r'))
    with open(args.history_file, 'r') as f:
        global_user_history = json.load(f)

    print("Indexing retrieval results...")
    retrieval_map = {}
    with open(args.retrieval_results_file, 'r') as f:
        for line in tqdm(f, desc="Indexing"):
            try:
                record = json.loads(line)
                retrieval_map[record['task_id']] = record
            except: pass

    newly_processed_count = 0
    discarded_count = 0

    try:
        for split, tasks in instructions.items():
            if split not in llama_data: llama_data[split] = []
            
            print(f"\nProcessing {split} set...")
            for task in tqdm(tasks):
                user_id = task['user_id']
                timestamp = task['timestamp']
                task_id_key = f"{user_id}_{timestamp}"            # for retrieval_map lookup
                dedup_key   = f"{user_id}_{timestamp}_{task['type']}"  # for processed-task tracking
                input_text = task['task']
                task_type = task['type']
                target_info = task.get('target', {})

                if dedup_key in processed_task_ids: continue

                full_prompt = ""
                target = ""
                mem_log = ""

                # ============================================================
                # 分支 A: Recommend 任务 (修复版：方案二 + 注入 Product Info)
                # ============================================================
                if task_type == 'recommend':
                    retrieved_data = retrieval_map.get(task_id_key)
                    ground_truth_asin = target_info.get('product_info', {}).get('parent_asin')

                    if not retrieved_data or not ground_truth_asin:
                        discarded_count += 1
                        continue

                    full_top_500 = retrieved_data.get('search_result', {}).get('top_500_results', [])
                    
                    # 1. 寻找 GT 对象
                    gt_item_obj = next((x for x in full_top_500 if x['asin'] == ground_truth_asin), None)
                    if not gt_item_obj:
                        p_info = product_database.get(ground_truth_asin)
                        if p_info:
                            gt_item_obj = {"asin": ground_truth_asin, "product_info": p_info}
                        else:
                            discarded_count += 1
                            continue
                    else:
                        # 【修复1】如果从 top500 拿到的对象没有 product_info，手动注入
                        if 'product_info' not in gt_item_obj:
                            gt_item_obj['product_info'] = product_database.get(ground_truth_asin)

                    # 2. 确定初始候选池 (Raw Pool)
                    raw_pool = []
                    target_k = args.candidate_pool_size

                    if split == 'train':
                        # Train: 1 GT + (K-1) Random
                        raw_pool.append(gt_item_obj)
                        top_100 = full_top_500[:50]
                        negatives = [x for x in top_100 if x['asin'] != ground_truth_asin]
                        needed = target_k - 1
                        selected_negatives = random.sample(negatives, min(len(negatives), needed))
                        raw_pool.extend(selected_negatives)
                    else:
                        # Test: Top K
                        raw_pool = full_top_500[:target_k]
                        if not raw_pool:
                            discarded_count += 1
                            continue

                    # 3. 动态截断 & 注入详情 (Enrich & Prune)
                    final_display_list = []
                    current_tokens = 0
                    
                    # 3.1 Train集 优先处理 GT
                    pool_to_iterate = raw_pool
                    if split == 'train':
                        # GT 已经在上面注入了 info，可以直接用
                        gt_str_list = build_taskspe_memory([gt_item_obj], task_type)
                        if gt_str_list:
                            gt_str = gt_str_list[0]
                            gt_cost = estimate_tokens(gt_str) + 10
                            final_display_list.append(gt_item_obj)
                            current_tokens += gt_cost
                            pool_to_iterate = [x for x in raw_pool if x['asin'] != ground_truth_asin]

                    # 3.2 遍历填充剩余
                    for item in pool_to_iterate:
                        asin = item['asin']
                        
                        # 【修复2】核心逻辑：必须从 DB 查详情，构造 rich item
                        p_info = product_database.get(asin)
                        if not p_info: 
                            continue # 没详情的跳过
                        
                        rich_item = {'asin': asin, 'product_info': p_info}
                        
                        # 格式化
                        item_str_list = build_taskspe_memory([rich_item], task_type)
                        if not item_str_list: continue
                        
                        cost = estimate_tokens(item_str_list[0]) + 10
                        if current_tokens + cost > args.max_candidate_token_budget:
                            break
                        
                        final_display_list.append(rich_item)
                        current_tokens += cost

                    # 4. 生成 Prompt (Shuffle)
                    prompt_list_shuffled = final_display_list[:]
                    random.shuffle(prompt_list_shuffled)
                    
                    # 此时 prompt_list_shuffled 里的元素都有 product_info 了，可以直接传
                    formatted_contents = build_taskspe_memory(prompt_list_shuffled, task_type)
                    
                    if not formatted_contents: # 双重保险
                        discarded_count += 1
                        continue

                    candidates_str_lines = []
                    for idx, content_str in enumerate(formatted_contents):
                        candidates_str_lines.append(f"[{idx+1}] {content_str}")
                    candidates_str = "\n".join(candidates_str_lines)
                    #  candidates_str = '' # 测试

                    # 5. 构建 Target (有序)
                    if split == 'train':
                        # target_asins = [ground_truth_asin]
                        # available_asins = set(item['asin'] for item in final_display_list)
                        
                        # for item in full_top_500:
                        #     asin = item['asin']
                        #     if asin == ground_truth_asin: continue
                        #     if asin in available_asins:
                        #         target_asins.append(asin)
                        #     if len(target_asins) >= 10: break
                        # target = ", ".join(target_asins)
                        target = ground_truth_asin
                    else:
                        target = ground_truth_asin

                    # 6. History
                    raw_user_hist = global_user_history.get(user_id, [])
                    time_valid_hist = [h for h in raw_user_hist if h.get('review', {}).get('timestamp', 0) < timestamp]
                    # final_history_str = ''
                    final_history_str = "No purchase history available."
                    
                    # if time_valid_hist:
                    #     hist_cand_strs = build_taskspe_memory(time_valid_hist, task_type)
                    #     if hist_cand_strs:
                    #         retrieval_res = retrieve_top_k_memories_score(input_text, hist_cand_strs, sim_model, sim_tokenizer, k=1)
                    #         if retrieval_res and retrieval_res[0][1] >= args.similarity_threshold:
                    #             final_history_str = f"- {retrieval_res[0][0]}"

                    full_prompt = PARAM_PROMPT_RECOMMEND.replace('<Instruction>', input_text) \
                                                        .replace('<History>', final_history_str) \
                                                        .replace('<Candidates>', candidates_str) \
                                                        .replace('<Tool>', 'get_recommendations_by_history')
                    
                    # full_prompt = LLAMA3_PROMPT_RECOMMEND.replace('<Instruction>', input_text) \
                    #                                     .replace('<History>', final_history_str) \
                    #                                     .replace('<Candidates>', candidates_str) \
                    #                                     .replace('<Tool>', 'get_recommendations_by_history')
                    mem_log = f"Rec_Candidates: {len(prompt_list_shuffled)}"

                # ================================================================
                # 分支 B: Search / Review (保持 Code 4 原始逻辑)
                # ================================================================
                else:
                    raw_user_hist = global_user_history.get(user_id, [])
                    ori_history = [item for item in raw_user_hist if item['review']["timestamp"] < timestamp]
                    history_items = []
                    if ori_history:
                        history_items = retrieve_top_k_memories(input_text, ori_history, sim_model, sim_tokenizer, k=6) if task_type == 'search' \
                                        else retrieve_top_k_memories(input_text, ori_history, sim_model, sim_tokenizer, k=args.mem_length)
                    
                    mem_strs = format_memory_code4_style(history_items, task_type)
                    
                    memory_text_raw = ' | '.join(mem_strs) if mem_strs else "No purchase history."
                    tokenized_memory = llama_tokenizer(memory_text_raw, return_tensors=None, truncation=True, max_length=args.mem_token_length)
                    memory_text_truncated = llama_tokenizer.decode(tokenized_memory["input_ids"], skip_special_tokens=True)
                    # memory_text_truncated = ''
                    
                    tool_text = 'search_product_by_query' if task_type == 'search' else 'add_product_review'
                    base_prompt = PARAM_PROMPT_SEARCH if task_type == 'search' else PARAM_PROMPT_REVIEW
                    # base_prompt = LLAMA3_PROMPT_GENERATE
                    
                    current_instruction = input_text
                    if task_type == 'review':
                        current_instruction += prettify_product_info(target_info.get('product_info'))

                    temp_prompt = base_prompt.replace('<Instruction>', current_instruction) \
                                             .replace('<Tool>', tool_text)
                    
                    if '<History>' in base_prompt:
                        full_prompt = temp_prompt.replace('<History>', memory_text_truncated)
                    else:
                        full_prompt = temp_prompt.replace('<Memory>', memory_text_truncated)

                    if task_type == 'search':
                        if split == 'train':
                            target = generate_search_query(input_text, mem_strs)
                        else:
                            target = target_info.get('product_info', {}).get('title', '')
                    else:
                        target = target_info.get('review', {}).get('text', '')
                    
                    mem_log = "Search/Review Code4 Logic"

                # 保存
                if full_prompt and target:
                    llama_data[split].append({
                        'instruction': input_text,
                        'prompt': full_prompt,
                        'target': target,
                        'mem': mem_log
                    })
                    processed_task_ids.add(dedup_key)
                    newly_processed_count += 1
                
                if newly_processed_count % args.save_interval == 0:
                    save_checkpoint(llama_data, processed_task_ids, data_checkpoint_path, ids_checkpoint_path)

    except KeyboardInterrupt:
        print("Interrupted...")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        save_checkpoint(llama_data, processed_task_ids, data_checkpoint_path, ids_checkpoint_path)
        print(f"Final Save. Success: {len(processed_task_ids)}, Discarded: {discarded_count}")
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(llama_data, f, indent=4)

if __name__ == '__main__':
    main()
