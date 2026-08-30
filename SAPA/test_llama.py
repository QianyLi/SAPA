from transformers import GenerationConfig
from transformers import LlamaTokenizer, LlamaForCausalLM
import torch
import os
from tqdm import tqdm
import argparse
from peft import PeftModel
from utils import load_param_prompt_beam_search, load_function_prompt, load_param_prompt
import json
from accelerate import PartialState
from accelerate.utils import gather_object
import re

def parse_args():
    parser = argparse.ArgumentParser(description="Test Llama model")

    parser.add_argument('--model_path', type=str, default='output/', help='model path')
    parser.add_argument('--data_path', type=str, default='data/', help='data path')
    parser.add_argument('--history_path', type=str, default='data/user_history.json', help='history path')
    parser.add_argument('--device', type=str, default='cuda', help='device') 
    parser.add_argument('--base_model', type=str, default='meta-llama/Llama-2-7b-chat-hf', help='base model')
    parser.add_argument('--sample_num', type=int, default=None, help='number of samples')
    parser.add_argument('--num_beams', type=int, default=1, help='number of beams')
    parser.add_argument('--split', type=str, default='test', help='split to evaluate')
    parser.add_argument('--float16', action='store_true', help='use float16')
    parser.add_argument('--bf16', action='store_true', help='use bf16')
    parser.add_argument('--test_on', type=str, default='function', choices=['function', 'param', 'function_param'], help='test on tool or input')
    parser.add_argument('--max_new_tokens', type=int, default=512, help='max new tokens')
    parser.add_argument('--memory_token_length', type=int, default=768, help='memory token length')
    parser.add_argument('--tool_file', type=str, default='data/', help='task file')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('--res_file', type=str, default='output/', help='result file')
    parser.add_argument('--temperature', type=float, default=0, help='temperature')
    parser.add_argument('--do_sample', action='store_true', help='do sample')
    return parser.parse_args()


# def batch_inference(model, tokenizer, inputs, labels, batch_size, device, args, max_new_tokens):
#     rec = {'search': [], 'rec': [], 'review': []}

#     generation_config = GenerationConfig(
#         num_beams=args.num_beams,
#         max_new_tokens=max_new_tokens,
#         num_return_sequences=args.num_beams,
#         early_stopping=True if args.num_beams != 1 else False,
#         use_cache=True,
        
#         temperature=args.temperature if args.temperature > 0 else 0,
#         do_sample=args.do_sample,
#     )

#     model.eval()  
#     with torch.no_grad():
#         for i in tqdm(range(0, len(inputs), batch_size), desc='Evaluating batches'):
#             batch_inputs = inputs[i:i+batch_size]
#             batch_labels = labels[i:i+batch_size]

#             tokenized_prompts = [
#                 tokenizer(input_text, return_tensors="pt").input_ids
#                 for input_text in batch_inputs
#             ]

#             res = []
#             with distributed_state.split_between_processes(tokenized_prompts) as batched_prompts:
#                 for batch in batched_prompts:
#                     batch = batch.to(distributed_state.device)

#                     beams = model.generate(batch, generation_config=generation_config)
#                     #print(tokenizer.decode(beams[0], skip_special_tokens=True).strip())

#                     if args.test_on == 'function':
#                         beams = [tokenizer.decode(x, skip_special_tokens=True).strip().split('### Tool:\n')[-1] for x in beams]
#                     elif args.test_on == 'param':
#                         # beams = [tokenizer.decode(x, skip_special_tokens=True).strip().split('### Tool Parameter:\n')[-1] for x in beams]
#                         decoded_beams = []
#                         for x in beams:
#                             # 1. 解码成字符串
#                             full_text = tokenizer.decode(x, skip_special_tokens=True).strip()
                            
#                             # 2. 截取 Tool Parameter 之后的部分 (缩小搜索范围)
#                             if "### Tool Parameter:" in full_text:
#                                 output_part = full_text.split("### Tool Parameter:")[-1]
#                             else:
#                                 output_part = full_text # 万一没找到分隔符，就搜全文

#                             # 3. 【核心】使用正则提取 ASIN
#                             # 逻辑：寻找以 'B' 开头，后面跟着 9 个大写字母或数字的字符串
#                             # 这是亚马逊 ASIN 的标准格式
#                             asin_match = re.search(r'\b(B[A-Z0-9]{9})\b', output_part)
                            
#                             if asin_match:
#                                 # 找到了！提取出干净的 ID (例如 "B088HHY2XK")
#                                 final_res = asin_match.group(1)
#                             else:
#                                 # 没找到 ASIN 格式，说明模型可能生成了 Query 或者其他东西
#                                 # 对 Search 任务，我们需要保留原文本
#                                 # 去掉可能的 "ASIN:" 前缀和空白
#                                 final_res = output_part.replace("ASIN:", "").strip()

#                             decoded_beams.append(final_res)

#                         # 更新 beams
#                         beams = decoded_beams

#                     res.extend(beams)
#             res = gather_object(res)

#             for j in range(len(batch_labels)):
#                 label = batch_labels[j]
#                 result[tasks[i+j]] = res[j*args.num_beams:(j+1)*args.num_beams]
        
        

#     return rec

def batch_inference(model, tokenizer, inputs, labels, batch_size, device, args, max_new_tokens):
    rec = {'search': [], 'rec': [], 'review': []}

    generation_config = GenerationConfig(
        num_beams=args.num_beams,
        max_new_tokens=max_new_tokens,
        num_return_sequences=args.num_beams,
        early_stopping=True if args.num_beams != 1 else False,
        use_cache=True,
        temperature=args.temperature if args.temperature > 0 else 0,
        do_sample=args.do_sample,
    )

    model.eval()  
    with torch.no_grad():
        for i in tqdm(range(0, len(inputs), batch_size), desc='Evaluating batches'):
            batch_inputs = inputs[i:i+batch_size]
            batch_labels = labels[i:i+batch_size]

            tokenized_prompts = [
                tokenizer(input_text, return_tensors="pt").input_ids
                for input_text in batch_inputs
            ]

            res = []
            with distributed_state.split_between_processes(tokenized_prompts) as batched_prompts:
                for batch in batched_prompts:
                    batch = batch.to(distributed_state.device)
                    beams = model.generate(batch, generation_config=generation_config)

                    if args.test_on == 'function':
                        beams = [tokenizer.decode(x, skip_special_tokens=True).strip().split('### Tool:\n')[-1] for x in beams]
                    elif args.test_on == 'param':
                        decoded_beams = []
                        for x in beams:
                            full_text = tokenizer.decode(x, skip_special_tokens=True).strip()
                            
                            # 1. 截取 Tool Parameter 之后的内容
                            if "### Tool Parameter:" in full_text:
                                output_part = full_text.split("### Tool Parameter:")[-1]
                            else:
                                output_part = full_text

                            # 2. 使用正则提取纯净的 ASIN (B开头+9位字符)
                            # 这样可以过滤掉 "ASIN:", 句号, 或者其他废话
                            asin_match = re.search(r'\b(B[A-Z0-9]{9})\b', output_part)
                            
                            if asin_match:
                                final_res = asin_match.group(1)
                            else:
                                # 如果是 Search 任务或者没匹配到，就去除 ASIN: 前缀保留原样
                                final_res = output_part.replace("ASIN:", "").strip()

                            decoded_beams.append(final_res)
                        
                        beams = decoded_beams

                    res.extend(beams)
            
            # 收集所有 GPU 的结果
            res = gather_object(res)

            # ==============================================================================
            # 【核心修改】列表补全逻辑 (List Completion Strategy)
            # ==============================================================================
            for j in range(len(batch_labels)):
                # 获取当前样本的 Prompt
                current_prompt = batch_inputs[j]
                
                # 获取 LLM 预测的最佳答案 (取 Beam 1，即概率最高的那个)
                # res 列表是展平的，所以要根据 num_beams 计算索引
                llm_prediction = res[j * args.num_beams] 
                
                final_ranked_list = []

                # --- 1. 从 Prompt 中提取原始候选集 (Original Candidates) ---
                # 假设 Prompt 里也是用 ASIN: Bxxxx 格式标记候选集的
                # 使用 findall 抓取所有出现的 ASIN
                # 注意：这会抓到 User History 里的 ASIN 和 Candidates 里的 ASIN
                # 更好的做法是只在 "### Candidate List" 之后抓取，或者简单的全抓取后去重
                
                # 简单且鲁棒的做法：抓取 Prompt 里所有 ASIN，保留顺序
                all_asins_in_prompt = re.findall(r'\b(B[A-Z0-9]{9})\b', current_prompt)
                
                # 为了防止把 History 里的混进来，我们通常认为 Candidates 在 Prompt 的末尾
                # 或者，利用我们之前的格式 [1] ASIN: ...
                # 这里使用一个 trick：只取最后 100 个 ASIN (假设 History 不会超过 100 个)
                # 或者更严谨：split 之后再找
                if "### Candidate List" in current_prompt:
                    cand_part = current_prompt.split("### Candidate List")[-1]
                    candidate_pool = re.findall(r'\b(B[A-Z0-9]{9})\b', cand_part)
                else:
                    # 如果没找到标记，可能是 Search/Review 任务，或者 Prompt 格式不对
                    candidate_pool = []

                # --- 2. 构建最终列表 (Head Insertion) ---
                
                # 第一位：LLM 的预测 (如果它看起来像个 ASIN)
                if re.match(r'^B[A-Z0-9]{9}$', llm_prediction):
                    final_ranked_list.append(llm_prediction)
                else:
                    # 如果 LLM 输出的是 "running shoes" (Search任务)，直接保留
                    final_ranked_list.append(llm_prediction)

                # 后续位：用原始候选集补齐 (去重)
                for cand in candidate_pool:
                    if cand not in final_ranked_list:
                        final_ranked_list.append(cand)
                
                # --- 3. 针对不同任务类型的截断 ---
                # 如果是 Recommend，我们需要 Top 10 或 Top 50 列表进行评估
                # 如果是 Search/Review，通常只需要 Top 1 (LLM生成的内容)
                
                # 这里统一保存列表，评测脚本自己会取 Top K
                # 限制最大保存数量，比如 50，防止文件过大
                result[tasks[i+j]] = final_ranked_list[:50]

    return rec

# def batch_inference_10(model, tokenizer, inputs, labels, batch_size, device, args, max_new_tokens):
#     rec = {'search': [], 'rec': [], 'review': []}
#     result = {} # 用于存储最终结果
    
#     # 获取需要返回的序列数量，如果没有设置默认为 1
#     num_return = getattr(args, 'num_return_sequences', 1)

#     generation_config = GenerationConfig(
#         num_beams=args.num_beams,
#         max_new_tokens=max_new_tokens,
        
#         # 【修改点 A】设置返回数量
#         num_return_sequences=num_return, 
        
#         early_stopping=True if args.num_beams > 1 else False,
#         use_cache=True,
#         eos_token_id=tokenizer.eos_token_id, 
#         pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
#         temperature=args.temperature if args.temperature > 0 else 0,
#         do_sample=args.do_sample,
#         top_p=getattr(args, 'top_p', 1.0), # 如果设置了 top_p
#     )

#     model.eval()  
#     with torch.no_grad():
#         for i in tqdm(range(0, len(inputs), batch_size), desc='Evaluating batches'):
#             batch_inputs = inputs[i:i+batch_size]
#             batch_labels = labels[i:i+batch_size]

#             tokenized_prompts = [
#                 tokenizer(input_text, return_tensors="pt").input_ids
#                 for input_text in batch_inputs
#             ]

#             raw_text_results = []
            
#             with distributed_state.split_between_processes(tokenized_prompts) as batched_prompts:
#                 for batch in batched_prompts:
#                     batch = batch.to(distributed_state.device)
                    
#                     # 生成
#                     # 注意：如果 input batch size 是 4，num_return 是 10
#                     # 那么 beams_output 的行数会变成 40 (4 * 10)
#                     beams_output = model.generate(batch, generation_config=generation_config)
                    
#                     # 解码
#                     input_len = batch.shape[1]
#                     generated_tokens = beams_output[:, input_len:]
#                     decoded = [tokenizer.decode(g, skip_special_tokens=True).strip() for g in generated_tokens]
                    
#                     raw_text_results.extend(decoded)

#             # 收集所有卡的结果
#             # gathered_results 的长度应该是：Total_Input_Samples * num_return
#             gathered_results = gather_object(raw_text_results)

#             # 仅主进程处理结果逻辑
#             if distributed_state.is_main_process:
#                 # 【修改点 B】结果切分逻辑
#                 # 现在的 gathered_results 是扁平的，我们需要按 num_return 进行切分
#                 # 例如：[Input1_Seq1, Input1_Seq2... Input1_Seq10, Input2_Seq1...]
                
#                 # 当前 batch 在全局 inputs 中的起始位置
#                 # 注意：这里逻辑要小心，因为 gather_object 会收集所有 rank 的数据
#                 # 所以最好是只在循环结束后统一处理，但在 batch 循环里处理需要精细计算索引。
#                 # 为了简单起见，我们在 batch 循环里只处理当前 batch 的逻辑可能比较乱，
#                 # 建议：上面的 gather_object 会导致所有 GPU 等待。
#                 # 更好的写法是把 decoded 存起来，最后统一处理。但为了改动最小，这里假设 gather 顺序是对的。
                
#                 # 实际上，在 batch 内部做 gather_object 会比较慢。
#                 # 但根据你原本的代码结构，我们适配如下：
                
#                 # 我们只需要知道当前 batch 处理了多少个 input
#                 # 在多卡环境下，gathered_results包含了本轮 batch 所有卡处理的总数
                
#                 # 这里的逻辑其实有点风险：如果多卡数据切分不均，gather 回来的列表顺序需要确认。
#                 # 但通常 accelerate 保证顺序。
                
#                 # 计算当前 batch 实际处理的 input 数量（跨所有卡）
#                 # 这里略微复杂，为了稳妥，我们用一种更简单的方式：
#                 # 1. 把 gathered_results 按照 num_return 切块
                
#                 chunked_results = []
#                 for k in range(0, len(gathered_results), num_return):
#                     chunked_results.append(gathered_results[k : k + num_return])
                
#                 # 2. 这里的 chunked_results 长度应该等于 len(batch_inputs) (如果 batch_size 设置得当且没丢数据)
#                 # 但因为你是分 batch 循环的，这里的 gathered_results 其实对应的是 inputs[i : i+batch_size] 这一批
                
#                 for j, outputs_list in enumerate(chunked_results):
#                     idx = i + j # 全局索引
#                     if idx >= len(tasks): break

#                     processed_list = []
#                     for raw_res in outputs_list:
#                         # 清洗逻辑
#                         if args.test_on == 'function':
#                             final = raw_res.split('### Tool:\n')[-1].strip()
#                         elif args.test_on == 'param':
#                             final = raw_res.split('### Response:\n')[-1].strip()
#                             # 如果是 recommend，还要做正则提取等
#                             if 'recommend' in tasks[idx]: # 简单判断
#                                 match = re.search(r'\b(B[A-Z0-9]{9})\b', final)
#                                 if match: final = match.group(1)
#                         processed_list.append(final)

#                     # 存入结果字典
#                     # 结果格式：result[task_id] = ['结果1', '结果2', ..., '结果10']
#                     result[tasks[idx]] = processed_list

#     return result # 返回 result 而不是 rec，或者你在外面更新 rec


if __name__ == '__main__':

    args = parse_args()
    device = torch.device(args.device)
    print(device)

    base_model = args.base_model
    model_path = args.model_path

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    local_rank = int(os.environ.get("LOCAL_RANK") or 0)

    distributed_state = PartialState()

    if args.float16:
        torch_dtype = torch.float16
    elif args.bf16:
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32
    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    model = LlamaForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch_dtype,
            device_map=distributed_state.device
        )

    # 4. 【关键步骤】同样地，调整模型的词嵌入层大小以匹配新的Tokenizer
    #    这会将模型的词汇表大小从32000调整为32002，以匹配你的LoRA权重
    
    # 测试
    # test_string = "B0BK13W9G9"
    # model.resize_token_embeddings(len(test_string))

    # tokenized_output = tokenizer.tokenize(test_string)
    # print(f"'{test_string}' is tokenized into: {tokenized_output}")

    model = PeftModel.from_pretrained(
            model,
            model_path,
            torch_dtype=torch_dtype,
            device_map=distributed_state.device
        )


    print('tokenizer loaded from '+model_path)
    print('model loaded from '+model_path)

    model.to(device)
    model.eval()

    data_path = args.data_path
    batch_size = args.batch_size

    sample_num = args.sample_num
    num_beams = args.num_beams

    valid_modes = args.split.split(',')

    rec = {'search':[], 'rec':[], 'review':[]}
    result = {}
    for valid_mode in valid_modes:

        if args.test_on == 'function':
            tasks, total_inputs, total_labels = load_function_prompt(args.data_path, valid_mode)
        elif args.test_on == 'param':
            tasks, total_inputs, total_labels = load_param_prompt_beam_search(args.data_path, args.tool_file, valid_mode, args.memory_token_length, tokenizer)

        print('Data loaded from '+data_path)

        # 判断任务类型
        if valid_mode in ['search', 'review']:
            args.num_beams = 1
            args.temperature = 0
            args.do_sample = False
        else:
            args.num_beams = 1
            args.temperature = 0
            args.do_sample = False
        rec = batch_inference(model, tokenizer, total_inputs, total_labels, batch_size, device, args, args.max_new_tokens)

    with open(args.res_file, 'w') as f:
        json.dump(result, f, indent=2)
                
