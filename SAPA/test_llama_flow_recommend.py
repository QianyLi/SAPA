from transformers import GenerationConfig
from transformers import LlamaTokenizer, LlamaForCausalLM, AutoTokenizer
import torch
torch.manual_seed(42)
import os
from tqdm import tqdm
import argparse
from peft import PeftModel
from utils import load_param_prompt_beam_search_split_recommend_no_cat, load_function_prompt, load_param_prompt_beam_search
import json
from accelerate import PartialState
from accelerate.utils import gather_object


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
    parser.add_argument('--single_sample', action='store_true', default=False, help='Run on single sample')
    return parser.parse_args()



# ... (parse_args 函数保持不变) ...

def batch_inference(model, tokenizer, inputs, labels, batch_size, device, args, max_new_tokens, tasks):
    generation_config = GenerationConfig(
        num_beams=args.num_beams,
        max_new_tokens=max_new_tokens,
        num_return_sequences=args.num_beams,
        early_stopping=True if args.num_beams != 1 else False,
        use_cache=True,
        temperature=args.temperature if args.temperature > 0 else 1.0,
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
                    decoded_beams = tokenizer.batch_decode(beams, skip_special_tokens=True)
                    
                    final_batch_results = []
                    if args.test_on == 'function':
                        final_batch_results = [text.strip().split('### Tool:\n')[-1] for text in decoded_beams]
                    elif args.test_on == 'param':
                        for text in decoded_beams:
                            text = text.strip()
                            if '### Tool Parameter:\n' in text:
                                content = text.split('### Tool Parameter:\n')[-1]
                            elif '### Response:\n' in text:
                                content = text.split('### Response:\n')[-1]
                            else:
                                content = text
                            final_batch_results.append(content)
                    
                    res.extend(final_batch_results)
            
            res = gather_object(res)

            if distributed_state.is_main_process:
                for j in range(len(batch_labels)):
                    idx = i + j
                    if idx >= len(tasks): break
                    
                    current_task_id = tasks[idx]
                    current_prompt = batch_inputs[j]
                    
                    # 获取该任务的所有模型输出 (如果 num_beams > 1 则有多个)
                    task_beams = res[j*args.num_beams:(j+1)*args.num_beams]
                    
                    processed_beams = []
                    
                    processed_beams = task_beams

                    # 最终 result[id] 将会是一个列表，例如 ["ASIN1", "ASIN2", "ASIN3"...]
                    result[current_task_id] = processed_beams
                    
                    # 实时输出第一个结果进行调试
                    print(f"\n{'='*20} TASK DEBUG {'='*20}")
                    print(f"ID: {current_task_id}")
                    print(f"Top 1 Output: {processed_beams[0] if processed_beams else 'None'}")
                    print(f"Total Rows: {len(processed_beams)}")
                
                # 实时保存
                with open(args.res_file, 'w') as f:
                    json.dump(result, f, indent=2)

    return result

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
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = LlamaForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch_dtype,
            device_map=distributed_state.device
        )
    model = PeftModel.from_pretrained(
            model,
            model_path,
            torch_dtype=torch_dtype,
            device_map=distributed_state.device
        )
    
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


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
    # Resume support: if res_file already has partial results, load them and skip.
    if args.res_file and os.path.exists(args.res_file):
        try:
            with open(args.res_file) as _f:
                _prev = json.load(_f)
            if isinstance(_prev, dict):
                result.update(_prev)
                if distributed_state.is_main_process:
                    print(f"[resume] loaded {len(result)} existing entries from {args.res_file}")
        except Exception as _e:
            if distributed_state.is_main_process:
                print(f"[resume] failed to read {args.res_file}: {_e}; starting fresh")
    for valid_mode in valid_modes:

        if args.test_on == 'function':
            tasks, total_inputs, total_labels = load_function_prompt(args.data_path, valid_mode)
        elif args.test_on == 'param':
            tasks, total_inputs, total_labels = load_param_prompt_beam_search_split_recommend_no_cat(args.data_path, args.tool_file, valid_mode, args.memory_token_length, tokenizer)
        print('Data loaded from '+data_path)

        # Resume: drop tasks that are already in `result`.
        if result:
            keep = [i for i, t in enumerate(tasks) if t not in result]
            if len(keep) < len(tasks) and distributed_state.is_main_process:
                print(f"[resume] {valid_mode}: skip {len(tasks)-len(keep)} completed, run {len(keep)}")
            tasks = [tasks[i] for i in keep]
            total_inputs = [total_inputs[i] for i in keep]
            total_labels = [total_labels[i] for i in keep]
            if not tasks:
                continue

        print(total_inputs[:2],total_labels[:2])  # 调试：打印前两个输入，检查数据是否正确加载

        # 判断任务类型
        # NOTE: was greedy (num_beams=1) for recommend; we now sample 10 candidates
        # so the eval can do RRF fusion of N seed-based retrievals (matches the
        # search/review aggregation paradigm).
        if args.single_sample:
            args.num_beams = 1
            args.temperature = 0
            args.do_sample = False
        elif valid_mode in ['recommend']:
            args.num_beams = 1
            args.temperature = 0
            args.do_sample = False
        else:  # search / review
            args.num_beams = 10
            args.temperature = 1
            args.do_sample = True
        # args.num_beams = 10
        # args.temperature = 1
        # args.do_sample = True

        rec = batch_inference(model, tokenizer, total_inputs, total_labels, batch_size, device, args, args.max_new_tokens, tasks)
    with open(args.res_file, 'w') as f:
        json.dump(result, f, indent=2)
                

          

        
