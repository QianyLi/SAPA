from scipy.special import comb
from typing import Any, Dict, List
import os
import json
import random
import argparse
import multiprocessing
import sys
from math import comb
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from transformers import LlamaTokenizer, LlamaForCausalLM
from peft import PeftModel
import torch
import time
# 请确保你的环境中包含 PersonalWAB 相关的包
from PersonalWAB.agents.base import BaseAgent
from PersonalWAB.envs import get_env


def run(
    args: argparse.Namespace,
    ckpt_path,
):
    if args.max_steps == -1 and args.user_mode != "no":
        raise ValueError("Max steps must be set for user simulation mode")

    env = get_env(
        args.env,
        user_mode=args.user_mode,
        user_model=args.user_model,
        task_split=args.task_split,
        max_steps=args.max_steps,
    )
    end_index = (
        len(env.tasks) if args.end_index == -1 else min(args.end_index, len(env.tasks))
    )
    results = []
    lock = multiprocessing.Lock()
    print(
        f"Running {args.task_split} tasks {args.start_index} to {end_index} (checkpoint path: {ckpt_path})"
    )
    for i in range(args.num_trials):
        idxs = list(range(args.start_index, end_index))
        finished_idxs = []
        if os.path.exists(ckpt_path):
            with open(ckpt_path, "r") as f:
                finished_tasks = json.load(f)
                # 兼容处理：检查是否是包含统计信息的列表
                if isinstance(finished_tasks, list) and len(finished_tasks) > 0:
                    # 过滤掉统计字典（通常没有 task_id）
                    for res in finished_tasks:
                        if 'task_id' in res:
                            results.append(res)
                            finished_idxs.append(res["task_id"])
        
        idxs = [idx for idx in idxs if idx not in finished_idxs]

        def _run(idx: int) -> dict:
            isolated_env = get_env(
                args.env,
                user_mode=args.user_mode,
                user_model=args.user_model,
                task_split=args.task_split,
                max_steps=args.max_steps,
            )

            isolated_agent = agent_factory(
                tools_info=env.functions_info,
                sys_prompt=env.sys_prompt,
                args=args,
            )

            action_acc, res_acc, info = isolated_agent.act(
                isolated_env,
                idx,
                verbose=args.verbose,
                temperature=args.temperature,
                max_steps=env.max_steps,
                memory=args.agent_memory,
                memory_length=args.memory_length,
            )
            
            result = {
                "task_id": idx,
                "action_acc": action_acc,
                "res_acc": res_acc,
                "info": info,
                "traj": isolated_agent.get_messages(),
                "trial": i,
            }

            with lock:
                data = []
                if os.path.exists(ckpt_path):
                    with open(ckpt_path, "r") as f:
                        try:
                            data = json.load(f)
                        except json.JSONDecodeError:
                            data = []
                else:
                    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
                
                # 如果文件中包含之前的统计头部，建议在追加时小心，这里简单追加到列表
                # 实际生产中可能需要先剔除旧的统计头，或者最后再统一加
                with open(ckpt_path, "w") as f:
                    json.dump(data + [result], f, indent=2)
            return result

        # 注意：如果使用本地模型且显存紧张，建议将 max_workers 设为 1
        with ThreadPoolExecutor(max_workers=args.max_concurrency) as executor:
            for res in tqdm(executor.map(_run, idxs), total=len(idxs), desc=f"Trial {i}"):
                results.append(res)
    return results


def agent_factory(tools_info, sys_prompt, args: argparse.Namespace) -> BaseAgent:

    if args.agent_strategy == "function_calling":
        tools_info = [
            tool for tool in tools_info if tool["function"]["name"] != "get_product_details_by_asin"
        ]
        if (
            "gpt" in args.model
            or "mistralai/Mi" in args.model
            or "meta-llama/Meta-Llama-3-" in args.model
        ):
            from PersonalWAB.agents.gpt_function_calling_agent import (
                GPTFunctionCallingAgent,
                initialize_client,
            )

            if "gpt" in args.model:
                initialize_client(
                    api_key=os.getenv("OPENAI_API_KEY")
                )

            return GPTFunctionCallingAgent(tools_info, sys_prompt, model=args.model, 
                                           function_selection_file=args.sapa_function_file, memory_file=args.interec_memory_file)

    elif args.agent_strategy == "react" or args.agent_strategy == "react_reflect":
        tools_info = [
            tool for tool in tools_info if tool["function"]["name"] != "get_product_details_by_asin"
        ]
        from PersonalWAB.agents.chat_react_agent import ChatReActAgent, initialize_create

        if "gpt" in args.model:
            initialize_create(mode="openai")
        if args.agent_strategy == "react":
            return ChatReActAgent(tools_info, sys_prompt, model=args.model, )
        elif args.agent_strategy == "react_reflect":
            return ChatReActAgent(tools_info, sys_prompt, model=args.model, reflection=True)
    elif args.agent_strategy == "recmind":

        if (
            "gpt" in args.model
            or "mistralai/Mi" in args.model
            or "meta-llama/Meta-Llama-3-" in args.model
        ):
            from PersonalWAB.agents.gpt_function_calling_agent import (
                GPTFunctionCallingAgent,
                initialize_client,
            )

            if "gpt" in args.model:
                initialize_client(
                    api_key=os.getenv("OPENAI_API_KEY")
                )
            elif (
                "mistralai/Mi" in args.model or "meta-llama/Meta-Llama-3-" in args.model
            ):
                initialize_client(
                    api_key=os.getenv("ANYSCALE_API_KEY"),
                    base_url="https://api.endpoints.anyscale.com/v1",
                )

            return GPTFunctionCallingAgent(tools_info, sys_prompt, model=args.model)
    
    elif args.agent_strategy == "sapa":
        from PersonalWAB.agents.sapa_agent import SAPAAgent
        function_file = args.sapa_function_file
        param_file = args.sapa_param_file
        if args.sapa_generate == 0:
            '''To save time, simply use pre-generated results to evaluate'''
            return SAPAAgent(function_file, param_file, None, sys_prompt, None)
        
        # 修改点：建议将硬编码的路径 'meta-llama/Llama-2-7b-chat-hf' 改为变量或从 args 传入
        # 如果你的服务器无法访问 HuggingFace，请确保这里填写的是本地绝对路径
        base_model_path = os.getenv('PWAB_BASE_MODEL', 'meta-llama/Llama-2-7b-chat-hf')
        # 如果没有本地路径，使用 'meta-llama/Llama-2-7b-chat-hf'
        
        llama_model, llama_tokenizer = load_llama_model(args.sapa_model_path, base_model_path, torch.float16)
        return SAPAAgent(function_file, None, llama_model, sys_prompt, llama_tokenizer, max_length=1024, memory_token_length=args.mem_token_length)
    elif args.agent_strategy == "rise":
        from PersonalWAB.agents.rise_agent import RISEAgent, initialize_client
        if "gpt" in args.model:
            initialize_client(api_key=os.getenv("OPENAI_API_KEY"))
        return RISEAgent(
            tools_info=tools_info,
            sys_prompt=sys_prompt,
            model=args.model,
            function_selection_file=args.sapa_function_file,
            memory_file=args.interec_memory_file,
            tts_n=args.tts_n,
        )
    else:
        raise ValueError(f"Unknown agent strategy: {args.agent_strategy}")


global_model = None
global_tokenizer = None

def load_llama_model(model_path, base_model, torch_dtype):
    """
    修复后的模型加载函数：
    1. 使用 device_map="auto" 自动处理权重加载（解决 Meta tensor 错误）。
    2. 移除了 .to('cuda')，防止冲突。
    3. 增加了 Tokenizer 加载的鲁棒性。
    """
    global global_model, global_tokenizer
    if global_model is None and global_tokenizer is None:
        print(f"Loading tokenizer...")
        try:
            # 优先从微调路径加载 tokenizer
            global_tokenizer = LlamaTokenizer.from_pretrained(model_path)
        except Exception as e:
            print(f"Failed to load tokenizer from {model_path}, trying base model {base_model}. Error: {e}")
            global_tokenizer = LlamaTokenizer.from_pretrained(base_model)
        
        global_tokenizer.padding_side = "left"

        print(f"Loading base model from: {base_model} with device_map='auto'")
        # 关键修复：添加 device_map="auto"
        global_model = LlamaForCausalLM.from_pretrained(
                base_model,
                torch_dtype=torch_dtype,
                device_map="auto", 
            )
        
        print(f"Loading PEFT adapter from: {model_path}")
        # 关键修复：PEFT 加载也需要 device_map="auto" (通常它会跟随 base model，但显式指定更安全)
        global_model = PeftModel.from_pretrained(
                global_model,
                model_path,
                torch_dtype=torch_dtype,
                device_map="auto",
            )

        # 关键修复：移除 .to('cuda')，因为 device_map="auto" 已经把模型放到了正确的设备上
        # 如果此时再手动 to('cuda')，会尝试移动 Meta tensor 导致报错
        # if torch.cuda.is_available():
        #     global_model.to('cuda')
            
    return global_model, global_tokenizer


def calculate_statistics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    task_types = ["search", "recommend", "review"]
    
    stats = {
        task_type: {
            "action_sum": 0,
            "res_sum": 0,
            "total_count": 0,
            "interaction_count": 0  
        }
        for task_type in task_types
    }
    
    global_stats = {
        "action_sum": 0,
        "res_sum": 0,
        "total_count": 0,
        "interaction_count": 0  
    }

    for result in results:
        # 跳过统计头（如果有）
        if 'task_id' not in result and 'run_args' in result:
            continue

        if 'info' not in result:
            continue
        task_type = result.get("info", {}).get("task", {}).get("type")
        if task_type not in task_types:
            continue
        
        # 增加安全性检查，防止 completion_tokens 不存在
        usage = result.get('info', {}).get('usage', {})
        interaction_count = 0
        if usage and 'completion_tokens' in usage:
            interaction_count = len(usage['completion_tokens'])
            
        stats[task_type]["interaction_count"] += interaction_count
        global_stats["interaction_count"] += interaction_count

        action_acc_list = result.get("action_acc", [0])
        res_acc_list = result.get("res_acc", [0])

        if not action_acc_list: action_acc_list = [0]
        if not res_acc_list: res_acc_list = [0]
        
        valid_indexes = [i for i, acc in enumerate(action_acc_list) if acc == 1]

        if valid_indexes:
            best_index = max(valid_indexes, key=lambda i: res_acc_list[i] if i < len(res_acc_list) else 0)
            best_action_acc = action_acc_list[best_index]
            best_res_acc = res_acc_list[best_index] if best_index < len(res_acc_list) else 0

            stats[task_type]["action_sum"] += best_action_acc
            global_stats["action_sum"] += best_action_acc

            stats[task_type]["res_sum"] += best_res_acc
            global_stats["res_sum"] += best_res_acc
        else:
            stats[task_type]["action_sum"] += 0
            global_stats["action_sum"] += 0
            stats[task_type]["res_sum"] += 0
            global_stats["res_sum"] += 0

        stats[task_type]["total_count"] += 1
        global_stats["total_count"] += 1

    final_stats = {}
    for task_type in task_types:
        total_count = stats[task_type]["total_count"]
        interaction_count = stats[task_type]["interaction_count"]
        if total_count > 0:
            avg_action_acc = stats[task_type]["action_sum"] / total_count
            avg_res_acc = stats[task_type]["res_sum"] / total_count
            avg_interaction_times = interaction_count / total_count  
        else:
            avg_action_acc, avg_res_acc, avg_interaction_times = 0, 0, 0
        
        final_stats[task_type] = {
            "total_count": total_count,
            "avg_interaction_times": avg_interaction_times,
            "avg_action_acc": avg_action_acc,  
            "avg_res_acc": avg_res_acc  
        }
    
    global_total_count = global_stats["total_count"]
    global_interaction_count = global_stats["interaction_count"]
    if global_total_count > 0:
        global_avg_action_acc = global_stats["action_sum"] / global_total_count
        global_avg_res_acc = global_stats["res_sum"] / global_total_count
        global_avg_interaction_times = global_interaction_count / global_total_count  
    else:
        global_avg_action_acc, global_avg_res_acc, global_avg_interaction_times = 0, 0, 0
    
    final_stats["overall"] = {
        "total_count": global_total_count,
        "avg_interaction_times": global_avg_interaction_times, 
        "avg_action_acc": global_avg_action_acc,
        "avg_res_acc": global_avg_res_acc
    }
    
    return final_stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_trials", type=int, default=1)
    parser.add_argument(
        "--env", type=str, choices=["pwab"], default="pwab"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        choices=[
            # openai api models
            "gpt-4-turbo",
            "gpt-4-0125-preview",
            "gpt-4-1106-preview",
            "gpt-4-32k-0613",
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-1106",
            "gpt-3.5-turbo-0125",
            "gpt-4o",
            "gpt-4o-mini",
            # custom models
            "finetune/llama",
        ],
    )
    parser.add_argument(
        "--user_mode",
        type=str,
        default="no",
        choices=["no", "naive", "human"],
    )
    parser.add_argument(
        "--user_model",
        type=str,
        default="gpt-4o-mini",
    )
    parser.add_argument(
        "--agent_strategy",
        type=str,
        default="function_calling",
        choices=["function_calling", "react", "react_reflect", "recmind", "sapa", "rise"],
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--task_split", type=str, default="test", choices=["train", "test"]
    )
    parser.add_argument(
        "--agent_memory", type=str, default="none", choices=["taskspe", "taskspe_rise", "last", "relevant", "random", "recmind", "interecagent", "none"]
    )
    parser.add_argument(
        "--memory_length", type=int, default=1, help="Max memory length"
    )
    parser.add_argument(
        "--max_steps", type=int, default=-1, help="Max step number for agents to run, -1 for single round and no user simulation"
    )
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=-1, help="Run all tasks if -1")
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument("--log_dir", type=str, default="results")
    parser.add_argument("--num_gpus", type=int, default=None)
    parser.add_argument(
        "--max_concurrency",
        type=int,
        default=1,
        help="Number of tasks to run in parallel",
    )
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--shuffle", type=int, default=0)
    parser.add_argument("--interec_memory_file", type=str, default=None)
    
    parser.add_argument("--sapa_param_file", type=str, default=None)
    parser.add_argument("--sapa_function_file", type=str, default=None)
    parser.add_argument("--sapa_generate", type=int, default=0)
    parser.add_argument("--sapa_model_path", type=str, default='finetune/output/input/Llama-2-7b-chat-hf/')
    parser.add_argument("--mem_token_length", type=int, default=768)
    parser.add_argument("--tts_n", type=int, default=10)

    args = parser.parse_args()
    print(args)
    random.seed(args.seed)

    time_str = datetime.now().strftime("%m%d%H%M")
    turn_sig = 'singleturn' if args.max_steps == -1 else 'multiturn'
    
    # 清理模型名称字符串，防止路径错误
    model_name_safe = args.model.split('/')[-1]
    file_str = f'''{args.log_dir}/{turn_sig}/step{args.max_steps}_{args.agent_strategy}-{model_name_safe}-{args.temperature}_mem{args.agent_memory}_range{args.start_index}-{args.end_index}_user{args.user_model}_{time_str}.json'''
    
    if args.resume_from:
        file_str = args.resume_from
        print(f"Resuming from {file_str}")

    if not os.path.exists(os.path.dirname(file_str)):
        os.makedirs(os.path.dirname(file_str), exist_ok=True)

    # 运行
    results = run(
        args=args,
        ckpt_path=file_str,
    )

    # 统计
    final_res = calculate_statistics(results)
    for task_type, stats in final_res.items():
        print(f"\nTask type: {task_type}")
        for key, value in stats.items():
            print(f"{key}: {value}")

    # 计算总成本
    total_cost = 0
    for r in results:
        # 兼容旧数据和新数据，并过滤掉头部统计字典
        if 'info' in r and 'usage' in r['info'] and 'total_price' in r['info']['usage']:
            total_cost += r['info']['usage']['total_price']

    # 准备最终结果：头部是统计，后面是详情
    # 移除之前的统计头（如果有），避免重复叠加
    clean_results = [r for r in results if 'task_id' in r]
    
    total = {'run_args': vars(args), 'total cost': total_cost, 'results': final_res}
    clean_results.insert(0, total)

    with open(file_str, "w") as f:
        json.dump(clean_results, f, indent=2)
        print(f"\n Results saved to {file_str}\n")
        print(f"Total cost: {total_cost}")


if __name__ == "__main__":
    main()
