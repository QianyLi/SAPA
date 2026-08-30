import os
from torch.utils.data import Dataset
from transformers import (
    Trainer,
    TrainingArguments,
    TrainerCallback,
    DataCollatorWithPadding,
    GenerationConfig,
)
import torch
import logging
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import random
import wandb
from typing import Dict, List
import torch.nn.functional as F
import sys
import json
import re

import time

TRUNCATION_COUNT = 0 
class LLaMaDataset(Dataset):
    def __init__(self, tokenizer, json_file, max_length, split="train", subset_size=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.subset_size = subset_size
        
        with open(json_file, 'r', encoding="utf-8") as f:
            data = json.load(f)
        self.dataset = data.get(split, [])
        if self.subset_size is not None:
            indices = list(range(len(self.dataset)))
            sampled_indices = random.sample(indices, self.subset_size)
            self.dataset = [self.dataset[i] for i in sampled_indices]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        input_text = item['prompt']
        target_text = item['target']
        full_prompt = input_text
        return preprocess_function(full_prompt, target_text, self.tokenizer, self.max_length)

TRUNCATION_COUNT = 0
class LlaMaTrainerwithTemperature_qwen25(Trainer):
    def __init__(self, temperature=1.0, vocab_size=None, *args, **kwargs):
        if "tokenizer" in kwargs and "processing_class" not in kwargs:
            kwargs["processing_class"] = kwargs.pop("tokenizer")
        super().__init__(*args, **kwargs)
        self.temperature = temperature
        self.vocab_size = vocab_size

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")

        outputs = model(**inputs)
        logits = outputs.logits

        if self.temperature != 1.0:
            logits = logits / self.temperature

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        vocab_size = shift_logits.size(-1)

        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)

        shift_logits = shift_logits.view(-1, vocab_size)
        shift_labels = shift_labels.view(-1).to(shift_logits.device)

        loss = loss_fct(shift_logits, shift_labels)

        return (loss, outputs) if return_outputs else loss

# class LLaMaDataset_modified(Dataset):
#     def __init__(self, tokenizer, json_file: str, max_length: int = 4096, split: str = "train"):
#         self.tokenizer = tokenizer
#         self.max_len = max_length
#         with open(json_file, "r", encoding="utf-8") as f:
#             data = json.load(f)
#         self.samples = data.get(split, [])

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         item = self.samples[idx]
#         prompt = item["prompt"]
#         target = item["target"]
        
#         # 1. 拼接 Prompt + Target + EOS
#         full_text = prompt + target + self.tokenizer.eos_token
        
#         # 2. 对整个文本进行分词
#         tokenized_full = self.tokenizer(
#             full_text,
#             truncation=True,
#             max_length=self.max_len,
#             padding=False,
#             add_special_tokens=True
#         )
#         input_ids = torch.tensor(tokenized_full["input_ids"], dtype=torch.long)
#         labels = input_ids.clone()
        
#         # 3. 【核心修复】计算 Prompt 的长度，并 Mask 掉其对应的 Label
#         #    这样模型只为 Target 的预测负责。
        
#         # 再次对 Prompt 进行分词，计算它的真实长度
#         tokenized_prompt = self.tokenizer(
#             prompt,
#             add_special_tokens=True # 必须与 full_text 的设置一致
#         )
#         prompt_len = len(tokenized_prompt["input_ids"])

#         # 强制将 labels 中 Prompt 对应的部分设置为 -100
#         # -100 是 PyTorch/HF Trainer 约定俗成的 Loss 忽略值
#         if prompt_len < labels.shape[0]:
#              labels[:prompt_len] = -100
#         else:
#              # 如果 prompt 太长导致 target 被完全截断，就将所有 labels 设为 -100
#              # 这一步在新脚本中是多余的，因为你现在是 4096 长度，但写上更安全
#              labels[:] = -100 
        
#         return {"input_ids": input_ids, "labels": labels, "attention_mask": torch.ones_like(input_ids)}

class LLaMaDataset_qwen25(Dataset):
    def __init__(self, tokenizer, json_file, max_length, split="train", subset_size=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.subset_size = subset_size

        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.dataset = data.get(split, [])

        if self.subset_size is not None:
            indices = list(range(len(self.dataset)))
            sampled_indices = random.sample(indices, self.subset_size)
            self.dataset = [self.dataset[i] for i in sampled_indices]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        prefix_text = item["prompt"]
        target_text = item["target"]

        return preprocess_function_qwen25(
            prefix_text=prefix_text,
            target_text=target_text,
            tokenizer=self.tokenizer,
            max_length=self.max_length
        )


def preprocess_function_qwen25(prefix_text, target_text, tokenizer, max_length):
    global TRUNCATION_COUNT

    eos_token = tokenizer.eos_token
    if eos_token is None:
        eos_token = ""

    # Qwen2.5 的 prompt 已经由 apply_chat_template 生成，
    # 所以这里不要再额外添加 special tokens。
    response_text = f"{prefix_text}{target_text}{eos_token}"

    raw_tokenized = tokenizer(
        response_text,
        return_tensors=None,
        truncation=False,
        add_special_tokens=False
    )

    raw_len = len(raw_tokenized["input_ids"])

    prefix_tokenized = tokenizer(
        prefix_text,
        return_tensors=None,
        truncation=False,
        add_special_tokens=False
    )

    prefix_len = len(prefix_tokenized["input_ids"])

    if raw_len > max_length:
        TRUNCATION_COUNT += 1

        target_status = "Target PARTIAL"
        if prefix_len >= max_length:
            target_status = "Target LOST (!!!)"

        print(
            f"[Truncation #{TRUNCATION_COUNT}] "
            f"Raw: {raw_len} > Max: {max_length}. "
            f"Prefix: {prefix_len}. "
            f"Status: {target_status}"
        )

    tokenized = tokenizer(
        response_text,
        return_tensors=None,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        add_special_tokens=False
    )

    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]

    labels = input_ids.copy()

    prefix_tokenized_truncated = tokenizer(
        prefix_text,
        return_tensors=None,
        max_length=max_length,
        truncation=True,
        add_special_tokens=False
    )

    output_start_index = len(prefix_tokenized_truncated["input_ids"])

    # mask prompt 部分，只训练 target
    labels[:output_start_index] = [-100] * output_start_index

    # mask padding 部分，避免模型学习 pad token
    labels = [
        label if mask == 1 else -100
        for label, mask in zip(labels, attention_mask)
    ]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

import torch
from torch.utils.data import Dataset

class LLaMaDataset_LLAMA3(Dataset):
    def __init__(self, tokenizer, json_file, max_length, split="train", subset_size=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        with open(json_file, 'r', encoding="utf-8") as f:
            data = json.load(f)
        self.dataset = data.get(split, [])
        if subset_size is not None:
            self.dataset = random.sample(self.dataset, min(subset_size, len(self.dataset)))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        # 这里的 item['prompt'] 应该是已经填入变量后的完整字符串
        # 建议在外部处理好 Prompt 字符串，或者在这里调用上面的模板
        prefix_text = item['prompt'] 
        target_text = item['target']
        return preprocess_function_llama3(prefix_text, target_text, self.tokenizer, self.max_length)

def preprocess_function_llama3(prefix_text, target_text, tokenizer, max_length):
    # 1. 构造 Llama 3 格式
    # 确保 prefix_text 已经包含了 <|begin_of_text|> 等标记
    # 并在 target 后面加上结束符 <|eot_id|>
    full_prompt = prefix_text
    full_response = f"{target_text}<|eot_id|>"

    # 2. 分别编码以精确计算长度
    # add_special_tokens=False 因为我们在模板里手动加了
    prefix_ids = tokenizer.encode(full_prompt, add_special_tokens=False)
    target_ids = tokenizer.encode(full_response, add_special_tokens=False)

    # 3. 组合并截断
    input_ids = prefix_ids + target_ids
    if len(input_ids) > max_length:
        # 优先保留 Prompt 的头部和 Target
        input_ids = input_ids[:max_length]
    
    # 4. 构造 Labels (Masking the prompt)
    # 前缀部分不计算 Loss
    mask_len = len(prefix_ids)
    labels = [-100] * mask_len + input_ids[mask_len:]
    
    # 5. Padding
    padding_len = max_length - len(input_ids)
    if padding_len > 0:
        input_ids += [tokenizer.pad_token_id] * padding_len
        labels += [-100] * padding_len
    
    # 6. 构造 Attention Mask
    attention_mask = [1] * (max_length - padding_len) + [0] * padding_len

    return {
        "input_ids": torch.tensor(input_ids[:max_length], dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask[:max_length], dtype=torch.long),
        "labels": torch.tensor(labels[:max_length], dtype=torch.long),
    }


def preprocess_function(prefix_text, target_text, tokenizer, max_length):
    
    global TRUNCATION_COUNT # 声明使用全局变量
    
    # 构造完整文本
    response_text = f"{prefix_text}{target_text}</s>"

    # ==========================================
    # 2. 新增：长度检查逻辑
    # ==========================================
    # 先获取不截断的真实长度，只拿 input_ids 长度即可
    # 注意：这里 truncation=False
    raw_tokenized = tokenizer(response_text, return_tensors=None, truncation=False)
    raw_len = len(raw_tokenized["input_ids"])
    
    if raw_len > max_length:
        TRUNCATION_COUNT += 1
        
        # 计算 Prefix 长度，判断是否会导致 Target 完全丢失
        prefix_len = len(tokenizer(prefix_text, truncation=False)["input_ids"])
        target_status = "Target PARTIAL"
        if prefix_len >= max_length:
            target_status = "Target LOST (!!!)"
            
        # 打印警告 (为了不刷屏，可以加个取模，或者只打印前几次)
        # 这里的 print 在多卡训练时会在每个进程打印，DeepSpeed 环境下主要关注 rank 0
        print(f"⚠️ [Truncation #{TRUNCATION_COUNT}] Raw: {raw_len} > Max: {max_length}. Prefix: {prefix_len}. Status: {target_status}")

    input = tokenizer(
        response_text,
        return_tensors=None,
        max_length=max_length,
        truncation=True,
        padding="max_length",  
    )
    
    input_ids = input["input_ids"]
    labels = input_ids.copy()

    prefix_tex = tokenizer(
        prefix_text,
        return_tensors=None,
        max_length=max_length,
        truncation=True,
    )["input_ids"]

    # target_ids = tokenizer(
    #     target_text,
    #     return_tensors=None,
    #     max_length=max_length,
    #     truncation=True,
    # )["input_ids"]

    output_start_index = len(prefix_tex)

    # 输出 prefix 和 target 的长度
    # print(f"prefix_ids: {len(prefix_tex)}, target_ids: {len(target_ids)}, input_ids: {len(input_ids)}, max_length: {max_length}")

    labels[:output_start_index] = [-100] * output_start_index  

    return {
        "input_ids": input_ids,
        "attention_mask": input["attention_mask"],
        "labels": labels,
    }

def preprocess_function_modified(prefix_text, target_text, tokenizer, max_length):
    """
    标准的 SFT 数据处理函数：
    1. 拼接 Prompt + Target + EOS
    2. 计算 Prompt 的长度
    3. 将 Labels 中 Prompt 的部分设为 -100 (Mask)
    4. 不做 Padding (交给 DataCollator 处理，它会自动填 -100)
    """
    
    # 1. 构造完整文本 (Llama 通常不需要 BOS，但需要 EOS)
    # 格式: Prompt + Target + </s>
    response_text = f"{prefix_text}{target_text}</s>"

    # 2. Tokenize 完整文本
    # 关键点：这里只做截断(truncation)，不做填充(padding)！
    # 返回 list 而不是 tensor
    tokenized_full = tokenizer(
        response_text,
        return_tensors=None, 
        max_length=max_length,
        truncation=True,
        add_special_tokens=True # 确保包含 BOS (如果 tokenizer 配置了的话)
    )
    
    input_ids = tokenized_full["input_ids"]
    attention_mask = tokenized_full["attention_mask"]
    
    # 3. 复制 input_ids 到 labels
    labels = input_ids.copy()

    # 4. 计算 Prompt (Prefix) 的长度用于 Mask
    # 我们单独 tokenize prefix 来获取精准长度
    tokenized_prefix = tokenizer(
        prefix_text,
        return_tensors=None,
        max_length=max_length,
        truncation=True, # 防止 prefix 本身就超长
        add_special_tokens=True
    )
    
    input_prefix_len = len(tokenized_prefix["input_ids"])

    # 保护措施：如果截断导致 prefix 比 full text 还长（极少见），取较小值
    mask_len = min(input_prefix_len, len(input_ids))

    # 5. Mask 掉 Prompt 部分
    # 将 labels 前面的部分设为 -100，这样计算 loss 时会忽略 prompt
    labels[:mask_len] = [-100] * mask_len

    # 6. 返回结果
    # 注意：这里返回的是不定长的 list，DataCollator 会把它们统一 Pad 到 max_length
    # 并且 DataCollator 会自动把 labels 的 Pad 部分设为 -100
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def load_function_prompt(data_file, split):
    data = json.load(open(data_file, encoding="utf-8"))
    tasks = []
    source = []
    target = []
    for i in range(len(data[split])):
        task= data[split][i]['instruction']
        source_text = data[split][i]['prompt']
        target_text = data[split][i]['target']
        source.append(source_text)
        target.append(target_text)
        tasks.append(task)
    return tasks, source, target


def load_param_prompt(data_file, tool_file, split, mem_token_length, tokenizer):
    data = json.load(open(data_file, encoding="utf-8"))
    tool_file = json.load(open(tool_file))
    tasks = []
    source = []
    target = []
    for i in range(len(data[split])):
        item = data[split][i]
        task = item['instruction']
        input_text = PARAM_PROMPT.replace('<Instruction>', task)
        mem = item['mem']
        tokenized_memory = tokenizer(mem, return_tensors=None, truncation=True, max_length=mem_token_length)
        memory_text = tokenizer.decode(tokenized_memory["input_ids"], skip_special_tokens=True)
        input_text = input_text.replace('<Memory>', memory_text)
        input_text = input_text.replace('<Tool>', tool_file[task][0])
        tool_input = item['target']
        
        tasks.append(task)
        source.append(input_text)
        target.append(tool_input)
    
    return tasks, source, target

def load_param_prompt_beam_search(data_file, tool_file, split, mem_token_length, tokenizer):
    # data_file 应该是 sft_data.json
    data = json.load(open(data_file, encoding="utf-8"))
    
    tasks = []
    source = []
    target = []

    # 兼容处理：检查是否有 'test' key，如果没有，假设整个 list 就是数据
    if isinstance(data, dict) and 'test' in data:
        data_list = data['test']
    elif isinstance(data, list):
        data_list = data
    else:
        # 如果你只跑了 train 的 sft 数据，这里可能要改读 train
        data_list = data.get('train', [])

    for item in data_list:
        # 1. 直接拿 SFT 里的成品 Prompt
        prompt_text = item.get('prompt')
        if not prompt_text: continue # 防御性编程

        # 2. 判断任务类型 (通过正则看 Tool)
        import re
        match = re.search(r"Tool:\s*(\w+)", prompt_text)
        if not match: continue
        tool_name = match.group(1)

        task_type = ''
        if 'search' in tool_name: task_type = 'search'
        elif 'recommend' in tool_name: task_type = 'recommend'
        elif 'review' in tool_name: task_type = 'review'
        
        # 3. 筛选 Split
        if task_type != split:
            continue

        # 4. 【核心】Recommend 任务追加引导词
        if task_type == 'recommend':
            # 防止重复添加
            if not prompt_text.strip().endswith("ASIN:"):
                # 确保换行正确
                if not prompt_text.endswith('\n'):
                    prompt_text += '\n'
                prompt_text += "ASIN:"
        
        tasks.append(item.get('instruction', '')) # 只是用来做 ID 或 Log
        source.append(prompt_text)
        target.append(item['target'])
    
    print(f"Loaded {len(source)} tasks for split {split}")
    return tasks, source, target

def load_param_prompt_beam_search_origin(data_file, tool_file, split, mem_token_length, tokenizer):
    # 'split' 参数现在代表 'search', 'recommend', 或 'review'
    data = json.load(open(data_file, encoding="utf-8"))
    tool_file = json.load(open(tool_file))
    tasks = []
    source = []
    target = []

    # 仍然假设所有测试数据都在 'test' 这个 key 下。
    if 'test' not in data:
        raise KeyError("未在 data.json 文件中找到 'test' 数据分割。请确认数据格式。")

    # 循环遍历 'test' 分割下的所有样本
    for item in data['test']:
        
        # <--- 修改点 1: 从 prompt 字段中提取工具名称 ---
        # 我们使用正则表达式来稳定地找到 "Tool:" 后面的内容
        prompt_text = item['prompt']
        match = re.search(r"Tool:\s*(\w+)", prompt_text)
        
        if not match:
            # 如果某个样本没有 Tool 标识，就跳过它
            continue
            
        tool_name = match.group(1)

        # <--- 修改点 2: 根据工具名称判断任务类型 ---
        task_type = ''
        if 'search' in tool_name:
            task_type = 'search'
        elif 'recommend' in tool_name:
            task_type = 'recommend'
        elif 'review' in tool_name: # 这是一个合理的推断
            task_type = 'review'
        else:
            # 如果遇到未知的工具类型，也跳过
            continue

        # <--- 修改点 3: 核心判断逻辑 (与之前相同) ---
        # 只有当样本的类型与函数传入的 'split' 参数相匹配时，才处理它。
        if task_type == split:
            # --- 下面的代码与你的原始代码完全一致 ---
            task = item['instruction']
            input_text = PARAM_PROMPT.replace('<Instruction>', task)
            mem = item['mem']
            tokenized_memory = tokenizer(mem, return_tensors=None, truncation=True, max_length=mem_token_length)
            memory_text = tokenizer.decode(tokenized_memory["input_ids"], skip_special_tokens=True)
            input_text = input_text.replace('<Memory>', memory_text)
            input_text = input_text.replace('<Tool>', tool_file[task][0])
            tool_input = item['target']
            
            tasks.append(task)
            source.append(input_text)
            target.append(tool_input)
    
    # 当循环结束时，返回的 lists 中将只包含特定任务类型的数据
    return tasks, source, target


def load_param_prompt_beam_search_split_recommend_no_cat(data_file, tool_file, split, mem_token_length, tokenizer):
    # 加载数据
    data = json.load(open(data_file, encoding="utf-8"))
    
    # 这里的 tool_file 如果只是为了拿到 ground truth tool name，可以保留
    # 但如果只是为了拼接 prompt，其实已经不需要了，因为 prompt 已经拼好了
    # tool_file = json.load(open(tool_file)) 
    
    tasks = []
    source = []
    target = []

    # 【注意】生成/测试阶段通常是用 'test' 集，而不是 'train'
    # 如果你是为了在训练集上做验证，可以用 'train'
    target_split = 'test'
    print(f"Loading data from split: {target_split} for task type: {split}")

    for item in data[target_split]:
        
        # 1. 判断任务类型
        # 由于你的 prompt 里已经包含了 Tool Call: xxx，我们可以直接用正则或者是字符串匹配
        prompt_text = item['prompt']
        
        # 简单粗暴的判断方式 (比正则更快)
        current_task_type = ''
        if 'search_product_by_query' in prompt_text:
            current_task_type = 'search'
        elif 'get_recommendations_by_history' in prompt_text:
            current_task_type = 'recommend'
        elif 'add_product_review' in prompt_text:
            current_task_type = 'review'
        
        # 2. 过滤任务类型
        if current_task_type != split:
            continue

        # 3. 直接获取数据
        task = item['instruction']
        
        # 【核心修改】直接使用已经生成好的完整 Prompt
        input_text = item['prompt'] 
        
        # 获取 Target (测试集也是有的，用来算指标)
        tool_input = item['target']
        
        tasks.append(task)
        source.append(input_text)
        target.append(tool_input)
    
    print(f"Loaded {len(tasks)} samples.")
    return tasks, source, target

class LlaMaTrainerwithTemperature(Trainer):
    def __init__(self, temperature=1.0, vocab_size=32000, *args, **kwargs):
        if "tokenizer" in kwargs and "processing_class" not in kwargs:
            kwargs["processing_class"] = kwargs.pop("tokenizer")
        super().__init__(*args, **kwargs)
        self.temperature = temperature
        self.vocab_size = vocab_size

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        logits = logits / self.temperature

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
        shift_logits = shift_logits.view(-1, self.vocab_size)
        shift_labels = shift_labels.view(-1)

        shift_labels = shift_labels.to(shift_logits.device)
        #print(shift_logits.shape, shift_labels.shape)
        loss = loss_fct(shift_logits, shift_labels)
        # torch.set_printoptions(threshold=torch.inf)
        # if loss == float('inf') or torch.isnan(loss):
        #     print('inf or nan loss')
        #     print(inputs.input_ids)

        return (loss, outputs) if return_outputs else loss


from openai import OpenAI, RateLimitError, APIError

HISTORY_PROMPT = '''
MEMORY <NUM>:
Product:
- Title: <TITLE>
- Parent Asin: <PARENT_ASIN>
- Main Category: <MAIN_CATEGORY>
- Average Rating: <AVERAGE_RATING>
- Rating Number: <RATING_NUMBER>
- Price: <PRICE>
- Store: <STORE>
- Details: <DETAILS>
- Description: <DESCRIPTION>
- Features: <FEATURES>
Review:
- Rating: <RATING>
- Text: <TEXT>
- Timestamp: <TIMESTAMP>
'''

TS_SEARCH_PROMPT = '''Title:<TITLE>
Main Category:<MAIN_CATEGORY>
Price:<PRICE>
Store:<STORE>
'''

TS_REC_PROMPT = '''Title:<TITLE>
Main Category:<MAIN_CATEGORY>
Asin:<ASIN>
'''

TS_REV_PROMPT = '''Rating:<RATING>
Text:<TEXT>
'''

PRODUCT_INFO_PROMPT = '''
The product is:
Title: <Title>
Price: <Price>
Store: <Store>
Main Category: <Main Category>
'''

PARAM_PROMPT = '''Below is an instruction that describes a task. Generate the tool parameter that appropriately completes the request. 
### Instruction:<Instruction>
  
Memory: <Memory>

Tool: <Tool>

### Tool Parameter:
'''

PARAM_PROMPT_RECOMMEND = '''Below is an instruction that describes a task. Select the tool parameter (Asin) that appropriately completes the request. 

### Instruction:<Instruction>

### User History (Reference):
<History>

### Candidate List (Options):
<Candidates>

Tool: <Tool>

### Requirement:
Please select the most suitable item from the "Candidate List" based on the Instruction and History.
**Output ONLY the Product ID (ASIN) of the selected item.** 
Do not output the product name, explanations, or any other text.

### Tool Parameter:
'''

# 推荐任务模板
LLAMA3_PROMPT_RECOMMEND = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a precise tool parameter generator. Your task is to select the most suitable Product ID (ASIN) from the candidate list. 
**Strict Constraint: Output ONLY the Product ID (ASIN). No explanations, no extra text.**<|eot_id|><|start_header_id|>user<|end_header_id|>

### Instruction:
<Instruction>

### User History (Reference):
<History>

### Candidate List (Options):
<Candidates>

### Tool:
<Tool>

### Task:
Select the most suitable ASIN from the "Candidate List" based on the Instruction and History.
Output ONLY the Product ID (ASIN).<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

# 搜索和评论任务模板 (通用)
LLAMA3_PROMPT_GENERATE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a precise tool parameter generator. Output ONLY the required parameter value without any explanation.<|eot_id|><|start_header_id|>user<|end_header_id|>

### Instruction:
<Instruction>
  
### Context:
Memory: <History>
Tool: <Tool>

### Task:
Generate the tool parameter that appropriately completes the request.
Output ONLY the value.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

PARAM_PROMPT_RECOMMEND_top10 = '''Below is an instruction that describes a task. Select the tool parameter (Asin) that appropriately completes the request. 

### Instruction:<Instruction>

### User History (Reference):
<History>

### Candidate List (Options):
<Candidates>

Tool: <Tool>

### Requirement:
Please select the **top 10 most suitable items** from the "Candidate List" based on the Instruction and History.
**Output ONLY the Product IDs (ASINs) of the selected items, ordered from most suitable to least suitable, separated by commas.**
Do not output product names, explanations, or any other text.

### Tool Parameter:
'''

PARAM_PROMPT_SEARCH = '''Below is an instruction that describes a task. Generate the tool parameter that appropriately completes the request. 
### Instruction:<Instruction>
  
Memory: <Memory>

Tool: <Tool>

### Tool Parameter:
'''
PARAM_PROMPT_REVIEW = '''Below is an instruction that describes a task. Generate the tool parameter that appropriately completes the request.
### Instruction:<Instruction>
  
Memory: <Memory>

Tool: <Tool>

### Tool Parameter:
'''


FUNCTION_PROMPT = '''Below is an instruction that describes a task. Choose a tool that appropriately completes the request.
### Instruction: <Instruction>

### Tool:
'''

PARAM_PROMPT_RECOMMEND_QWEN25 = [
    {
        "role": "system",
        "content": (
            "You are a precise tool-parameter generator. "
            "You must follow the output format strictly. "
            "Do not provide explanations."
        )
    },
    {
        "role": "user",
        "content": """Select the tool parameter (Asin) that appropriately completes the request.

### Instruction:
<Instruction>

### User History (Reference):
<History>

### Candidate List (Options):
<Candidates>

### Tool:
<Tool>

### Requirement:
Select the most suitable item from the Candidate List based on the Instruction and User History.

Output ONLY the Product ID (ASIN) of the selected item.
Do not output the product name.
Do not output explanations.
Do not output any extra text.

### Tool Parameter:"""
    }
]


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def encode_texts(texts, model, tokenizer, batch_size=32):
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        encoded_input = tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt')
        if torch.cuda.is_available():
            encoded_input = encoded_input.to('cuda')

        with torch.no_grad():
            model_output = model(**encoded_input)
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        all_embeddings.append(sentence_embeddings)

        torch.cuda.empty_cache()  # Clear memory cache
    return torch.cat(all_embeddings, dim=0)

def pretty_history(item, num):
    res = HISTORY_PROMPT.replace("<TITLE>", item['product_info']['title'])
    res = res.replace("<PARENT_ASIN>", item['product_info']['parent_asin'])
    res = res.replace("<AVERAGE_RATING>", str(item['product_info']['average_rating']))
    res = res.replace("<RATING_NUMBER>", str(item['product_info']['rating_number']))
    res = res.replace("<PRICE>", str(item['product_info']['price']))
    res = res.replace("<STORE>", str(item['product_info']['store']))
    res = res.replace("<DETAILS>", json.dumps(item['product_info']['details']))
    res = res.replace("<DESCRIPTION>", str(item['product_info']['description']))
    res = res.replace("<FEATURES>", str(item['product_info']['features']))
    res = res.replace("<MAIN_CATEGORY>", str(item['product_info']['main_category']))
    res = res.replace("<RATING>", str(item['review']['rating']))
    res = res.replace("<TEXT>", item['review']['text'])
    res = res.replace("<TIMESTAMP>", str(item['review']['timestamp']))
    res = res.replace("<NUM>", str(num))
    return res

def retrieve_top_k_memories(request, history, model, tokenizer, k=50):
    request_embedding = encode_texts([request], model, tokenizer)
    history_embeddings = encode_texts([pretty_history(item, i) for i, item in enumerate(history)], model, tokenizer)
    similarity = F.cosine_similarity(request_embedding, history_embeddings, dim=1)
    top_k = similarity.argsort(descending=True)[:k]
    torch.cuda.empty_cache()
    return [history[i] for i in top_k]

def retrieve_top_k_memories_score(request, history, model, tokenizer, k=50):
    # 1. 编码 Query
    request_embedding = encode_texts([request], model, tokenizer)
    
    # 2. 编码 History 
    # 注意：如果你的 history 已经是处理好的字符串列表（如主代码传来的 hist_cand_strs），
    # 这里可以直接用 history，不需要再调 pretty_history。
    # 如果 history 还是原始字典列表，则保留 pretty_history。
    # 假设这里传入的是已经处理好的文本列表：
    candidates_text = history
    # 如果 history 已经是 string，直接用: candidates_text = history
    
    history_embeddings = encode_texts(candidates_text, model, tokenizer)
    
    # 3. 计算相似度
    similarity = F.cosine_similarity(request_embedding, history_embeddings, dim=1)
    
    # 4. 获取 Top K 的索引
    # 确保 k 不超过历史记录的总数
    real_k = min(k, len(history))
    top_k_indices = similarity.argsort(descending=True)[:real_k]
    
    # 5. 组装结果 [(content, score), ...]
    results = []
    for idx in top_k_indices:
        idx = idx.item()              # 转为 Python int
        score = similarity[idx].item() # 转为 Python float
        content = history[idx]         # 获取对应的内容
        results.append((content, score))
        
    torch.cuda.empty_cache()
    
    # 返回格式: [("history_text_1", 0.85), ("history_text_2", 0.40), ...]
    return results

def retrieve_top_k_memories_formatted(request, history, model, tokenizer, k=50):
    request_embedding = encode_texts([request], model, tokenizer)
    # print("history:", history)
    history_embeddings = encode_texts(history, model, tokenizer)
    similarity = F.cosine_similarity(request_embedding, history_embeddings, dim=1)
    top_k = similarity.argsort(descending=True)[:k]
    torch.cuda.empty_cache()
    return [history[i] for i in top_k]

def prettify_product_info(product_info):
    res = PRODUCT_INFO_PROMPT.replace('<Title>', product_info['title'])
    res = res.replace('<Price>', str(product_info['price']))
    res = res.replace('<Store>', str(product_info['store']))
    res = res.replace('<Main Category>', str(product_info['main_category']))
    return res

def sup_pretty_history(item, task_type):
    if task_type == 'search':
        return TS_SEARCH_PROMPT.replace("<TITLE>", item['product_info']['title']).replace("<MAIN_CATEGORY>", str(item['product_info']['main_category'])).replace("<PRICE>", str(item['product_info']['price'])).replace("<STORE>", str(item['product_info']['store']))
    elif task_type == 'recommend':
        return TS_REC_PROMPT.replace("<TITLE>", item['product_info']['title']).replace("<MAIN_CATEGORY>", str(item['product_info']['main_category'])).replace("<ASIN>", str(item['product_info']['parent_asin']))
    elif task_type == 'review':
        return TS_REV_PROMPT.replace("<RATING>", str(item['review']['rating'])).replace("<TEXT>", item['review']['text'])

def build_taskspe_memory_origin(history, task_type):
    if len(history) == 0:
        return []
    if task_type == 'recommend':
        history = sorted(history, key=lambda x: x['review']["timestamp"], reverse=True)
    return [sup_pretty_history(item, task_type) for item in history]

def build_taskspe_memory(history: list, task_type: str) -> list:
    """
    Generates a list of memory strings from product history,
    tailoring the content based on the task type.
    """
    mem = []
    if not history:
        return []

    for item in history:
        # Safely get the product_info dictionary
        info = item.get('product_info')
        if not isinstance(info, dict):
            continue  # Skip this item if product_info is missing or invalid

        # --- Logic branch based on task_type ---

        if task_type == 'search':
            # For 'search', include details relevant to finding a product
            title = info.get('title', 'N/A')
            category = info.get('main_category', 'N/A')
            price = info.get('price', 'N/A')
            store = info.get('store', 'N/A')
            # Format into a single, structured string
            mem_string = f"Title: {title} | Category: {category} | Price: ${price} | Store: {store}"
            mem.append(mem_string)

        elif task_type == 'recommend':
            # For 'recommend', focus on identifying features and the item itself
            title = info.get('title', 'N/A')
            category = info.get('main_category', 'N/A')
            # parent_asin is the best identifier for a product family
            asin = info.get('parent_asin', 'N/A')
            mem_string = f"Title: {title} | Category: {category} | ASIN: {asin}"
            mem.append(mem_string)

        elif task_type == 'review':
            user_rating = info.get('rating', 'N/A')
            # 获取用户评论文本
            review_text = info.get('text', '').replace('\n', ' ') 
            
            # 获取商品标题 (为了检索时能匹配到相似商品)
            title = info.get('title', 'N/A')
            
            # 组合字符串：既包含商品名(方便检索相似品)，也包含评论内容(方便模仿风格)
            # 限制长度：防止单条评论过长撑爆 Prompt
            mem_string = f"Product: {title} | User Rating: {user_rating} | Review: {review_text[:200]}"
            mem.append(mem_string)
            
        else:
            # A fallback for any other task type, just includes the title
            title = info.get('title', 'N/A')
            mem_string = f"Title: {title}"
            mem.append(mem_string)

    return mem

def generate_search_query_modified(instruction, mem):
    prompt = '''You are an expert Search Query Optimizer. 
Your task is to convert the User Instruction and User History into a concise, high-precision keyword search query.

### Input Data:
Instruction: <Instruction>
Memory: <Memory>

### Rules:
1. **Analyze Intent:** Identify the core product type and key attributes (material, style, specific constraints) from the Instruction.
2. **Leverage History:** Check the 'Memory'. IF AND ONLY IF the Instruction implies a reference (e.g., "for *my* device", "buy *it* again"), extract the specific Brand or Model from the Memory. If the Instruction is generic, ignore the Memory.
3. **Keyword Extraction:** Select only the most distinct keywords.
4. **Negative Constraints (Critical):**
   - **NO Hallucinations:** DO NOT include brands or models that are not explicitly present in the Input or Memory.
   - **NO Stop Words:** Remove words like "I need", "looking for", "best", "recommendation".
   - **NO Repetition:** Do not repeat the same keyword twice.
5. **Format:** Output a single line of space-separated keywords.

Tool: search_product_by_query
'''

    prompt = prompt.replace('<Instruction>', instruction)
    prompt = prompt.replace('<Memory>', '|'.join(mem))
    messages = [{'role': 'system', 'content': prompt}]
    #print(prompt)
    client = OpenAI(
        api_key = os.getenv("OPENAI_API_KEY"),
        base_url="https://xiaoai.plus/v1")
    # response = client.chat.completions.create(
    #     model='gpt-4o-mini',
    #     messages=messages,
    #     temperature=0,
    # )
    # message = response.choices[0].message.content
    #print(message)
    try:
        response = client.chat.completions.create(
            model='gpt-4o-mini',
            messages=messages,
            temperature=0,
        )
        message = response.choices[0].message.content
    except Exception as e:
        print(f"[OpenAI ERROR] {e}")
        message = "<API调用失败>"
    return message

def generate_search_query_origin(instruction, mem):
    prompt = '''As a personalized shopping agent, you can help users search for products.

Rules:
- The user will provide a request.
- You need to use the tool to find the product, the params for the tool is a textual query.
- Make the best tool call based on the user's request and the memory provided.
- Information in the memory can help you make a better tool call.
- You have only one chance to make a tool call, so make sure you have the best input for the tool.
- The tool will be provided, you only need to provide the most appropriate input for the tool.
- Do not inlcude any tool name, other information, or explanation.

Instruction: <Instruction>

Memory: <Memory>

Tool: search_product_by_query
'''
    prompt = prompt.replace('<Instruction>', instruction)
    prompt = prompt.replace('<Memory>', '|'.join(mem))
    messages = [{'role': 'system', 'content': prompt}]
    #print(prompt)
    client = OpenAI(
        api_key = os.getenv("OPENAI_API_KEY"),
        base_url="https://xiaoai.plus/v1")
    # response = client.chat.completions.create(
    #     model='gpt-4o-mini',
    #     messages=messages,
    #     temperature=0,
    # )
    # message = response.choices[0].message.content
    #print(message)
    try:
        response = client.chat.completions.create(
            model='gpt-4o-mini',
            messages=messages,
            temperature=0,
        )
        message = response.choices[0].message.content
    except Exception as e:
        print(f"[OpenAI ERROR] {e}")
        message = "<API调用失败>"
    return message

def generate_search_query(instruction, mem):
    prompt = '''### Role:
You are an expert E-commerce Search Intent Specialist. Your goal is to transform a user's conversational request into a precise, high-recall search query.

### Core Strategy:
1. **Contextual Resolution (Priority):**
   - If the user mentions a general product (e.g., "my vacuum", "this watch", "the parts for it") and the Memory contains a specific brand/model that matches the category, replace the general term with the [Brand + Model/Title Keywords] from Memory.
   - Example: User says "my vacuum" + Memory shows "ILIFE V5s Pro" -> Query should use "ILIFE V5s Pro".

2. **Attribute Preservation (Do NOT lose info):**
   - Keep ALL discriminative attributes mentioned by the user. 
   - This includes: Flavors (e.g., "creamy butter"), Specific Textures (e.g., "waffle texture"), Technical Features (e.g., "tagless", "waterproof", "removable liner"), and Sizes (e.g., "plus-size", "queen-size").

3. **Aggressive Fluff Removal:**
   - Delete all conversational filler: "Hey there", "I'm super excited", "I'm on the hunt for", "Looking for", "Any awesome recommendations", "Thanks".

4. **Category Sanity Check:**
   - If the Memory is about a completely different category (e.g., Memory is "Shoes", User asks for "Potatoes"), IGNORE the Memory and focus only on the Instruction.

### Format Rules (Strict):
- **NO QUOTES:** Do not wrap the output in any quotation marks.
- **NO SENTENCES:** Do not output a grammatical sentence. Output a concise keyword string.
- **NO EXPLANATIONS:** Output ONLY the final query.

### Examples:
- History: [Title: Sony WH-1000XM5 Headphones...]
  User: "I need new ear pads for my headphones, prefer soft ones."
  Target: Sony WH-1000XM5 soft replacement ear pads

- History: [Title: KitchenAid Artisan 5-Quart Stand Mixer...]
  User: "Looking for a blue cover for it."
  Target: KitchenAid Artisan 5-Quart Stand Mixer blue cover

- History: [Title: Nike Air Max 270...]
  User: "Show me some red running shoes, must be breathable."
  Target: red breathable running shoes

- History: [None]
  User: "I'm on the hunt for some yummy instant mashed potatoes, must have a creamy butter flavor!"
  Target: creamy butter flavor instant mashed potatoes

- History: [Title: No nonsense Women's Classic Leggings...]
  User: "Hey! I want some comfy denim-style leggings like these."
  Target: No nonsense comfy denim-style leggings

### Task Execution:
History: <Memory>

Instruction: <Instruction>

'''
    prompt = prompt.replace('<Instruction>', instruction)
    prompt = prompt.replace('<Memory>', '|'.join(mem))
    messages = [{'role': 'system', 'content': prompt}]
    #print(prompt)
    client = OpenAI(
        api_key = os.getenv("OPENAI_API_KEY"),
        base_url="https://xiaoai.plus/v1")
    # response = client.chat.completions.create(
    #     model='gpt-4o-mini',
    #     messages=messages,
    #     temperature=0,
    # )
    # message = response.choices[0].message.content
    #print(message)
    try:
        response = client.chat.completions.create(
            model='gpt-4o-mini',
            messages=messages,
            temperature=0,
        )
        message = response.choices[0].message.content
    except Exception as e:
        print(f"[OpenAI ERROR] {e}")
        message = "<API调用失败>"
    return message

def generate_content_based_recommendations(target_product_info: dict, 
                                           user_history: list,
                                           sim_model, 
                                           sim_tokenizer,
                                           top_k=10) -> str:
    """
    模拟基于内容的推荐，找出用户历史中与目标商品最相似的商品。
    """
    target_title = target_product_info.get('title', '')
    target_category = target_product_info.get('main_category', '')
    target_asin = target_product_info.get('parent_asin', '')

    # 1. 筛选同品类的历史商品作为候选
    candidate_items = [
        item for item in user_history 
        if item['product_info'].get('main_category') == target_category
    ]

    if not candidate_items:
        return target_asin # 如果没有候选，返回目标商品本身作为保底

    # 2. 获取候选商品标题和目标商品标题
    candidate_titles = [item['product_info'].get('title', '') for item in candidate_items]
    
    # 3. 使用句子模型计算标题嵌入
    with torch.no_grad():
        target_embedding = sim_model(**sim_tokenizer(target_title, return_tensors='pt', truncation=True).to(sim_model.device)).pooler_output
        candidate_embeddings = sim_model(**sim_tokenizer(candidate_titles, return_tensors='pt', padding=True, truncation=True).to(sim_model.device)).pooler_output

    # =====================================================================
    # ==================== 这是唯一的修改点 ===============================
    # =====================================================================
    # 4. 计算余弦相似度 (使用 PyTorch 原生函数)
    # 原代码: cosine_scores = util.cos_sim(target_embedding, candidate_embeddings)[0]
    cosine_scores = F.cosine_similarity(target_embedding, candidate_embeddings, dim=1)
    # =====================================================================
    
    # 5. 根据相似度排序，选出top_k
    # 我们将分数和候选商品打包，然后排序
    scored_candidates = sorted(list(zip(cosine_scores, candidate_items)), key=lambda x: x[0], reverse=True)
    
    # 提取 top_k 个商品的 ASIN
    top_asins = [item['product_info']['parent_asin'] for score, item in scored_candidates[:top_k]]
    
    if not top_asins:
        return target_asin

    return ', '.join(top_asins)

import random
import re
try:
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("Spacy 'en_core_web_sm' model not found. Please run 'python -m spacy download en_core_web_sm'")
        nlp = None
except ImportError:
    spacy = None
    nlp = None

# 定义一些常见的停用词和噪音词
STOP_WORDS = {'&', 'for', 'the', 'a', 'in', 'of', 'with', 'is', 'an', 'to', 'by', 'on', 'pack', 'of', 'for'}

def extract_entities_and_keywords(title: str, brand: str = None) -> (dict, list):
    """
    使用Spacy和规则，从标题中提取核心实体和通用关键词。
    """
    entities = {'BRAND': set(), 'PRODUCT': set(), 'ATTRIBUTE': set()}
    if brand:
        entities['BRAND'].add(brand)

    # 正则表达式来捕获型号、尺寸等 (示例)
    model_pattern = r'\b([A-Z0-9]{2,}-[A-Z0-9]{2,}|[A-Z]{2,}\d{2,})\b'
    size_pattern = r'\b(\d+(\.\d+)?\s?(inch|oz|gb|mb|tb|cm|mm))\b'
    
    models = re.findall(model_pattern, title, re.IGNORECASE)
    sizes = re.findall(size_pattern, title, re.IGNORECASE)
    
    for m in models:
        entities['PRODUCT'].add(m)
    for s in sizes:
        entities['ATTRIBUTE'].add(s[0]) # s是元组, e.g., ('15.6 inch', '.6', 'inch')

    # 使用Spacy进行更通用的名词短语提取
    keywords = []
    if nlp:
        doc = nlp(title)
        # 提取名词和专有名词，作为通用描述
        for token in doc:
            if token.pos_ in ('NOUN', 'PROPN') and token.text.lower() not in STOP_WORDS:
                # 排除已经被识别为实体的词
                if token.text not in entities['PRODUCT'] and token.text not in entities['BRAND']:
                    keywords.append(token.text)
    else: # Spacy不可用时的降级方案
        keywords = [word for word in title.replace(',', ' ').split() if word.lower() not in STOP_WORDS]

    # 清理一下，把集合转为列表
    for key in entities:
        entities[key] = list(entities[key])

    return entities, keywords


def prefer_analyse_function(instruction):
    PREFERENCE_DIMENSIONS = [
        "style",         # e.g., Casual, Minimalist, Sporty, Elegant
        "features",      # e.g., with Pockets, Waterproof, Adjustable
        "brand",         # e.g., Halife, adidas, Spoonk
        "material",      # e.g., Cotton, Leather, Eco-friendly
        "quality_tier",  # e.g., High-end, Value for money, Budget-friendly
        "price_range",   # e.g., Affordable, Under $50, Premium
        "color_palette",  # e.g., Neutral colors, Bright colors, prefers Blue
        "values"       # e.g., Eco-friendly, Handmade, Local
    ]

    prompt = """
You are a top-tier e-commerce query analyst and strategist. Your task is to analyze a user's current query to determine which shopping dimensions are already specified and which ones need to be inferred from user history to make the query more precise.

---
[BACKGROUND]

1.  **User's Current Query**:"{{current_query}}"
2.  **Available Preference Dimensions**:
    {{PREFERENCE_DIMENSIONS}}

---
[YOUR TASK]

Based on the user's query, perform two tasks and return the result as a single JSON object:
1.  `explicitly_mentioned_dimensions`: Identify which dimensions from the list are already **clearly mentioned or strongly implied** in the query.
2.  `needed_expansion_dimensions`: Decide which **missing** dimensions are most crucial to supplement from the user's history to fully understand their intent. Choose 2 to 4 of the most important ones.

---
[EXAMPLES]

-   **Input Query**: "affordable Halife brand dress with pockets"
-   **Your Output**:
    {{
      "explicitly_mentioned_dimensions": ["price_range", "brand", "features"],
      "needed_expansion_dimensions": ["style", "material", "quality_tier"]
    }}

-   **Input Query**: "something for my back pain"
-   **Your Output**:
    {{
      "explicitly_mentioned_dimensions": [],
      "needed_expansion_dimensions": ["brand", "price_range", "features", "quality_tier"]
    }}
---

Now, process the following request. Return only the JSON object.

**User's Current Query**: "{{current_query}}"
"""
    prompt = prompt.replace('current_query', instruction)
    prompt = prompt.replace('PREFERENCE_DIMENSIONS', json.dumps(PREFERENCE_DIMENSIONS, indent=2, ensure_ascii=False))
    messages = [{'role': 'system', 'content': prompt}]    
    #print(prompt)
    client = OpenAI(
        api_key = os.getenv("OPENAI_API_KEY"),
        base_url="https://xiaoai.plus/v1")
    # response = client.chat.completions.create(
    #     model='gpt-4o-mini',
    #     messages=messages,
    #     temperature=0,
    # )
    # message = response.choices[0].message.content
    #print(message)
    last_exception = None

    max_retries=3
    timeout=60.0
    for attempt in range(max_retries):
        try:
            # a. 发起 API 调用
            response = client.chat.completions.create(
                model='gpt-4o-mini',
                messages=messages,
                temperature=0,
                response_format={"type": "json_object"}, # 再次强调需要JSON输出
                timeout=timeout
            )
            
            # b. 提取返回的字符串
            content_string = response.choices[0].message.content
            
            # c. 尝试解析 JSON
            # 这里的 json.loads() 已经足够健壮，因为 response_format 会保证返回的是合法的 JSON 字符串
            parsed_dict = json.loads(content_string)
            
            # d. 成功！返回解析后的字典
            return parsed_dict

        except RateLimitError as e:
            wait_time = (2 ** attempt) + np.random.uniform(0, 1) # 增加随机抖动
            print(f"  [Warning] Attempt {attempt + 1}/{max_retries}: Rate limit hit. Retrying in {wait_time:.2f}s...")
            last_exception = e
            time.sleep(wait_time)
            
        except APIError as e:
            wait_time = (2 ** (attempt + 1)) + np.random.uniform(0, 1)
            print(f"  [Warning] Attempt {attempt + 1}/{max_retries}: API Error (e.g., server busy). Retrying in {wait_time:.2f}s...")
            last_exception = e
            time.sleep(wait_time)
            
        except Exception as e:
            # 捕获包括超时、连接错误在内的所有其他异常
            wait_time = (2 ** attempt) + np.random.uniform(0, 1)
            print(f"  [Warning] Attempt {attempt + 1}/{max_retries}: An unexpected error occurred: {e}. Retrying in {wait_time:.2f}s...")
            last_exception = e
            time.sleep(wait_time)

    # --- 3. 所有重试失败后的处理 ---
    print(f"  [Error] 'prefer_analyse_function' failed after {max_retries} retries.")
    print(f"  [Error] Last exception: {last_exception}")
    
    # 返回一个安全的空字典，确保下游代码不会崩溃
    return {}

def extracted_preferences_function(history, needed_dimensions):
    prompt = """
You are a precise user preference extraction assistant. Your task is to analyze a user's relevant shopping history and extract preferences **only** for the specified dimensions.

---
[BACKGROUND]

1.  **User's Most Relevant Past Behaviors**:
    {{relevant_history}}

2.  **Required Preference Dimensions to Extract**:
    {{needed_dimensions}}

---
[YOUR TASK]

Carefully read the user's history and summarize their preferences, but **only for the dimensions listed above**. Return the result as a single JSON object where keys are the dimension names. If you cannot find a clear preference for a specific dimension, omit it from the output.

---
[EXAMPLE]

-   **Input History**: ["Purchased: Halife T-Shirt Dress", "Reviewed: 'fabric is soft and breathable'"]
-   **Input Needed Dimensions**: ["style", "material"]
-   **Your Output**:
    {{
      "style": "Prefers Casual and T-Shirt Dress styles.",
      "material": "Values soft and breathable fabrics."
    }}
---

Now, process the following request. Return only the JSON object.

**User's Most Relevant Past Behaviors**: {{relevant_history}}
**Required Preference Dimensions to Extract**: {{needed_dimensions}}
"""
    history_for_prompt = json.dumps(history, indent=2, ensure_ascii=False)
    prompt = prompt.replace('relevant_history', history_for_prompt)
    prompt = prompt.replace('needed_dimensions', json.dumps(needed_dimensions, indent=2, ensure_ascii=False))
    messages = [{'role': 'system', 'content': prompt}]
    #print(prompt)
    client = OpenAI(
        api_key = os.getenv("OPENAI_API_KEY"),
        base_url="https://xiaoai.plus/v1")
    # response = client.chat.completions.create(
    #     model='gpt-4o-mini',
    #     messages=messages,
    #     temperature=0,
    # )
    # message = response.choices[0].message.content
    #print(message)
    try:
        response = client.chat.completions.create(
            model='gpt-4o-mini',
            messages=messages,
            temperature=0,
        )
        message = response.choices[0].message.content
    except Exception as e:
        print(f"[OpenAI ERROR] {e}")
        message = "<API调用失败>"

    try:
        # 将 LLM 返回的 JSON 字符串解析为 Python 字典
        parsed_dict = json.loads(message)
        
        # 返回一个解析好的 Python 字典
        return parsed_dict
        
    except json.JSONDecodeError:
        # 如果解析失败，返回一个空字典，下游函数可以安全地处理
        print(f"  [Warning] 'prefer_analyse_function' 无法解析LLM的返回结果。")
        return {}
    return message

def construct_final_query(original_query, extracted_preferences):
    """Constructs the final augmented query string."""
    if not extracted_preferences: return original_query
    expansion_parts = [f"Preference for {key.replace('_', ' ')}: {value}" for key, value in extracted_preferences.items()]
    expansion_text = ". ".join(expansion_parts)
    return f"{original_query}. [User Profile Hint: {expansion_text}]"

def format_history_function(history_objects):
    """
    Takes a list of complex history dictionary objects and converts them
    into a list of simple, descriptive strings for processing.
    """
    formatted_strings = []
    for item in history_objects:
        title = item.get("product_info", {}).get("title", "N/A")
        review_price = item.get("product_info", {}).get("price", "N/A")
        review_text = item.get("review", {}).get("text", "")
        review_rating = item.get("review", {}).get("rating", "N/A")

        # 构建一条简洁但信息丰富的字符串
        # "Purchased Product: [Title] | User Review ([Rating]/5): [Review Text]"
        entry = f"Purchased Product: {title} | Price: {review_price}"
        if review_text:
            entry += f" | User Review ({review_rating}/5): {review_text}"
        
        formatted_strings.append(entry)
    return formatted_strings