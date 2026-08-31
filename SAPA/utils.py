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


    labels[:output_start_index] = [-100] * output_start_index


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


        prefix_text = item['prompt']
        target_text = item['target']
        return preprocess_function_llama3(prefix_text, target_text, self.tokenizer, self.max_length)

def preprocess_function_llama3(prefix_text, target_text, tokenizer, max_length):


    full_prompt = prefix_text
    full_response = f"{target_text}<|eot_id|>"


    prefix_ids = tokenizer.encode(full_prompt, add_special_tokens=False)
    target_ids = tokenizer.encode(full_response, add_special_tokens=False)


    input_ids = prefix_ids + target_ids
    if len(input_ids) > max_length:

        input_ids = input_ids[:max_length]


    mask_len = len(prefix_ids)
    labels = [-100] * mask_len + input_ids[mask_len:]


    padding_len = max_length - len(input_ids)
    if padding_len > 0:
        input_ids += [tokenizer.pad_token_id] * padding_len
        labels += [-100] * padding_len


    attention_mask = [1] * (max_length - padding_len) + [0] * padding_len

    return {
        "input_ids": torch.tensor(input_ids[:max_length], dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask[:max_length], dtype=torch.long),
        "labels": torch.tensor(labels[:max_length], dtype=torch.long),
    }


def preprocess_function(prefix_text, target_text, tokenizer, max_length):

    global TRUNCATION_COUNT


    response_text = f"{prefix_text}{target_text}</s>"


    raw_tokenized = tokenizer(response_text, return_tensors=None, truncation=False)
    raw_len = len(raw_tokenized["input_ids"])

    if raw_len > max_length:
        TRUNCATION_COUNT += 1


        prefix_len = len(tokenizer(prefix_text, truncation=False)["input_ids"])
        target_status = "Target PARTIAL"
        if prefix_len >= max_length:
            target_status = "Target LOST (!!!)"


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


    output_start_index = len(prefix_tex)


    labels[:output_start_index] = [-100] * output_start_index

    return {
        "input_ids": input_ids,
        "attention_mask": input["attention_mask"],
        "labels": labels,
    }

def preprocess_function_modified(prefix_text, target_text, tokenizer, max_length):
    'Documentation.'


    response_text = f"{prefix_text}{target_text}</s>"


    tokenized_full = tokenizer(
        response_text,
        return_tensors=None,
        max_length=max_length,
        truncation=True,
        add_special_tokens=True
    )

    input_ids = tokenized_full["input_ids"]
    attention_mask = tokenized_full["attention_mask"]


    labels = input_ids.copy()


    tokenized_prefix = tokenizer(
        prefix_text,
        return_tensors=None,
        max_length=max_length,
        truncation=True,
        add_special_tokens=True
    )

    input_prefix_len = len(tokenized_prefix["input_ids"])


    mask_len = min(input_prefix_len, len(input_ids))


    labels[:mask_len] = [-100] * mask_len


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

    data = json.load(open(data_file, encoding="utf-8"))

    tasks = []
    source = []
    target = []


    if isinstance(data, dict) and 'test' in data:
        data_list = data['test']
    elif isinstance(data, list):
        data_list = data
    else:

        data_list = data.get('train', [])

    for item in data_list:

        prompt_text = item.get('prompt')
        if not prompt_text: continue


        import re
        match = re.search(r"Tool:\s*(\w+)", prompt_text)
        if not match: continue
        tool_name = match.group(1)

        task_type = ''
        if 'search' in tool_name: task_type = 'search'
        elif 'recommend' in tool_name: task_type = 'recommend'
        elif 'review' in tool_name: task_type = 'review'


        if task_type != split:
            continue


        if task_type == 'recommend':

            if not prompt_text.strip().endswith("ASIN:"):

                if not prompt_text.endswith('\n'):
                    prompt_text += '\n'
                prompt_text += "ASIN:"

        tasks.append(item.get('instruction', ''))
        source.append(prompt_text)
        target.append(item['target'])

    print(f"Loaded {len(source)} tasks for split {split}")
    return tasks, source, target

def load_param_prompt_beam_search_origin(data_file, tool_file, split, mem_token_length, tokenizer):

    data = json.load(open(data_file, encoding="utf-8"))
    tool_file = json.load(open(tool_file))
    tasks = []
    source = []
    target = []


    if 'test' not in data:
        raise KeyError("未在 data.json 文件中找到 'test' 数据分割。请确认数据格式。")


    for item in data['test']:


        prompt_text = item['prompt']
        match = re.search(r"Tool:\s*(\w+)", prompt_text)

        if not match:

            continue

        tool_name = match.group(1)


        task_type = ''
        if 'search' in tool_name:
            task_type = 'search'
        elif 'recommend' in tool_name:
            task_type = 'recommend'
        elif 'review' in tool_name:
            task_type = 'review'
        else:

            continue


        if task_type == split:

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


def load_param_prompt_beam_search_split_recommend_no_cat(data_file, tool_file, split, mem_token_length, tokenizer):

    data = json.load(open(data_file, encoding="utf-8"))


    tasks = []
    source = []
    target = []


    target_split = 'test'
    print(f"Loading data from split: {target_split} for task type: {split}")

    for item in data[target_split]:


        prompt_text = item['prompt']


        current_task_type = ''
        if 'search_product_by_query' in prompt_text:
            current_task_type = 'search'
        elif 'get_recommendations_by_history' in prompt_text:
            current_task_type = 'recommend'
        elif 'add_product_review' in prompt_text:
            current_task_type = 'review'


        if current_task_type != split:
            continue


        task = item['instruction']


        input_text = item['prompt']


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

        loss = loss_fct(shift_logits, shift_labels)


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

        torch.cuda.empty_cache()
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

    request_embedding = encode_texts([request], model, tokenizer)


    candidates_text = history


    history_embeddings = encode_texts(candidates_text, model, tokenizer)


    similarity = F.cosine_similarity(request_embedding, history_embeddings, dim=1)


    real_k = min(k, len(history))
    top_k_indices = similarity.argsort(descending=True)[:real_k]


    results = []
    for idx in top_k_indices:
        idx = idx.item()
        score = similarity[idx].item()
        content = history[idx]
        results.append((content, score))

    torch.cuda.empty_cache()


    return results

def retrieve_top_k_memories_formatted(request, history, model, tokenizer, k=50):
    request_embedding = encode_texts([request], model, tokenizer)

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

        info = item.get('product_info')
        if not isinstance(info, dict):
            continue


        if task_type == 'search':

            title = info.get('title', 'N/A')
            category = info.get('main_category', 'N/A')
            price = info.get('price', 'N/A')
            store = info.get('store', 'N/A')

            mem_string = f"Title: {title} | Category: {category} | Price: ${price} | Store: {store}"
            mem.append(mem_string)

        elif task_type == 'recommend':

            title = info.get('title', 'N/A')
            category = info.get('main_category', 'N/A')

            asin = info.get('parent_asin', 'N/A')
            mem_string = f"Title: {title} | Category: {category} | ASIN: {asin}"
            mem.append(mem_string)

        elif task_type == 'review':
            user_rating = info.get('rating', 'N/A')

            review_text = info.get('text', '').replace('\n', ' ')


            title = info.get('title', 'N/A')


            mem_string = f"Product: {title} | User Rating: {user_rating} | Review: {review_text[:200]}"
            mem.append(mem_string)

        else:

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

    client = OpenAI(
        api_key = os.getenv("OPENAI_API_KEY"),
        base_url="https://xiaoai.plus/v1")


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

    client = OpenAI(
        api_key = os.getenv("OPENAI_API_KEY"),
        base_url="https://xiaoai.plus/v1")


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

    client = OpenAI(
        api_key = os.getenv("OPENAI_API_KEY"),
        base_url="https://xiaoai.plus/v1")


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
    'Documentation.'
    target_title = target_product_info.get('title', '')
    target_category = target_product_info.get('main_category', '')
    target_asin = target_product_info.get('parent_asin', '')


    candidate_items = [
        item for item in user_history
        if item['product_info'].get('main_category') == target_category
    ]

    if not candidate_items:
        return target_asin


    candidate_titles = [item['product_info'].get('title', '') for item in candidate_items]


    with torch.no_grad():
        target_embedding = sim_model(**sim_tokenizer(target_title, return_tensors='pt', truncation=True).to(sim_model.device)).pooler_output
        candidate_embeddings = sim_model(**sim_tokenizer(candidate_titles, return_tensors='pt', padding=True, truncation=True).to(sim_model.device)).pooler_output


    cosine_scores = F.cosine_similarity(target_embedding, candidate_embeddings, dim=1)


    scored_candidates = sorted(list(zip(cosine_scores, candidate_items)), key=lambda x: x[0], reverse=True)


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


STOP_WORDS = {'&', 'for', 'the', 'a', 'in', 'of', 'with', 'is', 'an', 'to', 'by', 'on', 'pack', 'of', 'for'}

def extract_entities_and_keywords(title: str, brand: str = None) -> (dict, list):
    'Documentation.'
    entities = {'BRAND': set(), 'PRODUCT': set(), 'ATTRIBUTE': set()}
    if brand:
        entities['BRAND'].add(brand)


    model_pattern = r'\b([A-Z0-9]{2,}-[A-Z0-9]{2,}|[A-Z]{2,}\d{2,})\b'
    size_pattern = r'\b(\d+(\.\d+)?\s?(inch|oz|gb|mb|tb|cm|mm))\b'

    models = re.findall(model_pattern, title, re.IGNORECASE)
    sizes = re.findall(size_pattern, title, re.IGNORECASE)

    for m in models:
        entities['PRODUCT'].add(m)
    for s in sizes:
        entities['ATTRIBUTE'].add(s[0])


    keywords = []
    if nlp:
        doc = nlp(title)

        for token in doc:
            if token.pos_ in ('NOUN', 'PROPN') and token.text.lower() not in STOP_WORDS:

                if token.text not in entities['PRODUCT'] and token.text not in entities['BRAND']:
                    keywords.append(token.text)
    else:
        keywords = [word for word in title.replace(',', ' ').split() if word.lower() not in STOP_WORDS]


    for key in entities:
        entities[key] = list(entities[key])

    return entities, keywords


def prefer_analyse_function(instruction):
    PREFERENCE_DIMENSIONS = [
        "style",
        "features",
        "brand",
        "material",
        "quality_tier",
        "price_range",
        "color_palette",
        "values"
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

    client = OpenAI(
        api_key = os.getenv("OPENAI_API_KEY"),
        base_url="https://xiaoai.plus/v1")


    last_exception = None

    max_retries=3
    timeout=60.0
    for attempt in range(max_retries):
        try:

            response = client.chat.completions.create(
                model='gpt-4o-mini',
                messages=messages,
                temperature=0,
                response_format={"type": "json_object"},
                timeout=timeout
            )


            content_string = response.choices[0].message.content


            parsed_dict = json.loads(content_string)


            return parsed_dict

        except RateLimitError as e:
            wait_time = (2 ** attempt) + np.random.uniform(0, 1)
            print(f"  [Warning] Attempt {attempt + 1}/{max_retries}: Rate limit hit. Retrying in {wait_time:.2f}s...")
            last_exception = e
            time.sleep(wait_time)

        except APIError as e:
            wait_time = (2 ** (attempt + 1)) + np.random.uniform(0, 1)
            print(f"  [Warning] Attempt {attempt + 1}/{max_retries}: API Error (e.g., server busy). Retrying in {wait_time:.2f}s...")
            last_exception = e
            time.sleep(wait_time)

        except Exception as e:

            wait_time = (2 ** attempt) + np.random.uniform(0, 1)
            print(f"  [Warning] Attempt {attempt + 1}/{max_retries}: An unexpected error occurred: {e}. Retrying in {wait_time:.2f}s...")
            last_exception = e
            time.sleep(wait_time)


    print(f"  [Error] 'prefer_analyse_function' failed after {max_retries} retries.")
    print(f"  [Error] Last exception: {last_exception}")


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

    client = OpenAI(
        api_key = os.getenv("OPENAI_API_KEY"),
        base_url="https://xiaoai.plus/v1")


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

        parsed_dict = json.loads(message)


        return parsed_dict

    except json.JSONDecodeError:

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


        entry = f"Purchased Product: {title} | Price: {review_price}"
        if review_text:
            entry += f" | User Review ({review_rating}/5): {review_text}"

        formatted_strings.append(entry)
    return formatted_strings
