
import json
from typing import Dict, List

from openai import OpenAI
from PersonalWAB.agents.base import BaseAgent
from PersonalWAB.agents.utils import (
    message_to_action,
    message_to_dict,
    pretty_print_conversation,
    pretty_history,
    encode_texts,
    HISTORY_PROMPT,
    MINI_HISTORY_PROMPT,
    INTEREC_MEMORY_PROMPT,
    RECMIND_ST_PROMPT,
    RECMIND_MT_PROPMT,
    INTEREC_PROMPT,
    INTEREC_UPDATE_MEM_PROMPT,
    TS_AGENT_PROMPT,
    TS_AGENT_MT_PROMPT,
    sup_search_pretty_history,
    sup_rec_pretty_history,
    sup_review_pretty_history,
    interecagent_pretty_history,
    mini_pretty_history,
    PARAM_PROMPT_SEARCH,      # Search 任务的模板
    PARAM_PROMPT_RECOMMEND,   # Recommend 任务的模板
    PARAM_PROMPT_REVIEW,      # Review 任务的模板
    retrieve_top_k_memories_score, # 你脚本里用到的那个带分数的检索函数
    build_taskspe_memory,     # 格式化函数
    prettify_product_info
)
from tenacity import retry, stop_after_attempt, wait_random_exponential
import sys
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import random
import faiss
from pyserini.search.lucene import LuceneSearcher
from sentence_transformers import CrossEncoder
import numpy as np
import copy
import os
from pathlib import Path

client = None
prompt_price_per_million = {
    "gpt-4o-mini": 0.15,
    "gpt-4o": 5,
    "gpt-4-turbo": 10,
    "gpt-4-32k-0613": 60,
    "gpt-3.5-turbo": 0.5,
    "meta-llama/Meta-Llama-3-8B-Instruct": 0.15,
    "meta-llama/Meta-Llama-3-70B-Instruct": 1.0,
}
completion_price_per_million = {
    "gpt-4o-mini": 0.60,
    "gpt-4o": 15,
    "gpt-4-turbo": 30,
    "gpt-4-32k-0613": 120,
    "gpt-3.5-turbo": 1.5,
    "meta-llama/Meta-Llama-3-8B-Instruct": 0.15,
    "meta-llama/Meta-Llama-3-70B-Instruct": 1.0,
}

def initialize_client(**kwargs):
    global client
    client = OpenAI(**kwargs)


@retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(10))
def chat_completion_request(
    messages: List[Dict[str, str]],
    model: str,
    tools=None,
    tool_choice="auto",
    temperature: float = 0.0,
):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        tool_choice=tool_choice,
        temperature=temperature,
    )
    message = response.choices[0].message
    if hasattr(message, "tool_calls") and message.tool_calls is not None:
        tool_call = message.tool_calls[0]
        json.loads(tool_call.function.arguments)
    return message, dict(response.usage)


class RISEAgent(BaseAgent):
    def __init__(self, tools_info, sys_prompt, model: str = "gpt-4o-mini", function_selection_file=None, memory_file=None, tts_n=10):
        self.tools = tools_info
        self.sys_prompt = sys_prompt
        self.model = model
        self.function_selection_file = json.load(open(function_selection_file)) if function_selection_file is not None else None
        self.memory_file = memory_file if memory_file is not None else None
        self.interecagent_memory = json.load(open(memory_file)) if memory_file is not None else None
        self.usage = {"completion_tokens": [], "prompt_tokens": [], "total_tokens":[]}
        self.tts_n = tts_n
        
        self.reset()

    def reset(self, memory='none'):
        self.messages = [{"role": "system", "content": self.sys_prompt}]
        if memory != 'none':
            self.messages = [{"role": "system", "content": self.sys_prompt + memory}]
        self.usage = {"completion_tokens": [], "prompt_tokens": [], "total_tokens": []}

    def act(self, env, index=None, verbose=False, temperature=0.0, max_steps=30, memory='none', memory_length=10):
        task_data = env.tasks[index]
        task_instruction = env.tasks[index]["task"]
        tool_name = self.function_selection_file[task_instruction][0]
        if tool_name == 'search_product_by_query':
            self.current_task_type = 'search'
        elif tool_name == 'get_recommendations_by_history':
            self.current_task_type = 'recommend'
        elif tool_name == 'add_product_review':
            self.current_task_type = 'review'
        else:
            self.current_task_type = 'search'

        if memory != 'none':
            memory_content = self.retrieve_memory(env, index, memory, memory_length)
        else:
            memory_content = 'none'

        if memory == 'taskspe_rise' and max_steps == 1:
            # 【单轮模式】：强制使用结构化的填空模板
            if self.current_task_type == 'recommend':
                # --- 推荐任务专属的候选词获取 ---
                task_id_key = f"{task_data['user_id']}_{task_data['timestamp']}"
                task_record = global_retrieval_map.get(task_id_key, {})
                raw_candidates = task_record.get('search_result', {}).get('top_500_results',[])[:10]
                
                candidate_strings =[]
                for i, cand in enumerate(raw_candidates):
                    asin = cand.get('asin')
                    details = global_asin_to_info.get(asin, "No detailed information.")
                    candidate_strings.append(f"[{i+1}] ASIN: {asin}\n{details}")
                candidates_prompt_text = "\n\n".join(candidate_strings) if candidate_strings else "No candidates."

                # 填充 Recommend 模板
                self.sys_prompt = PARAM_PROMPT_RECOMMEND.replace("<Instruction>", task_instruction) \
                                                        .replace("<History>", memory_content) \
                                                        .replace("<Candidates>", candidates_prompt_text) \
                                                        .replace("<Tool>", tool_name)
            
            elif self.current_task_type == 'search':
                # 填充 Search 模板
                self.sys_prompt = PARAM_PROMPT_SEARCH.replace("<Instruction>", task_instruction) \
                                                     .replace("<Memory>", memory_content) \
                                                     .replace("<Tool>", tool_name)
                                                     
            elif self.current_task_type == 'review':
                # 填充 Review 模板

                current_instruction = task_instruction
                
                # 从环境任务中提取 target 商品信息
                target_info = task_data.get('target', {})
                product_info = target_info.get('product_info')
                
                # 拼接上你训练时的 prettify_product_info
                if product_info:
                    current_instruction += prettify_product_info(product_info)
                self.sys_prompt = PARAM_PROMPT_REVIEW.replace("<Instruction>", task_instruction) \
                                                     .replace("<Memory>", memory_content) \
                                                     .replace("<Tool>", tool_name)

            # 【关键】：由于模板里已经包含了 <Memory> / <History> 占位符，
            # 这里的 reset 必须传 'none'，防止被重复拼接到 prompt 后面！
            self.reset(memory='none')

        else:
            # 【多轮模式】：保留你原来的通用 Agent 逻辑
            if memory == 'recmind':
                if max_steps == 1:
                    self.sys_prompt = RECMIND_ST_PROMPT
                else:
                    self.sys_prompt = RECMIND_MT_PROPMT.replace("<NUM>", str(max_steps))
            elif memory == 'interecagent':
                self.sys_prompt = INTEREC_PROMPT.replace("<NUM>", str(max_steps))
            elif memory == 'taskspe':
                if max_steps == 1:
                    self.sys_prompt = TS_AGENT_PROMPT
                else:
                    self.sys_prompt = TS_AGENT_MT_PROMPT.replace("<NUM>", str(max_steps))
            elif memory == 'taskspe_rise':
                self.sys_prompt = TS_AGENT_MT_PROMPT.replace("<NUM>", str(max_steps))

            # 多轮模式下，提示词模板里没有留空，所以这里依赖 reset 把 memory 拼在最后
            self.reset(memory_content)

        obs, info = env.reset(index=index)

        max_steps = max_steps if max_steps > 0 else 10
        action_acc = []
        res_acc = []

        # --- 🚀 关键：初始化记录变量 ---
        seen_seed_asins = set()

        self.messages.append({"role": "user", "content": obs})

        if verbose:
            self.render(1)
        for _ in range(max_steps):
            # 1. 动态生成候选
            candidates, step_usage = self.generate_candidates_by_task()
            
            # 定义一个基础 message 对象，用于后续的 ID 提取和历史记录
            # 我们默认借用第一个候选的“信封”（ID等元数据）
            message = candidates[0] # <--- 修改点 1: 确保 message 变量在循环作用域内可用

            for key in ['completion_tokens', 'prompt_tokens', 'total_tokens']:
                self.usage[key].append(step_usage.get(key, 0))
            
            # 2. 【核心】根据任务类型进行特化处理
            if self.current_task_type == 'search':
                action = self.process_search_task(candidates, env, index)
            elif self.current_task_type == 'recommend':
                action = self.process_recommend_task(candidates)
            elif self.current_task_type == 'review':
                action = self.process_review_task(candidates)
            else:
                action = message_to_action(message)

            # --- 🚀 关键：提取当前轮次的种子 ASIN ---
            current_seed_asin = None
            if self.current_task_type == 'recommend':
                # 从我们补全后的 action 中拿到第一个 ASIN
                asins = action.get("arguments", {}).get("product_sequence", [])
                if asins and isinstance(asins, list):
                    current_seed_asin = asins[0]
            # ---------------------------

            # 3. 把最终敲定的 action 发给环境
            obs, res, done, info_step = env.step(action)

            if action["name"] == "respond":
                self.messages.append({"role": "assistant", "content": message.content})
                self.messages.append({"role": "user", "content": obs})
            else:
                # <--- 修改点 2: 将 RISE 后处理后的 arguments 同步回消息对象
                # 这样 self.messages 里的记录就和 env 执行的动作完全一致了
                message.tool_calls = message.tool_calls[:1]
                message.tool_calls[0].function.arguments = json.dumps(action["arguments"]) # 强制同步
                
                self.messages.append(message)
                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": message.tool_calls[0].id,
                        "name": message.tool_calls[0].function.name,
                        "content": obs,
                    }
                )
            if verbose:
                self.render(2)

            if env.max_steps == -1 and done:
                action_acc.append(res[0])
                res_acc.append(res[1])
                break
            else:
                if done:
                    action_acc.append(res[0])
                    res_acc.append(res[1])
                if action["name"] == "stop":
                    break
                if self.current_task_type == 'recommend':
                    # A. 循环检测（防死循环，依然保留，这是合理的工程保护）
                    if current_seed_asin is not None:
                        if current_seed_asin in seen_seed_asins:
                            print(f"🚨 推荐任务检测到种子重复 (ASIN: {current_seed_asin})，为节省资源强制退出。")
                            break
                        else:
                            seen_seed_asins.add(current_seed_asin)
                
        if memory == 'interecagent':
            self.update_interecagent_memory(env, index, self.get_messages())
        
        self.usage.update(
            {"completion_price": [], "prompt_price": [], "total_price": []}
        )
        self.usage["completion_price"] = (
            completion_price_per_million[self.model]
            * sum(self.usage["completion_tokens"])
            / 1e6
        )
        self.usage["prompt_price"] = (
            prompt_price_per_million[self.model]
            * sum(self.usage["prompt_tokens"])
            / 1e6
        )
        self.usage["total_price"] = (
            self.usage["completion_price"] + self.usage["prompt_price"]
        )
        info["usage"] = self.usage
        return action_acc, res_acc, info
    
    def generate_candidates_by_task(self):
        """
        智能变频生成器：根据任务类型动态调整采样策略
        """
        from PersonalWAB.agents.gpt_function_calling_agent import client
        t_choice = "auto"
        current_tools = self.tools
        # 1. 根据你的要求设置参数
        if self.current_task_type in ['search', 'review']:
            temperature = 1.0
            n_samples = 10
        elif self.current_task_type == 'recommend':
            t_choice = {"type": "function", "function": {"name": "get_recommendations_by_history"}}
            temperature = 0.0
            n_samples = 1
            recommend_tool =[]
            for t in self.tools:
                if t['function']['name'] == 'get_recommendations_by_history':
                    t_copy = copy.deepcopy(t)
                    # 篡改工具描述，明确告诉 GPT 把答案填这里！
                    t_copy['function']['description'] = "Use this tool to submit your final recommendation. You MUST put the ASIN you selected from the Candidate List here."
                    t_copy['function']['parameters']['properties']['product_sequence']['description'] = "A list containing exactly ONE string: the ASIN you selected as the best recommendation."
                    recommend_tool.append(t_copy)
            
            # 替换当前使用的工具为定制版
            current_tools = recommend_tool 
        else:
            temperature = 0.0
            n_samples = 1

        # 2. 调用 API
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                tools=current_tools,
                tool_choice=t_choice,
                temperature=temperature,
                n=n_samples
            )
            return [choice.message for choice in response.choices], dict(response.usage)
        except Exception as e:
            print(f"API 调用出错: {e}")
            fallback_msg = type('obj', (object,), {'content': 'stop', 'tool_calls': None})
            return [fallback_msg], {"completion_tokens": 0, "prompt_tokens": 0, "total_tokens": 0}
        
    def process_review_task(self, candidates):
        """
        改进后的 Review 引擎：使用语义质心（Semantic Centroid）
        """
        agent_reviews_list = []
        valid_messages = []
        
        # 1. 提取出这 10 个候选里的评论文本
        for msg in candidates:
            # 假设你的 respond 逻辑里存的是评论内容
            if msg.content:
                agent_reviews_list.append(msg.content)
                valid_messages.append(msg)
        
        if not agent_reviews_list:
            return message_to_action(candidates[0])

        # 2. 调用你的非环境实现逻辑：寻找语义质心索引
        # 注意：这里的 sim_model 和 sim_tokenizer 是你在文件顶部定义的
        best_idx = self.get_consensus_best_index(agent_reviews_list, sim_tokenizer, sim_model)
        
        # 3. 返回那个最具有代表性的候选
        return message_to_action(valid_messages[best_idx])

    def get_consensus_best_index(self, sentences, tokenizer, model):
        """
        计算 10 条句子中，哪一条是语义上的‘最大公约数’
        """
        # 编码所有句子
        encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt').to('cuda')
        with torch.no_grad():
            model_output = model(**encoded_input)
        
        # Mean Pooling 得到句向量
        def mean_pooling(model_output, attention_mask):
            token_embeddings = model_output[0]
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        embeddings = F.normalize(embeddings, p=2, dim=1) # 归一化
        
        # 计算两两之间的相似度矩阵 (10x10)
        # cosine_sim_matrix[i][j] 表示第 i 条和第 j 条的相似度
        cosine_sim_matrix = torch.mm(embeddings, embeddings.t())
        
        # 计算每条句子与其他所有句子的相似度总和
        # 那个和别人最像的，就是‘共识’
        sum_scores = cosine_sim_matrix.sum(dim=1)
        best_idx = torch.argmax(sum_scores).item()
        
        return best_idx
    def process_recommend_task(self, candidates):
        """
        Recommend 引擎增强版：
        1. 拿到模型生成的 Top 1 ASIN。
        2. 在产品池中通过 Cosine 相似度检索最相似的 9 个邻居。
        3. 按照相似度从高到低排列，最大化 R.Acc 分数。
        """
        # # 模型在 Recommend 任务中 Temp=0, n=1，所以直接取第一个
        try:
            message = candidates[0]
            # 侦探点 1：检查 message 转换
            action = message_to_action(message)
            print(f"🕵️ 转换后的 action 名字: {action['name']}")

            # 侦探点 2：确保 arguments 是字典而不是字符串
            if isinstance(action["arguments"], str):
                action["arguments"] = json.loads(action["arguments"])
            
            # 侦探点 3：提取种子 ASIN
            raw_asins = action["arguments"].get("product_sequence", [])
            seed_asin = None
            if isinstance(raw_asins, list) and len(raw_asins) > 0:
                seed_asin = str(raw_asins[0]).strip()
            elif isinstance(raw_asins, str):
                seed_asin = raw_asins.strip()
            
            print(f"🕵️ 提取出的种子 ASIN: {seed_asin}")

            if seed_asin:
                # 侦探点 4：进入检索逻辑
                similar_9 = self.retrieve_similar_asins_by_cosine(seed_asin, top_k=9)
                print(f"🕵️ Cosine 检索返回了 {len(similar_9)} 个结果")
                
                final_10 = [seed_asin] + similar_9
                action["arguments"]["product_sequence"] = final_10[:10]
            
            return action

        except Exception as e:
            # 【重要修改】：这里加上 traceback，它会告诉你是哪一行报错了！
            import traceback
            print("❌ Recommend 引擎崩溃详细追踪：")
            traceback.print_exc() 
            return action # 保底返回

    def retrieve_similar_asins_by_cosine(self, target_asin, top_k=9):
        """
        利用 Faiss 索引进行真正的向量相似度检索
        """
        # 1. 拿文本
        target_text = global_asin_to_info.get(target_asin)
        if not target_text:
            print(f"⚠️ 库里找不到 ASIN {target_asin} 的详情，无法召回邻居")
            return []

        # 2. 算向量
        inputs = sim_tokenizer([target_text], padding=True, truncation=True, max_length=512, return_tensors='pt').to(sim_model.device)
        with torch.no_grad():
            model_output = sim_model(**inputs)
            token_embeddings = model_output[0]
            input_mask_expanded = inputs['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
            query_vector = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            query_vector = torch.nn.functional.normalize(query_vector, p=2, dim=1)
            
            # 【关键点！】：转为 float32 的 numpy 数组
            query_vector = query_vector.cpu().numpy().astype('float32')

        # 3. 搜索引
        D, I = global_vector_index.search(query_vector, top_k + 5) 
        
        # 4. 根据返回的索引行号 I，转换回 ASIN
        similar_asins = []
        for idx in I[0]:
            # 越过合法的索引范围判断
            if idx < 0 or idx >= len(global_all_asins):
                continue
            
            candidate_asin = global_all_asins[idx]
            
            # 排除掉自己，且只取前 top_k 个
            if candidate_asin != target_asin:
                similar_asins.append(candidate_asin)
            
            if len(similar_asins) >= top_k:
                break
                
        return similar_asins

    def process_search_task(self, candidates, env, index):
        """
        Search 引擎核心逻辑：
        1. 提取 10 个候选 Query。
        2. 多路 BM25 召回 (200条)。
        3. 利用原始指令 (Instruction) 进行 BGE 重排。
        """
        import json
        
        # 1. 提取所有生成的搜索词 (Queries)
        raw_queries = []
        for msg in candidates:
            action = message_to_action(msg)
            if action["name"] == "search_product_by_query":
                q = action["arguments"].get("query", "")
                if q: raw_queries.append(q)
        
        if not raw_queries:
            return message_to_action(candidates[0])

        # --- 🚀 关键修改点：展平并清洗 queries ---
        queries = []
        for q in raw_queries:
            if isinstance(q, list):
                # 如果是列表（比如 [ASIN1, ASIN2] 或 ["query"]），展平并转为字符串
                queries.extend([str(item) for item in q])
            else:
                # 如果是字符串或其他类型，转为字符串
                queries.append(str(q))
        
        # 使用 set(queries) 之前，现在可以确保里面全是字符串了
        unique_queries = list(set(queries))
        # ----------------------------------------

        # 获取原始长文本指令
        original_instruction = env.tasks[index]["task"]

        # 2. BM25 多路召回
        candidate_map = {} 
        for q in unique_queries: # 🚀 使用清洗后的唯一搜索词
            try:
                # 额外的安全保护：如果 q 还是奇怪的格式则跳过
                if not isinstance(q, str) or len(q.strip()) == 0:
                    continue
                    
                hits = global_bm25_searcher.search(q, k=20)
                for hit in hits:
                    if hit.docid not in candidate_map:
                        doc_obj = global_bm25_searcher.doc(hit.docid)
                        if doc_obj:
                            item = json.loads(doc_obj.raw())
                            content = item.get('contents', item.get('text', ""))
                            candidate_map[hit.docid] = {
                                "content": content,
                                "asin": item.get('asin', hit.docid)
                            }
            except Exception as e:
                print(f"BM25 Search Error for query '{q}': {e}")

        candidates_data = list(candidate_map.values())
        if not candidates_data:
            return message_to_action(candidates[0])

        # 3. BGE 重排序 (语义对齐)
        contents_list = [c["content"] for c in candidates_data]
        # 构造输入对: [[指令, 商品1], [指令, 商品2], ...]
        pairs = [[original_instruction, text] for text in contents_list]
        
        try:
            scores = global_reranker.predict(pairs, batch_size=32, show_progress_bar=False)
            # 将结果与分数绑定并排序
            ranked_results = sorted(zip(candidates_data, scores), key=lambda x: x[1], reverse=True)
            # 取 Top 10
            top_10_results = ranked_results[:10]
        except Exception as e:
            print(f"Rerank Error: {e}")
            top_10_results = [(c, 0) for c in candidates_data[:10]]

        # 4. 封装成 Action
        # 注意：这里我们把 Top 10 商品的 ASIN 列表塞进 query 参数里
        # 这样 env.step 拿到的就是你精选后的结果了
        final_asins = [res[0]["asin"] for res in top_10_results]
        
        # 我们借用第一个候选的结构
        final_action = message_to_action(candidates[0])
        final_action["arguments"]["query"] = final_asins # 此时 query 变成了一个列表
        
        return final_action
    
    def format_memory_code4_style(self, history_items, task_type):
        """Search/Review 专用格式化 (同步自你的脚本)"""
        mem_strs = []
        for item in history_items:
            if task_type == 'search':
                info = item.get('product_info', {})
                mem_strs.append(f"Title:{info.get('title', 'N/A')}\nMain Category:{info.get('main_category', 'N/A')}\nPrice:{info.get('price', 'N/A')}\nStore:{info.get('store', 'N/A')}")
            elif task_type == 'review':
                review = item.get('review', {})
                # 1. 先把文本取出来，并在外面做 replace 替换
                clean_text = review.get('text', '').replace('\n', ' ')
                # 2. 然后再放进 f-string 里
                mem_strs.append(f"Rating:{review.get('rating', 'N/A')}\nText:{clean_text}")
        return mem_strs

    def refresh_recommend_candidates_prompt(self, asins):
        """
        根据传入的 10 个 ASIN，动态生成用于 Prompt 的候选列表文本
        """
        candidate_strings = []
        for i, asin in enumerate(asins):
            # 从全局库里查详情
            details = global_asin_to_info.get(asin, "No detailed information available.")
            candidate_strings.append(f"[{i+1}] ASIN: {asin}\n{details}")
        
        return "\n\n".join(candidate_strings)

    def render(self, last_n=None):
        if last_n is not None:
            pretty_print_conversation(self.messages[-last_n:])
        else:
            pretty_print_conversation(self.messages)

    def get_messages(self) -> List[Dict[str, str]]:
        return [message_to_dict(message) for message in self.messages]
    
    def retrieve_memory(self, env, index, memory, memory_length):
        if memory == 'last':
            '''Short Memory of This User (most recent purchases and reviews)'''
            timestamp = env.tasks[index]["timestamp"]
            history = env.init_data["user_history"][env.tasks[index]["user_id"]]
            history = [item for item in history if item['review']["timestamp"] < timestamp]
            history = sorted(history, key=lambda x: x['review']["timestamp"], reverse=True) 
            history = history[:memory_length]
            mem = '\nYour Memory of This User (most recent user past purchases and reviews, sorted by timestamp, most recent first):\n\n'
            mem = mem + "\n".join([pretty_history(item, i) for i, item in enumerate(history)])
            return mem
        
        elif memory == 'relevant':
            '''Relevant Memory of This User (most relevant purchases and reviews)'''
            timestamp = env.tasks[index]["timestamp"]
            history = env.init_data["user_history"][env.tasks[index]["user_id"]]
            history = [item for item in history if item['review']["timestamp"] < timestamp]
            history = self.retrieve_top_k_memories(env.tasks[index]['task'], history, sim_model, sim_tokenizer, k=memory_length)
            mem = '\nYour Memory of This User (most relevant user past purchases and reviews, sorted by relevance):\n\n'
            mem = mem + "\n".join([pretty_history(item, i) for i, item in enumerate(history)])
            return mem
        
        elif memory == 'random':
            '''Random Memory of This User (random purchases and reviews)'''
            timestamp = env.tasks[index]["timestamp"]
            history = env.init_data["user_history"][env.tasks[index]["user_id"]]
            history = [item for item in history if item['review']["timestamp"] < timestamp]
            history = sorted(history, key=lambda x: x['review']["timestamp"], reverse=True)
            history = random.sample(history, min(memory_length, len(history)))
            mem = '\nYour Memory of This User (random chosed user past purchases and reviews):\n\n'
            mem = mem + "\n".join([pretty_history(item, i) for i, item in enumerate(history)])
            return mem
        
        elif memory == 'recmind':
            '''RecMind Memory (all purchases and reviews)'''
            timestamp = env.tasks[index]["timestamp"]
            history = env.init_data["user_history"][env.tasks[index]["user_id"]]
            history = [item for item in history if item['review']["timestamp"] < timestamp]
            history = sorted(history, key=lambda x: x['review']["timestamp"], reverse=True)[:memory_length]
            mem = '\nPersonalized Memory of This User (all user past purchases and reviews):\n\n'
            mem = mem + "\n".join([mini_pretty_history(item, i) for i, item in enumerate(history)])
            return mem

        elif memory == 'interecagent':
            '''InterecAgent Memory (history, like, dislike and expect)'''
            timestamp = env.tasks[index]["timestamp"]
            history = env.init_data["user_history"][env.tasks[index]["user_id"]]
            history = [item for item in history if item['review']["timestamp"] < timestamp]
            history = sorted(history, key=lambda x: x['review']["timestamp"], reverse=True)[:memory_length]
            history = [interecagent_pretty_history(item) for item in history]
            assert self.interecagent_memory is not None
            mem = "\n\nYour Memory of This User:\nHistory:\n" 
            mem = mem + "\n".join(history) 
            mem = mem + "\nLike:\n" + str(self.interecagent_memory[env.tasks[index]["user_id"]]["like"])
            mem = mem + "\nDislike:\n" + str(self.interecagent_memory[env.tasks[index]["user_id"]]["dislike"])
            mem = mem + "\nExpect:\n" + str(self.interecagent_memory[env.tasks[index]["user_id"]]["expect"])
            return mem
    
        elif memory == 'taskspe':
            '''Task-specific Memory'''
            timestamp = env.tasks[index]["timestamp"]
            assert self.function_selection_file is not None
            tool_name = self.function_selection_file[env.tasks[index]["task"]][0] 
            if tool_name == 'search_product_by_query':
                task_type = 'search'
            elif tool_name == 'get_recommendations_by_history':
                task_type = 'recommend'
            elif tool_name == 'add_product_review':
                task_type = 'review'   
            
            history = env.init_data["user_history"][env.tasks[index]["user_id"]]
            history = [item for item in history if item['review']["timestamp"] < timestamp]
            if len(history) == 0:
                return 'none'
            history = self.retrieve_top_k_memories(env.tasks[index]['task'], history, sim_model, sim_tokenizer, k=memory_length)
            history = build_taskspe_memory(history, task_type)
            
            mem = f'- The task type is {task_type}\n- Use {tool_name} tool to complete task\n'
            mem = mem + '- Task-specific Memory:\n'
            mem = mem + "|".join([item for item in history])
            return mem
        
        elif memory == 'taskspe_rise':
            # 1. 基本信息准备
            timestamp = env.tasks[index]["timestamp"]
            user_id = env.tasks[index]["user_id"]
            tool_name = self.function_selection_file[env.tasks[index]["task"]][0]
            
            # 识别任务类型
            if tool_name == 'search_product_by_query':
                task_type = 'search'
                k_value = 6  # <--- 你的要求：Search 取 Top-6
            elif tool_name == 'get_recommendations_by_history':
                task_type = 'recommend'
                k_value = 1  # <--- 你的要求：Recommend 取 Top-1
            else:
                task_type = 'review'
                k_value = memory_length # Review 保持默认

            # 2. 获取该用户的历史记录
            raw_history = env.init_data["user_history"][user_id]
            # 过滤掉未来的数据
            history = [item for item in raw_history if item['review']["timestamp"] < timestamp]
            
            if len(history) == 0:
                return 'No purchase history.'

            # 3. 执行检索 (使用你脚本里的逻辑)
            if task_type == 'recommend':
                # 【修改点】：先将字典列表转换为字符串列表
                history_as_strings = build_taskspe_memory(history, task_type)
                
                # 现在传入的是字符串列表，不会再报错了
                retrieval_res = retrieve_top_k_memories_score(
                    env.tasks[index]['task'], history_as_strings, sim_model, sim_tokenizer, k=1
                )
                
                # 检查分数阈值：小于 0.5 舍弃
                if retrieval_res and retrieval_res[0][1] >= 0.5:
                    # retrieval_res[0][0] 已经是格式化好的字符串了
                    mem_str = f"- {retrieval_res[0][0]}"
                else:
                    mem_str = "No relevant purchase history available."
            
            else:
                # Search 和 Review 逻辑
                relevant_items = self.retrieve_top_k_memories(
                    env.tasks[index]['task'], history, sim_model, sim_tokenizer, k=k_value
                )
                # 使用你脚本里的 format_memory_code4_style
                mem_strs = self.format_memory_code4_style(relevant_items, task_type)
                memory_text_raw = ' | '.join(mem_strs) if mem_strs else "No purchase history."

                if llama_tokenizer is not None and memory_text_raw != "No purchase history.":
                    tokenized_memory = llama_tokenizer(
                        memory_text_raw, return_tensors=None, truncation=True, max_length=768
                    )
                    mem_str = llama_tokenizer.decode(tokenized_memory["input_ids"], skip_special_tokens=True)
                else:
                    # 对于 GPT-4o-mini 或没加载 tokenizer 的情况，按字符粗略截断防止过长
                    mem_str = memory_text_raw[:2600]

            return mem_str

        else:
            return 'none'
    
    
    def update_interecagent_memory(self, env, index, messages):
        assert self.interecagent_memory is not None
        assert self.memory_file is not None
        user_id = env.tasks[index]["user_id"]
        prompt = INTEREC_UPDATE_MEM_PROMPT
        actions = ""
        for mes in messages[1:]:
            if mes["role"] == "user"or mes["role"] == "tool":
                actions += mes["role"] +':'+ mes["content"] + "\n"
            elif mes["role"] == "assistant":
                if 'function_call' in mes:
                    actions += 'assistant:' + mes["function_call"] + "\n"
                else:
                    actions += 'assistant:' + mes["content"] + "\n"
        prompt = prompt.replace("{conversation}", actions)

        client = OpenAI()

        profile = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": prompt}],
        ).choices[0].message.content
        profile = profile.replace("```json", "").replace(
                "```", "").replace("\n", "")
        profile = json.loads(profile)

        cur_memory = self.interecagent_memory[user_id]
        like_set = set(cur_memory['like'])
        dislike_set = set(cur_memory['dislike'])
        expect_set = set(cur_memory['expect'])
        like_set -= set(profile.get('dislike', []))
        like_set.update(profile.get('like', []))
        dislike_set -= set(profile.get('like', []))
        dislike_set.update(profile.get('dislike', []))
        expect_set.update(profile.get('expect', []))
        if len(expect_set) > 30:
            expect_set = set(list(expect_set)[-30:])
        self.interecagent_memory[user_id] = {
            'like': list(like_set),
            'dislike': list(dislike_set),
            'expect': list(expect_set)
        }
        with open(self.memory_file, "w") as f:
            json.dump(self.interecagent_memory, f, indent=2)
        
        
    def retrieve_top_k_memories(self, request, history, model, tokenizer, k=50):
        '''Retrieve top k relevant memories'''
        request_embedding = encode_texts([request], model, tokenizer)
        history_embeddings = encode_texts([pretty_history(item, i) for i, item in enumerate(history)], model, tokenizer)
        similarity = F.cosine_similarity(request_embedding, history_embeddings, dim=1)
        top_k = similarity.argsort(descending=True)[:k]
        del request_embedding, history_embeddings, similarity
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return [history[i] for i in top_k]

sim_model_name = os.getenv('PWAB_SIM_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
sim_tokenizer = AutoTokenizer.from_pretrained(sim_model_name)
sim_model = AutoModel.from_pretrained(sim_model_name)
sim_model.eval()
if torch.cuda.is_available():
    sim_model.to('cuda')

try:
    llama_tokenizer = AutoTokenizer.from_pretrained(
        os.getenv('PWAB_BASE_MODEL', 'meta-llama/Llama-2-7b-chat-hf')
    )
    print("✅ LLaMA Tokenizer 加载成功（用于精确截断 Memory）！")
except Exception as e:
    print(f"⚠️ LLaMA Tokenizer 加载失败，将降级为字符长度截断: {e}")
    llama_tokenizer = None

# 2. BGE 重排模型 (用于 Search 精排)
try:
    global_reranker = CrossEncoder(
        os.getenv('PWAB_RERANKER_MODEL', 'BAAI/bge-reranker-v2-m3'), device='cuda'
    )
    print("✅ 全局 BGE Reranker 加载成功！")
except Exception as e:
    print(f"❌ 全局 Reranker 加载失败: {e}")
    global_reranker = None

# 3. Faiss 和 BM25 索引
repo_root = Path(__file__).resolve().parents[2]
vector_index_path = os.getenv(
    'PWAB_VECTOR_INDEX',
    str(repo_root / 'PersonalWAB/envs/pwab/functions/search/faiss_dense.index'),
)
bm25_index_path = os.getenv(
    'PWAB_BM25_INDEX_DIR',
    str(repo_root / 'PersonalWAB/envs/pwab/functions/search/indexes'),
)
try:
    global_vector_index = faiss.read_index(vector_index_path)
    global_bm25_searcher = LuceneSearcher(bm25_index_path)
    print("✅ 全局检索索引加载成功！")
except Exception as e:
    global_vector_index = None
    global_bm25_searcher = None
    print(f"⚠️ 检索索引不可用，将跳过全局索引初始化: {e}")

# 4. 全量商品信息映射库
global_all_asins =[]
global_asin_to_info = {}
products_path = os.getenv(
    'PWAB_PRODUCTS_JSONL',
    str(repo_root / 'PersonalWAB/envs/pwab/functions/data/Products/all_products.jsonl'),
)
try:
    with open(products_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            asin = item.get('id')
            if asin:
                global_all_asins.append(asin)
                global_asin_to_info[asin] = item.get('contents', '')
except Exception as e:
    print(f"⚠️ 商品库不可用，将跳过全局商品映射初始化: {e}")
print(f"✅ 全局商品库加载成功，共 {len(global_all_asins)} 条！")

# 5. Recommend 候选池 (Top 500)
global_retrieval_map = {}
retrieval_file_path = os.getenv(
    'PWAB_RETRIEVAL_FILE', str(repo_root / 'SAPA/output_process/baseline_results_top500.jsonl')
)

try:
    with open(retrieval_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line)
            # 你文件里的 key 直接就是 task_id
            t_id = record.get('task_id')
            if t_id:
                global_retrieval_map[t_id] = record
    print(f"✅ 全局推荐候选池加载成功！当前规模: {len(global_retrieval_map)} 条任务。")
except Exception as e:
    print(f"❌ 加载候选池文件失败: {e}")
print("✅ 全局推荐候选池加载成功！\n-----------------------------------\n")

