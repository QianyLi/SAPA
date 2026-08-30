from typing import Any, Dict, List
import json
import sys
import os
# Note: top-level pyserini import removed; the RISE-mode block below imports it
# only when the pyserini backend is actually selected (avoids breaking on hosts
# where the Java/anserini stack is not usable).
# Vanilla-baseline mode
# FOLDER_PATH = os.path.dirname(__file__)
# def load_searcher():
#     print('BM25 searcher loaded.')
#     return LuceneSearcher(os.path.join(FOLDER_PATH, 'search', 'indexes'))

# def search_product_by_query(data: Dict[str, Any], query: str) -> List[Dict[str, Any]]:

#     hits = bm25_searcher.search(query)
#     results = []
#     for i in range(0, len(hits)):
#         docid = hits[i].docid  
#         doc = bm25_searcher.doc(docid) 
#         item = doc.raw()
#         #item = hits[i].raw()
#         item = json.loads(item)
#         p = item['contents']
#         results.append('Product ' + str(i) + ': \n' + p + '\n')

#     if results:
#         return results
#     return []


# search_product_by_query.__info__ = {
#     "type": "function",
#     "function": {
#         "name": "search_product_by_query",
#         "description": "Search for products by a query string. The information of the top 10 products will be returned.",
#         "parameters": {
#             "type": "object",
#             "properties": {
#                 "query": {
#                     "type": "string",
#                     "description": "The query string to search for products, such as 'laptop' or 'phone'.",
#                 },
#             },
#             "required": ["query"],
#         },
#     },
# }
# 
# bm25_searcher = load_searcher()

# from sentence_transformers import SentenceTransformer
# import faiss
# import numpy as np
# import json
# import os
# from typing import Any, Dict, List

# FOLDER_PATH = os.path.dirname(__file__)

# class DenseRetriever:
#     def __init__(self, index_path: str, corpus_path: str, model_name: str = "all-MiniLM-L6-v2"):

#         self.index_path = index_path
#         self.corpus_path = corpus_path
#         self.model = SentenceTransformer(model_name)
#         self.index = self._load_or_build_index()

#     def _load_or_build_index(self):

#         if os.path.exists(self.index_path):
#             print("Loading existing FAISS index...")
#             index = faiss.read_index(self.index_path)
#         else:
#             print("Building FAISS index from corpus...")
#             with open(self.corpus_path, "r", encoding="utf-8") as f:
#                 corpus = [json.loads(line.strip()) for line in f]
#             documents = [doc["contents"] for doc in corpus]

#             embeddings = self.model.encode(documents, convert_to_numpy=True, show_progress_bar=True)
#             dimension = embeddings.shape[1]

#             index = faiss.IndexFlatL2(dimension)
#             index.add(embeddings)

#             faiss.write_index(index, self.index_path)
#         return index

#     def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:

#         query_embedding = self.model.encode([query], convert_to_numpy=True)
#         distances, indices = self.index.search(query_embedding, top_k)

#         # 加载语料库
#         with open(self.corpus_path, "r", encoding="utf-8") as f:
#             corpus = [json.loads(line.strip()) for line in f]

#         results = []
#         for dist, idx in zip(distances[0], indices[0]):
#             if idx != -1:
#                 doc = corpus[idx]
#                 results.append('Product'+ str(idx) + ': \n' + doc["contents"] + '\n',
#                     )
#         return results


# dense_retriever = DenseRetriever(
#     index_path=os.path.join(FOLDER_PATH, "search", "faiss_index"),
#     corpus_path="data/Products/all_products.jsonl",
# )

# def search_product_by_query(data: Dict[str, Any], query: str) -> List[Dict[str, Any]]:
#     results = dense_retriever.search(query, top_k=10)
#     if results:
#         return results
#     return []

# search_product_by_query.__info__ = {
#     "type": "function",
#     "function": {
#         "name": "search_product_by_query",
#         "description": "Search for products by a query string using dense retrieval. The information of the top 10 products will be returned.",
#         "parameters": {
#             "type": "object",
#             "properties": {
#                 "query": {
#                     "type": "string",
#                     "description": "The query string to search for products, such as 'laptop' or 'phone'.",
#                 },
#             },
#             "required": ["query"],
#         },
#     },
# }

# from sentence_transformers import SentenceTransformer
# import faiss
# import numpy as np
# import json
# import os
# import sys
# from typing import Any, Dict, List

# # ================= 配置区域 =================

# # 1. 设置你刚才生成的文件的绝对路径 (直接硬编码，最稳妥)
# CORPUS_PATH = "PersonalWAB/envs/pwab/functions/data/Products/all_products.jsonl"

# # 2. 设置索引保存的位置 (保存在当前脚本同级目录下的 search 文件夹中)
# FOLDER_PATH = os.path.dirname(os.path.abspath(__file__))
# INDEX_DIR = os.path.join(FOLDER_PATH, "search")
# INDEX_PATH = os.path.join(INDEX_DIR, "faiss_dense_bge_m3.index")

# # ================= 核心逻辑 =================

# class DenseRetriever:
#     def __init__(self, index_path: str, corpus_path: str, model_name: str = "BAAI/bge-m3"):
#         self.index_path = index_path
#         self.corpus_path = corpus_path
        
#         print(f"Loading Model: {model_name} ...")
#         self.model = SentenceTransformer(model_name)

#         # 确保索引文件夹存在
#         if not os.path.exists(os.path.dirname(self.index_path)):
#             os.makedirs(os.path.dirname(self.index_path))

#         # 1. 优先加载语料库到内存 (解决 IO 瓶颈)
#         self.documents = self._load_corpus()
        
#         # 2. 加载或构建 FAISS 索引
#         self.index = self._load_or_build_index()

#     def _load_corpus(self) -> List[Dict]:
#         """一次性加载所有 JSONL 数据到内存"""
#         if not os.path.exists(self.corpus_path):
#             raise FileNotFoundError(f"❌ 严重错误：找不到数据文件！\n路径: {self.corpus_path}")
            
#         print(f"Loading corpus from: {self.corpus_path}")
#         documents = []
#         with open(self.corpus_path, "r", encoding="utf-8") as f:
#             for line in f:
#                 try:
#                     documents.append(json.loads(line.strip()))
#                 except json.JSONDecodeError:
#                     continue # 跳过坏行
#         print(f"✅ Loaded {len(documents)} documents.")
#         return documents

#     def _load_or_build_index(self):
#         """加载现有索引，或者重新计算 Embeddings 并构建索引"""
#         if os.path.exists(self.index_path):
#             print(f"Loading existing FAISS index from {self.index_path}...")
#             index = faiss.read_index(self.index_path)
#         else:
#             print("⚠️ Index not found. Building new FAISS index (this takes time)...")
            
#             # 提取文本内容
#             # 假设你的 jsonl 里核心文本字段是 'contents'，如果不是，请修改这里
#             doc_texts = []
#             for doc in self.documents:
#                 # 兼容性处理：优先取 contents，没有则取 text，再没有转 string
#                 content = doc.get("contents", doc.get("text", str(doc)))
#                 doc_texts.append(content)

#             # 编码 (normalize_embeddings=True 使得内积等价于余弦相似度)
#             embeddings = self.model.encode(doc_texts, convert_to_numpy=True, show_progress_bar=True, normalize_embeddings=True)
            
#             dimension = embeddings.shape[1]
            
#             # 使用 Inner Product (IP) 索引
#             index = faiss.IndexFlatIP(dimension)
#             index.add(embeddings)

#             print(f"Saving index to {self.index_path}...")
#             faiss.write_index(index, self.index_path)
            
#         return index

#     def search(self, query: str, top_k: int = 10) -> List[str]:
#         # 编码查询并搜索
#         query_embedding = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
#         distances, indices = self.index.search(query_embedding, top_k)

#         results = []
#         for dist, idx in zip(distances[0], indices[0]):
#             if idx != -1 and idx < len(self.documents):
#                 doc = self.documents[idx]
#                 content = doc.get("contents", str(doc)) # 获取内容
                
#                 # 格式化输出
#                 results.append(f"Product {idx}: \n{content}\n")
        
#         return results

# # ================= 初始化全局对象 =================

# print("Initializing Dense Retriever system...")
# try:
#     dense_retriever = DenseRetriever(
#         index_path=INDEX_PATH,
#         corpus_path=CORPUS_PATH
#     )
#     print("✅ System ready.")
# except Exception as e:
#     print(f"❌ Initialization Failed: {e}")
#     dense_retriever = None

# # ================= 接口函数 =================

# def search_product_by_query(data: Dict[str, Any], query: str) -> List[str]:
#     """
#     Search for products by a query string using dense retrieval.
#     """
#     if dense_retriever is None:
#         return ["Error: Search system is not initialized."]
        
#     results = dense_retriever.search(query, top_k=10)
#     if results:
#         return results
#     return []

# search_product_by_query.__info__ = {
#     "type": "function",
#     "function": {
#         "name": "search_product_by_query",
#         "description": "Search for products by a query string using dense retrieval.",
#         "parameters": {
#             "type": "object",
#             "properties": {
#                 "query": {
#                     "type": "string",
#                     "description": "The query string to search for products, e.g., 'laptop'.",
#                 },
#             },
#             "required": ["query"],
#         },
#     },
# }

# # ================= 本地测试 (可选) =================
# if __name__ == "__main__":
#     # 这段代码只有直接运行此文件时才会执行
#     print("\n--- Testing Search ---")
#     test_query = "laptop"
#     print(f"Searching for: {test_query}")
#     res = search_product_by_query({}, test_query)
#     for item in res[:3]: # 只打印前3个结果
#         print(item)

# from typing import Any, Dict, List, Union
# from pyserini.search.lucene import LuceneSearcher
# import json
# import sys
# import os
# from collections import defaultdict

# # ================= 初始化区域 =================
# FOLDER_PATH = os.path.dirname(__file__)

# def load_searcher():
#     # 确保索引路径正确
#     index_path = os.path.join(FOLDER_PATH, 'search', 'indexes')
#     print(f'Loading BM25 searcher from: {index_path}')
#     return LuceneSearcher(index_path)

# # 全局加载 searcher
# bm25_searcher = load_searcher()

# # ================= 核心函数修改 =================
# def search_product_by_query(data: Dict[str, Any], query: Union[str, List[str]]) -> List[str]:
#     """
#     支持多路召回的搜索函数。
#     参数 query: 可以是一个字符串，也可以是包含多个生成的 query 的列表 (List[str])。
#     """
    
#     # 1. 统一输入格式：如果是单条字符串，转为列表
#     if isinstance(query, str):
#         queries = [query]
#     else:
#         queries = query  # 已经是列表了

#     # 2. RRF (Reciprocal Rank Fusion) 初始化
#     # doc_scores 存储 {docid: rrf_score}
#     doc_scores = defaultdict(float)
#     k_constant = 60  # RRF 经典常数
    
#     # 3. 遍历每一条 query 进行搜索
#     # 既然有10条query，我们每条取 Top 20，这样召回范围足够广，又不会太慢
#     hits_per_query = 20 
    
#     for single_query in queries:
#         # 跳过空字符串
#         if not single_query or not single_query.strip():
#             continue
            
#         try:
#             # 执行 BM25 搜索
#             hits = bm25_searcher.search(single_query, k=hits_per_query)
            
#             # 累加 RRF 分数
#             for rank, hit in enumerate(hits):
#                 # 公式: 1 / (k + rank)，rank 从 0 开始
#                 doc_scores[hit.docid] += 1.0 / (k_constant + rank)
                
#         except Exception as e:
#             # 防止某条生成的 query包含非法字符导致报错，跳过即可
#             # print(f"Query error: {e}") 
#             continue

#     # 4. 排序：根据 RRF 总分从高到低排序
#     # items() 返回 (docid, score) 元组
#     sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    
#     # 5. 格式化输出 Top 10
#     results = []
#     top_k_final = 10
    
#     # 截取前 10 个
#     final_candidates = sorted_docs[:top_k_final]
    
#     for i, (docid, score) in enumerate(final_candidates):
#         try:
#             # 反查文档详细内容
#             doc = bm25_searcher.doc(docid)
#             if doc:
#                 item = json.loads(doc.raw())
#                 # 兼容不同数据格式，优先取 contents
#                 p = item.get('contents', item.get('text', str(item)))
                
#                 # 保持和你原来一样的返回格式，以便后续 eval 代码能跑通
#                 results.append('Product ' + str(i) + ': \n' + p + '\n')
#         except Exception as e:
#             print(f"Error extracting doc {docid}: {e}")
#             continue

#     if results:
#         return results
#     return []

# # ================= 接口描述 =================
# search_product_by_query.__info__ = {
#     "type": "function",
#     "function": {
#         "name": "search_product_by_query",
#         "description": "Search for products. Accepts a single query string or a list of query strings (for multi-query fusion).",
#         "parameters": {
#             "type": "object",
#             "properties": {
#                 "query": {
#                     "type": "string", # JSON Schema 没法直接定义 list[str]，但在 python 内部我们已经处理了
#                     "description": "The query string(s) to search for products.",
#                 },
#             },
#             "required": ["query"],
#         },
#     },
# }

# RISE模式
from typing import Any, Dict, List, Union
from sentence_transformers import CrossEncoder
import json
import os
import pickle
import re
import torch

# ================= 1. 模型路径配置 =================
FOLDER_PATH = os.path.dirname(__file__)
INDEX_PATH = os.path.join(FOLDER_PATH, 'search', 'indexes')

# Override with a local cache path when needed.
RERANKER_PATH = os.getenv("PWAB_RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")

# ================= 2. Backend 选择 =================
# 默认走 pyserini/Lucene (PersonalWAB 原索引)。
# 若环境变量 PWAB_BM25_INDEX 指向一个 rank_bm25 pickle 文件，则切到纯 Python BM25,
# 用 ablation/build_bm25_python_index.py 为新数据集 (Amazon Office/Beauty/Electronics) 建的索引。
_BM25_PICKLE = os.environ.get("PWAB_BM25_INDEX", "").strip() or None

bm25_searcher = None       # pyserini LuceneSearcher (Backend A)
_PY_BM25 = None            # dict from rank_bm25 pickle (Backend B)
_PY_BM25_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _py_tokenize(text):
    return [t.lower() for t in _PY_BM25_TOKEN_RE.findall(text or "")]


if _BM25_PICKLE:
    print(f"[search] loading Python BM25 index from: {_BM25_PICKLE}")
    with open(_BM25_PICKLE, "rb") as _f:
        _PY_BM25 = pickle.load(_f)
    print(f"[search] Python BM25 ready | docs={len(_PY_BM25['asins'])}")
else:
    from pyserini.search.lucene import LuceneSearcher
    print(f'Loading BM25 searcher from: {INDEX_PATH}')
    bm25_searcher = LuceneSearcher(INDEX_PATH)

print(f'Loading Reranker from: {RERANKER_PATH} ...')
try:
    # 自动判断显卡
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # 加载 CrossEncoder
    reranker = CrossEncoder(RERANKER_PATH, device=device, max_length=512)
    print(f"✅ Reranker loaded on {device}")
except Exception as e:
    print(f"❌ Reranker load failed: {e}")
    reranker = None

# record the exact items after removing the repeat

# STATS_LOG_PATH = "SAPA/data/best/multi_query_recall_stats.jsonl"

def search_product_by_query(
    data: Dict[str, Any], 
    query: Union[str, List[str]], 
    target_asin: str = ""   # 接收目标商品 ID
) -> List[str]:
    
    # --- A. 获取原始指令 ---
    original_instruction = data.get('instruction', "")
    sample_id = original_instruction

    # --- B. BM25 多路召回 (恢复你的原始逻辑) ---
    if isinstance(query, str):
        queries = [query]
    else:
        queries = query # 正常接收那 10 条生成的 Query

    candidate_map = {}

    if _PY_BM25 is not None:
        # ---- Backend B: pure-Python BM25 (per-dataset index) ----
        import numpy as np
        asins = _PY_BM25["asins"]
        contents_all = _PY_BM25["contents"]
        bm25 = _PY_BM25["bm25"]
        for q in queries:
            if not q or not isinstance(q, str) or not q.strip():
                continue
            try:
                toks = _py_tokenize(q)
                if not toks:
                    continue
                scores = bm25.get_scores(toks)
                # top-20 to mirror Lucene k=20
                top_idx = np.argpartition(-scores, min(20, len(scores) - 1))[:20]
                top_idx = top_idx[np.argsort(-scores[top_idx])]
                for idx in top_idx:
                    docid = asins[idx]
                    if docid not in candidate_map:
                        candidate_map[docid] = contents_all[idx]
            except Exception as e_search:
                print(f"❌ Python BM25 error on query '{q}': {e_search}")
                continue
    else:
        # ---- Backend A: original pyserini/Lucene path (PersonalWAB) ----
        for q in queries:
            if not q or not isinstance(q, str) or not q.strip():
                continue

            try:
                # 【关键区别】：这里是你的原始逻辑，每条 Query 固定搜 20 条！
                hits = bm25_searcher.search(q, k=20)

                for hit in hits:
                    if hit.docid not in candidate_map:
                        try:
                            doc_object = bm25_searcher.doc(hit.docid)
                            if doc_object is not None:
                                raw_json = doc_object.raw()
                                item = json.loads(raw_json)
                                content = item.get('contents', item.get('text', str(item)))
                                # candidate_map 存储: {docid: 商品内容文本}
                                candidate_map[hit.docid] = content
                        except Exception as e_parse:
                            print(f"❌ Error fetching doc {hit.docid}: {e_parse}")
                            continue
            except Exception as e_search:
                print(f"❌ Search Error on query '{q}': {e_search}")
                continue
    
    # 【新增逻辑 1】：分离 ID 和 内容，并统计实际产生的 N
    docids_list = list(candidate_map.keys())
    contents_list = list(candidate_map.values())
    
    # 这就是你多路查询真实产生的候选池大小 N (比如 30 左右)
    actual_N = len(candidate_map) 

    # 【核心统计 1：重排前的 Recall@N】
    # 真值是否被这 10 条 Query 的合力捞进了池子里？
    hit_before_rerank = 1 if target_asin and target_asin in docids_list else 0

    # 如果没搜到任何结果
    if not contents_list:
        recall_log = os.getenv("PWAB_RECALL_LOG", "SAPA/data/best/recall_exp4_multi.jsonl")
        os.makedirs(os.path.dirname(recall_log) or ".", exist_ok=True)
        with open(recall_log, "a", encoding="utf-8") as f:
            log_data = {
                "instruction": sample_id,
                "target_asin": target_asin,
                "pool_size_N": 0,
                "hit_before_rerank": 0,
                "hit_after_rerank": 0
            }
            f.write(json.dumps(log_data, ensure_ascii=False) + "\n")
        return[]

    # --- C. BGE 重排序 ---
    if reranker and original_instruction:
        try:
            pairs = [[original_instruction, doc] for doc in contents_list]
            scores = reranker.predict(pairs, batch_size=32, show_progress_bar=False)
            
            # 【关键修改】：将 docid 加入排序，追踪真值去向
            ranked_results = sorted(zip(docids_list, contents_list, scores), key=lambda x: x[2], reverse=True)
            
            final_top10_ids =[docid for docid, content, score in ranked_results[:10]]
            final_top10_contents = [content for docid, content, score in ranked_results[:10]]
            
        except Exception as e_rank:
            print(f"❌ Rerank Error: {e_rank}")
            final_top10_ids = docids_list[:10]
            final_top10_contents = contents_list[:10]
    else:
        final_top10_ids = docids_list[:10]
        final_top10_contents = contents_list[:10]

    # 【核心统计 2：重排后的 Recall@10】
    # BGE 是否成功把真值排进了前 10 名？
    hit_after_rerank = 1 if target_asin and target_asin in final_top10_ids else 0

    # 【保存统计结果】：强烈建议主方法存为 recall_exp4_multi.jsonl
    with open("recall_exp4_multi.jsonl", "a", encoding="utf-8") as f:
        log_data = {
            "instruction": sample_id,
            "target_asin": target_asin,
            "pool_size_N": actual_N,  # 记录下真实的 N
            "hit_before_rerank": hit_before_rerank, 
            "hit_after_rerank": hit_after_rerank    
        }
        f.write(json.dumps(log_data, ensure_ascii=False) + "\n")

    # --- D. 格式化输出 ---
    # NOTE: emit ASIN inside the rendered string so downstream `target_asin in res[i]`
    # substring matching works regardless of whether the underlying index stored ASIN
    # in its `contents` field (PersonalWAB's pyserini index does; our per-dataset
    # rank_bm25 index does not, since we keep contents focused on textual signal).
    results = []
    for i, (asin, content) in enumerate(zip(final_top10_ids, final_top10_contents)):
        results.append('Product ' + str(i) + ' [ASIN: ' + str(asin) + ']: \n' + content + '\n')

    return results

# dynamic search

# pool_size_map = {}
# with open(STATS_LOG_PATH, "r") as f:
#     for line in f:
#         item = json.loads(line)
#         pool_size_map[item['id']] = item['final_candidates_count_N']

# def search_product_by_query(data: Dict[str, Any], query: Union[str, List[str]]) -> List[str]:
#     # --- A. 获取原始指令 ---
#     original_instruction = data.get('instruction', "")
    
#     # 【重要修复】：这里应该用 data 里的 id，而不是 instruction
#     # 因为你在保存 stats 时存的是 data.get('id')
#     sample_id = original_instruction
    
#     # 直接使用全局的 pool_size_map，不需要传参
#     N = pool_size_map.get(sample_id, 100)

#     # --- B. BM25 单路召回 ---
#     if isinstance(query, list):
#         queries = [query[0]] 
#     else:
#         queries = [query] 

#     candidate_map = {} 

#     for q in queries:
#         if not q or not isinstance(q, str) or not q.strip(): 
#             continue
            
#         try:
#             # 使用从全局变量里拿到的 N
#             hits = bm25_searcher.search(q, k=N)
            
#             for hit in hits:
#                 if hit.docid not in candidate_map:
#                     try:
#                         doc_object = bm25_searcher.doc(hit.docid)
#                         if doc_object is not None:
#                             raw_json = doc_object.raw() 
#                             item = json.loads(raw_json)
#                             content = item.get('contents', item.get('text', str(item)))
#                             candidate_map[hit.docid] = content
#                     except Exception as e_parse:
#                         print(f"❌ Error fetching doc {hit.docid}: {e_parse}")
#                         continue
#         except Exception as e_search:
#             print(f"❌ Search Error on query '{q}': {e_search}")
#             continue
    
#     candidates_list = list(candidate_map.values())
    
#     if not candidates_list:
#         return []

#     # --- C. BGE 重排序 ---
#     if reranker and original_instruction:
#         try:
#             pairs = [[original_instruction, doc] for doc in candidates_list]
#             scores = reranker.predict(pairs, batch_size=32, show_progress_bar=False)
#             ranked_results = sorted(zip(candidates_list, scores), key=lambda x: x[1], reverse=True)
#             final_top10 = [doc for doc, score in ranked_results[:10]]
#         except Exception as e_rank:
#             print(f"❌ Rerank Error: {e_rank}")
#             final_top10 = candidates_list[:10]
#     else:
#         final_top10 = candidates_list[:10]

#     # --- D. 格式化输出 ---
#     results = []
#     for i, p in enumerate(final_top10):
#         results.append('Product ' + str(i) + ': \n' + p + '\n')

#     return results


# ================= 3. 核心搜索函数_my improve =================

# def search_product_by_query(data: Dict[str, Any], query: Union[str, List[str]]) -> List[str]:
#     # --- A. 获取原始指令 ---
#     original_instruction = data.get('instruction', "")
    
#     # --- B. BM25 多路召回 ---
#     if isinstance(query, str):
#         queries = [query]
#     else:
#         queries = query

#     candidate_map = {}

#     for q in queries:
#         if not q or not isinstance(q, str) or not q.strip(): 
#             continue
            
#         try:
#             # 1. 搜索拿到 hits (只包含 ID 和分数)
#             hits = bm25_searcher.search(q, k=20)
            
#             for hit in hits:
#                 if hit.docid not in candidate_map:
#                     try:
#                         # 2. 关键修复：通过 docid 去查完整的文档内容 ！！！
#                         # 错误写法: raw = hit.raw()
#                         # 正确写法: doc = bm25_searcher.doc(hit.docid)
                        
#                         doc_object = bm25_searcher.doc(hit.docid)
                        
#                         if doc_object is not None:
#                             raw_json = doc_object.raw() # 获取原始 JSON 字符串
#                             item = json.loads(raw_json)
                            
#                             # 提取内容 (优先取 contents)
#                             content = item.get('contents', item.get('text', str(item)))
#                             candidate_map[hit.docid] = content
                            
#                     except Exception as e_parse:
#                         # 打印一下，万一还有解析错误
#                         print(f"❌ Error fetching doc {hit.docid}: {e_parse}")
#                         continue
#         except Exception as e_search:
#             print(f"❌ Search Error on query '{q}': {e_search}")
#             continue
    
#     candidates_list = list(candidate_map.values())
    
#     # 如果没搜到东西，返回空
#     if not candidates_list:
#         return []

#     # --- C. BGE 重排序 ---
#     if reranker and original_instruction:
#         try:
#             # 构造输入对
#             pairs = [[original_instruction, doc] for doc in candidates_list]
            
#             # 打分
#             scores = reranker.predict(pairs, batch_size=32, show_progress_bar=False)
            
#             # 排序
#             ranked_results = sorted(zip(candidates_list, scores), key=lambda x: x[1], reverse=True)
            
#             # 取 Top 10
#             final_top10 = [doc for doc, score in ranked_results[:10]]
#         except Exception as e_rank:
#             print(f"❌ Rerank Error: {e_rank}")
#             final_top10 = candidates_list[:10]
#     else:
#         final_top10 = candidates_list[:10]

#     # --- D. 格式化输出 ---
#     results = []
#     for i, p in enumerate(final_top10):
#         results.append('Product ' + str(i) + ': \n' + p + '\n')

#     return results

#multi
# def search_product_by_query(data: Dict[str, Any], query: Union[str, List[str]]) -> List[str]:
#     """
#     修改后的搜索工具：
#     1. 🚀 RISE 模式：如果 query 是一个列表 (Agent 已经找好了 Top 10 ASINs)。
#     2. 🏛️ 传统模式：如果 query 是一个字符串 (Baseline 还在传关键词)。
#     """
    
#     # ---------------------------------------------------------
#     # 🚀 RISE 模式：Agent 直接提交了它选出的 Top 10 ID (ASINs/DocIDs)
#     # ---------------------------------------------------------
#     if isinstance(query, list):
#         results = []
#         # 我们直接按照 Agent 给出的顺序，去图书馆（Lucene）里把书取出来
#         for i, docid in enumerate(query[:10]):
#             doc_obj = bm25_searcher.doc(str(docid)) # 根据 ID 取文档
#             if doc_obj:
#                 item = json.loads(doc_obj.raw())
#                 p = item.get('contents', str(item)) # 拿到商品描述文本
#                 # 保持原有的格式化输出，确保阅卷老师能看懂
#                 results.append(f'Product {i}: \n{p}\n')
#             else:
#                 # 容错：万一 ID 在库里没找到
#                 results.append(f'Product {i}: \n[ID: {docid}] Details not found.\n')
        
#         return results if results else ["No results found."]

#     # ---------------------------------------------------------
#     # 🏛️ 传统模式：原本的关键词搜索逻辑（保留兼容性）
#     # ---------------------------------------------------------
#     # 这里的 query 是字符串，如 "Nike black running shoes"
#     hits = bm25_searcher.search(query, k=10) # 原始逻辑通常取 10 个
#     results = []
#     for i in range(len(hits)):
#         docid = hits[i].docid
#         doc = bm25_searcher.doc(docid)
#         item = json.loads(doc.raw())
#         p = item.get('contents', "")
#         results.append('Product ' + str(i) + ': \n' + p + '\n')

#     return results if results else ["No results found."]

# def search_product_by_query_rrf(data: Dict[str, Any], query: Union[str, List[str]], k_rrf: int = 60) -> List[str]:
#     # --- A. 获取原始指令 ---
#     original_instruction = data.get('instruction', "")
    
#     # --- B. 准备查询 ---
#     if isinstance(query, str):
#         queries = [query]
#     else:
#         queries = query 

#     candidate_map = {}      # 存储 docid -> content
#     rrf_score_map = {}      # 存储 docid -> RRF 累计分数

#     # --- C. 执行多路召回并计算 RRF 分数 ---
#     for q in queries:
#         if not q or not isinstance(q, str) or not q.strip(): 
#             continue
            
#         try:
#             # 这里的 k 可以设大一点，比如每路召回 20-50 个
#             hits = bm25_searcher.search(q, k=20)
            
#             # rank 从 1 开始计算
#             for rank, hit in enumerate(hits, start=1):
#                 docid = hit.docid
                
#                 # 1. 计算 RRF 分数公式: 1 / (k + rank)
#                 # k 是平滑常数，默认通常取 60
#                 score = 1.0 / (k_rrf + rank)
                
#                 # 2. 累加分数（核心：如果商品在多路查询中重复出现，分数会叠加）
#                 if docid in rrf_score_map:
#                     rrf_score_map[docid] += score
#                 else:
#                     rrf_score_map[docid] = score
                    
#                     # 3. 只有第一次见到该 docid 时才获取内容（优化性能）
#                     try:
#                         doc_object = bm25_searcher.doc(docid)
#                         if doc_object:
#                             raw_json = doc_object.raw()
#                             item = json.loads(raw_json)
#                             content = item.get('contents', item.get('text', str(item)))
#                             candidate_map[docid] = content
#                     except Exception as e_parse:
#                         print(f"❌ Error fetching doc {docid}: {e_parse}")
#                         continue
#         except Exception as e_search:
#             print(f"❌ Search Error on query '{q}': {e_search}")
#             continue

#     # --- D. 根据 RRF 分数排序 ---
#     # rrf_score_map 里的 key 是 docid，value 是累计的 RRF 分数
#     # 我们按分数从高到低排列
#     sorted_docids = sorted(rrf_score_map.items(), key=lambda x: x[1], reverse=True)
    
#     # 取出前 10 个 docid 对应的内容
#     final_top10_docs = []
#     for docid, score in sorted_docids[:10]:
#         if docid in candidate_map:
#             final_top10_docs.append(candidate_map[docid])

#     # --- E. 格式化输出 (供消融实验对比) ---
#     results = []
#     for i, p in enumerate(final_top10_docs):
#         results.append(f'Product {i}: \n{p}\n')

#     return results

# 保持接口定义
search_product_by_query.__info__ = {
    "type": "function",
    "function": {
        "name": "search_product_by_query",
        "description": "Multi-query Recall + BGE Reranking",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"}
            },
            "required": ["query"],
        },
    },
}
