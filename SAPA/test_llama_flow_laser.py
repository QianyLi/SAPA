import json
import torch
import faiss
import numpy as np

import argparse
import re
from tqdm import tqdm
from FlagEmbedding import BGEM3FlagModel

def parse_args():
    parser = argparse.ArgumentParser(description="LASER Expansion using FAISS and BGE-M3")
    parser.add_argument('--input_file', type=str, required=True, help='模型原始输出(1个ASIN的JSON)')
    parser.add_argument('--index_file', type=str, required=True, help='faiss_dense_bge_m3.index 路径')
    parser.add_argument('--all_products_jsonl', type=str, required=True, help='全量商品库，用于查找种子文本')
    parser.add_argument('--output_file', type=str, default='faiss_expanded_res.json')
    parser.add_argument('--model_path', type=str, default='BAAI/bge-m3') # 请指向你的BGE-M3路径
    return parser.parse_args()

class FaissLaserExpander:
    def __init__(self, index_path, all_products_path, model_path):
        print("Loading FAISS index...")
        self.index = faiss.read_index(index_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print("Loading BGE-M3 model...")
        self.model = BGEM3FlagModel(model_path, use_fp16=True)
        
        # 加载全库文本和 ID 映射
        self.asin_to_contents = {}
        self.id_to_asin = [] # FAISS 索引 i 对应的 ASIN
        
        print("Loading product metadata and building ID mapping...")
        with open(all_products_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                data = json.loads(line)
                asin = data['id']
                self.asin_to_contents[asin] = data['contents']
                self.id_to_asin.append(asin)
        
        # 简单检查索引和映射是否匹配
        if self.index.ntotal != len(self.id_to_asin):
            print(f"Warning: Index size ({self.index.ntotal}) != Mapping size ({len(self.id_to_asin)})")

    def expand(self, seed_asin, top_k=10):
        if seed_asin not in self.asin_to_contents:
            return [seed_asin]
        
        # 1. 获取种子文本并编码
        seed_text = self.asin_to_contents[seed_asin]
        # BGE-M3 encode 返回的是 list of numpy
        query_vector = self.model.encode([seed_text], batch_size=1, max_length=512)['dense_vecs']
        query_vector = np.array(query_vector).astype('float32')
        
        # 2. FAISS 搜索
        # D 是距离/相似度, I 是对应的行索引
        D, I = self.index.search(query_vector, top_k)
        
        # 3. 转换回 ASIN
        res = []
        for idx in I[0]:
            if idx != -1: # 过滤无效索引
                res.append(self.id_to_asin[idx])
        
        return res

def main():
    args = parse_args()

    # 初始化加载器
    expander = FaissLaserExpander(args.index_file, args.all_products_jsonl, args.model_path)

    print(f"Loading task results from {args.input_file}")
    with open(args.input_file, 'r') as f:
        tasks = json.load(f)

    expanded_results = {}
    rec_count = 0

    print("Expanding Recommend seeds...")
    for instruction, output_data in tqdm(tasks.items()):
        if not output_data or not isinstance(output_data, list):
            expanded_results[instruction] = output_data
            continue

        first_content = str(output_data[0]).strip()
        
        # 判断是否为 Recommend 任务的 ASIN 格式
        if re.match(r'^B[0-9A-Z]{9}', first_content):
            rec_count += 1
            seed = first_content.split(',')[0].strip()
            
            # 使用 FAISS 扩展
            top_10 = expander.expand(seed, top_k=10)
            expanded_results[instruction] = top_10
        else:
            # Search/Review 任务保持原样
            expanded_results[instruction] = output_data

    with open(args.output_file, 'w') as f:
        json.dump(expanded_results, f, indent=2, ensure_ascii=False)

    print(f"Successfully expanded {rec_count} recommendations. Saved to {args.output_file}")

if __name__ == "__main__":
    main()
