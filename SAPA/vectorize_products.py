import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
import time

# --- 脚本配置 ---

# 输入的映射文件名 (由上一个脚本生成)
input_map_filename = 'data/asin_document_map.json'

# 输出文件的基本名称
output_vector_filename = 'data/product_vectors.npy'
output_asin_list_filename = 'data/product_asins.json'

# =======================
# === 关键修改之处在这里 ===
# =======================
# 直接将模型名称变量指向您的本地文件夹路径
# 确保这个路径是包含模型文件（如 pytorch_model.bin, config.json 等）的文件夹
local_model_path = os.getenv('PWAB_SIM_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')

# (可选) 检查本地路径是否存在，给出更友好的提示
if not os.path.isdir(local_model_path):
    print(f"错误：指定的本地模型路径 '{local_model_path}' 不存在或不是一个文件夹。")
    print("请检查您的路径是否正确。")
    exit()

# 设置编码时使用的设备。如果有GPU，会自动使用 'cuda'。
device = 'cuda' # or 'cpu'

# --- 脚本主程序 ---

print("--- 开始商品向量化流程 ---")

# 1. 检查输入文件是否存在
if not os.path.exists(input_map_filename):
    print(f"错误：输入文件 '{input_map_filename}' 不存在。")
    print("请先运行 'create_mapping.py' 脚本来生成此文件。")
    exit()

# 2. 从文件加载 asin -> document 的映射
print(f"正在从 '{input_map_filename}' 加载商品文档...")
with open(input_map_filename, 'r', encoding='utf-8') as f:
    asin_to_document_map = json.load(f)

if not asin_to_document_map:
    print("错误：映射文件为空，无法进行向量化。")
    exit()

# 3. 准备要编码的 asin 列表和文档列表
asins = list(asin_to_document_map.keys())
documents = list(asin_to_document_map.values())
total_products = len(asins)
print(f"共找到 {total_products} 个商品待处理。")


# 4. 从本地路径加载 Sentence Transformer 模型
# ==================================================
print(f"正在从本地路径加载 Sentence Transformer 模型: '{local_model_path}' ...")
try:
    # 直接将本地路径传入 SentenceTransformer
    model = SentenceTransformer(local_model_path, device=device)
    print(f"模型已成功从本地加载到设备: {model.device}")
except Exception as e:
    print(f"从本地路径加载模型时出错: {e}")
    print("请确保该路径下的模型文件是完整且未损坏的。")
    exit()
# ==================================================


# 5. 对所有文档进行编码
print("\n开始对所有商品文档进行编码...")
print(f"这可能需要一些时间，具体取决于商品数量和您的硬件（GPU会快很多）。")
start_time = time.time()

product_vectors = model.encode(
    documents,
    show_progress_bar=True,
    batch_size=32
)

end_time = time.time()
processing_time = end_time - start_time
print(f"\n编码完成！总耗时: {processing_time:.2f} 秒。")
print(f"平均每个商品的编码时间: {(processing_time / total_products) * 1000:.2f} 毫秒。")


# 6. 验证输出的形状
print(f"\n生成的向量矩阵形状: {product_vectors.shape}")
print(f"  - 行数 (商品数量): {product_vectors.shape[0]}")
print(f"  - 列数 (向量维度): {product_vectors.shape[1]}")


# 7. 保存结果到文件
print("\n正在保存 asin 列表和向量矩阵...")
np.save(output_vector_filename, product_vectors)
print(f"  - 向量矩阵已保存到: '{output_vector_filename}'")

with open(output_asin_list_filename, 'w', encoding='utf-8') as f:
    json.dump(asins, f)
print(f"  - 对应的 asin 列表已保存到: '{output_asin_list_filename}'")


print("\n--- 商品向量化流程全部完成！ ---")
print("您现在可以使用这两个输出文件进行 Faiss 聚类了。")
