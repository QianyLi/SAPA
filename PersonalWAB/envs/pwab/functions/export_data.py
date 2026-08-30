import os
import json
from tqdm import tqdm  # 如果没有安装 tqdm，可以 pip install tqdm，或者把下面相关的代码删掉

# 1. 设定路径 (保持和你之前 BM25 代码一致的逻辑)
FOLDER_PATH = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(FOLDER_PATH, 'search', 'indexes')
OUTPUT_DIR = os.path.join(FOLDER_PATH, 'data', 'Products')
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'all_products.jsonl')

def export_index_to_jsonl():
    # 检查索引是否存在
    if not os.path.exists(INDEX_PATH):
        print(f"❌ 错误：找不到索引路径: {INDEX_PATH}")
        return

    # 确保输出目录存在
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"创建目录: {OUTPUT_DIR}")

    print(f"正在加载 BM25 索引: {INDEX_PATH} ...")
    try:
        from pyserini.search.lucene import LuceneSearcher
        searcher = LuceneSearcher(INDEX_PATH)
    except Exception as e:
        print(f"❌ 加载索引失败: {e}")
        return

    # 获取索引中的文档总数
    num_docs = searcher.num_docs
    print(f"索引加载成功！共有 {num_docs} 条文档。")
    print(f"正在导出到: {OUTPUT_FILE} ...")

    # 打开文件准备写入
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        # 遍历所有文档 ID (从 0 到 num_docs-1)
        # 使用 tqdm 显示进度条
        for i in tqdm(range(num_docs), unit="doc"):
            try:
                # Pyserini 获取文档的方法
                doc = searcher.doc(i)
                if doc is None:
                    continue
                
                # 获取原始 JSON 字符串
                raw_json = doc.raw()
                
                # 解析 JSON 确保格式正确
                item = json.loads(raw_json)
                
                # 你的 Dense Retriever 需要 'contents' 字段
                # 如果原始数据里没有 contents，我们需要构建一个
                if 'contents' not in item:
                    # 尝试用 text 或其他字段作为 contents
                    # 这里假设 item 本身就是你要存的内容
                    pass 

                # 写入一行 JSONL
                # ensure_ascii=False 保证中文能正常显示
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

            except Exception as e:
                print(f"导出第 {i} 条数据时出错: {e}")
                continue

    print(f"\n✅ 成功导出！文件已保存至: {OUTPUT_FILE}")

if __name__ == "__main__":
    export_index_to_jsonl()