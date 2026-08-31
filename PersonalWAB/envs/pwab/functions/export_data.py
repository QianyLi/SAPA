import os
import json
from tqdm import tqdm

FOLDER_PATH = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(FOLDER_PATH, 'search', 'indexes')
OUTPUT_DIR = os.path.join(FOLDER_PATH, 'data', 'Products')
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'all_products.jsonl')

def export_index_to_jsonl():
    if not os.path.exists(INDEX_PATH):
        print(f"❌ 错误：找不到索引路径: {INDEX_PATH}")
        return

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

    num_docs = searcher.num_docs
    print(f"索引加载成功！共有 {num_docs} 条文档。")
    print(f"正在导出到: {OUTPUT_FILE} ...")

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for i in tqdm(range(num_docs), unit="doc"):
            try:
                doc = searcher.doc(i)
                if doc is None:
                    continue

                raw_json = doc.raw()

                item = json.loads(raw_json)

                if 'contents' not in item:
                    pass

                f.write(json.dumps(item, ensure_ascii=False) + '\n')

            except Exception as e:
                print(f"导出第 {i} 条数据时出错: {e}")
                continue

    print(f"\n✅ 成功导出！文件已保存至: {OUTPUT_FILE}")

if __name__ == "__main__":
    export_index_to_jsonl()
