import torch
import numpy as np
from typing import Any, Dict, List
import argparse
import json
import os
FOLDER_PATH = os.path.dirname(__file__)

model = None  
# we trained the recommender model with 5000 users to maximize the non-cold-start items in our test set.
# There might be other better solutions
maxlen = 30
usernum = 19005
itemnum = 104284
product_id_map = json.load(open(os.path.join(FOLDER_PATH,'recommender/data/1000_ids.json'), 'r'))
device = 'cuda' if torch.cuda.is_available() else 'cpu'  

Product_Prompt = '''
Product <Num>:
- Title: <TITLE>
- Parent Asin: <PARENT_ASIN>
- Average Rating: <AVERAGE_RATING>
- Rating Number: <RATING_NUMBER>
- Price: <PRICE>
- Store: <STORE>
- Features: <FEATURES>
- Description: <DESCRIPTION>
- Details: <DETAILS>
- Category: <CATEGORY>
'''

def pretty_product(product: dict, num) -> str:
    # print(product)
    res = Product_Prompt.replace('<CATEGORY>', str(product['main_category']))
    res = res.replace('<Num>', str(num))
    res = res.replace('<TITLE>', product['title'])
    res = res.replace('<AVERAGE_RATING>', str(product['average_rating']))
    res = res.replace('<RATING_NUMBER>', str(product['rating_number']))
    res = res.replace('<FEATURES>', str(product['features']))
    res = res.replace('<DESCRIPTION>', str(product['description']))
    res = res.replace('<PRICE>', str(product['price']))
    res = res.replace('<STORE>', str(product['store']))
    res = res.replace('<DETAILS>', json.dumps(product['details']))
    res = res.replace('<PARENT_ASIN>', product['parent_asin'])
    return res


def load_sasrec_model(model_path: str, usernum: int, itemnum_: int):

    global model, itemnum
    from PersonalWAB.envs.pwab.functions.recommender.model import SASRec
    itemnum = itemnum_
    rec_args = argparse.Namespace(
        maxlen=maxlen,
        hidden_units=50,
        num_blocks=2,
        num_heads=2,
        dropout_rate=0.5,
        device=device
    )
    model = SASRec(usernum, itemnum, rec_args) 
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    model.to(device)
    model.eval()  
    print('SASRec recommender loaded.')

def predict_next_item(model, item_sequence, maxlen, itemnum, k, device='cuda', positive_indices=None):
    seq = np.zeros([maxlen], dtype=np.int32)
    seq[-len(item_sequence):] = item_sequence
    item_indices = np.arange(1, itemnum + 1)

    with torch.no_grad():
        user_ids = np.array([0])
        logits = model.predict(*[np.array(l) for l in [[user_ids], [seq], item_indices]])
        logits = -logits.cpu().numpy().flatten()

        if positive_indices is not None:
            result = np.full(logits.shape, np.inf)
            positive_indices = [index - 1 for index in positive_indices]
            result[positive_indices] = logits[positive_indices]

        top_k_items = np.argsort(result)[:k] + 1
    return top_k_items

# def get_recommendations_by_history(data: Dict[str, Any], product_sequence: List[str]):
#     global model, itemnum, product_id_map
#     recommendations = []
    
#     item_sequence = [product_id_map.get(product_asin, -1) for product_asin in product_sequence if product_asin in product_id_map]
#     if len(item_sequence) == 0:
#         return recommendations  
#     # if len(item_sequence) < 4:
#     #     item_sequence = [item_sequence[0]] * 3 + item_sequence

#     item_sequence = item_sequence[-maxlen:]  
    
#     positive_indices = list(product_id_map.values()) 
#     recommended_items = predict_next_item(model, item_sequence, maxlen, itemnum, 10, device=device, positive_indices=positive_indices)
    
#     asin_reverse_map = {v: k for k, v in product_id_map.items()} 
#     for item_id in recommended_items:
#         asin = asin_reverse_map.get(item_id)
#         recommendations.append(asin)
    
#     recommendations = [pretty_product(data['all_products'][asin], i) for i, asin in enumerate(recommendations)]

#     return recommendations  
def get_recommendations_by_history(data: Dict[str, Any], product_sequence: List[str]):
    global model, itemnum, product_id_map
    recommendations = []
    
    # ---------------------------------------------------------
    # 🚀 RISE 模式：判断是否已经是最终答案
    # 如果 Agent 传进来的已经是 10 个 ASIN 结果，且这些 ASIN 在库里，
    # 我们就认为 Agent 已经做好了决策，直接跳过 SASRec 的预测逻辑。
    # ---------------------------------------------------------
    # 逻辑：如果长度正好是 10，且我们开启了“直接推荐”模式
    # （或者简单判断：如果 product_sequence 里的 ASIN 看起来不像历史，而是结果）
    
    # 为了保险，我们检查 data 字典里是否有我们自定义的标记，或者根据长度判断
    if len(product_sequence) == 10:
        # 这里的逻辑是：如果 Agent 传了 10 个，我们直接将其视为结果进行格式化
        recommendations = product_sequence
    else:
        # ---------------------------------------------------------
        # 🏛️ Vanilla-baseline mode: original SASRec logic
        # ---------------------------------------------------------
        item_sequence = [product_id_map.get(product_asin, -1) for product_asin in product_sequence if product_asin in product_id_map]
        if len(item_sequence) == 0:
            return recommendations  

        item_sequence = item_sequence[-maxlen:]  
        
        positive_indices = list(product_id_map.values()) 
        recommended_items = predict_next_item(model, item_sequence, maxlen, itemnum, 10, device=device, positive_indices=positive_indices)
        
        asin_reverse_map = {v: k for k, v in product_id_map.items()} 
        for item_id in recommended_items:
            asin = asin_reverse_map.get(item_id)
            recommendations.append(asin)
    
    # ---------------------------------------------------------
    # 统一格式化输出（各 agent 模式共用）
    # ---------------------------------------------------------
    # 注意：我们要确保 data['all_products'] 包含我们检索出来的 ASIN
    final_res = []
    for i, asin in enumerate(recommendations):
        product_info = data.get('all_products', {}).get(asin)
        if product_info:
            final_res.append(pretty_product(product_info, i))
        else:
            # 容错：如果全量库里没这个 ASIN，至少返回一个 ASIN ID
            final_res.append(f"Product {i}: ASIN {asin} (Details not found)")
            
    return final_res


get_recommendations_by_history.__info__ = {
    "type": "function",
    "function": {
        "name": "get_recommendations_by_history",
        "description": "Get recommended products based on an input sequence of product asins. Maximum length of the input sequence is 30. Input sequence should follow the time order, with the most recent product asin at the end.",
        "parameters": {
            "type": "object",
            "properties": {
                "product_sequence": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "description": "A product asin from the input sequence.",
                    },
                    "description": "A list of product asins based on which recommendations are to be made. Should follow the time order.",
                },
            },
            "required": ["product_sequence"],
        },
    },
}


load_sasrec_model(os.path.join(FOLDER_PATH,'recommender/checkpoint/pwab.pth'), usernum=usernum, itemnum_=itemnum)
