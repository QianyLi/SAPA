import json
from typing import Any, Dict, List

from termcolor import colored
import torch
import torch.nn.functional as F

def display_conversation(
    messages: List[Dict[str, Any]], include_system_messages: bool = True
) -> str:
    message_displays = []
    for message in messages:
        if not isinstance(message, dict):
            message_displays.append(str(message.tool_calls[0].function))
            continue
        if message["role"] == "system" and include_system_messages:
            message_displays.append(f"system: {message['content']}")
        elif message["role"] == "user":
            message_displays.append(f"user: {message['content']}")
        elif message["role"] == "assistant" and message.get("tool_calls"):
            message_displays.append(f"assistant: {json.dumps(message['tool_calls'][0])}")
        elif message["role"] == "assistant" and not message.get("tool_calls"):
            message_displays.append(f"assistant: {message['content']}")
        elif message["role"] == "tool":
            message_displays.append(f"tool ({message['name']}): {message['content']}")
    return "\n".join(message_displays)


def pretty_print_conversation(messages: List[Dict[str, Any]]) -> None:
    role_to_color = {
        "system": "red",
        "user": "green",
        "assistant": "yellow",
        "tool": "magenta",
    }

    for message in messages:
        if not isinstance(message, dict):
            print(colored(str(message.tool_calls[0].function)))
            continue
        if message["role"] == "system":
            print(colored(f"system: {message['content']}\n", role_to_color[message["role"]]))
        elif message["role"] == "user":
            print(colored(f"user: {message['content']}\n", role_to_color[message["role"]]))
        elif message["role"] == "assistant" and message.get("function_call"):
            print(
                colored(f"assistant: {message['function_call']}\n", role_to_color[message["role"]])
            )
        elif message["role"] == "assistant" and not message.get("function_call"):
            print(colored(f"assistant: {message['content']}\n", role_to_color[message["role"]]))
        elif message["role"] == "tool":
            print(
                colored(
                    f"tool ({message['name']}): {message['content']}\n",
                    role_to_color[message["role"]],
                )
            )


def message_to_action(message):
    if message.tool_calls is not None:
        tool_call = message.tool_calls[0]
        return {
            "name": tool_call.function.name,
            "arguments": json.loads(tool_call.function.arguments),
        }
    else:
        return {"name": "respond", "arguments": {"content": message.content}}


def message_to_dict(message):
    if isinstance(message, dict):
        return message
    else:
        return {"role": "assistant", "function_call": str(message.tool_calls[0].function)}


PARAM_PROMPT = '''Below is an instruction that describes a task. Generate a tool parameter that appropriately completes the request. 
### Instruction: <Insturction> 

Memory: <Memory>

Tool: <Tool>

### Tool Parameter:
'''


PRODUCT_PROMPT = '''
The product is:
Title: <Title>
Price: <Price>
Store: <Store>
Main Category: <Main Category>
'''


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
        del encoded_input
        del model_output
        del sentence_embeddings
        torch.cuda.empty_cache()
    return torch.cat(all_embeddings, dim=0)


def prettify_product_info(product_info):
    res = PRODUCT_PROMPT.replace('<Title>', product_info['title'])
    res = res.replace('<Price>', str(product_info['price']))
    res = res.replace('<Store>', str(product_info['store']))
    res = res.replace('<Main Category>', str(product_info['main_category']))
    return res

def load_input_prompt(instruction, type, product, memory, tokenizer, mem_length):

    task = instruction
    task_type = type
    product_info = product
    mem = memory
    if task_type == 'search':
        prefix_text = PARAM_PROMPT.replace('<Insturction>', task)
        memory_text = mem
        tool_text = 'search_product_by_query'
        tokenized_memory = tokenizer(
            memory_text,
            return_tensors=None,
            truncation=True,
            max_length=mem_length  
        )
        truncated_memory_ids = tokenized_memory["input_ids"]
        memory_text_truncated = tokenizer.decode(truncated_memory_ids, skip_special_tokens=True)
        truncated_full_text = prefix_text.replace('<Memory>', memory_text_truncated).replace('<Tool>', tool_text)

    elif task_type == 'recommend':
        prefix_text = PARAM_PROMPT.replace('<Insturction>', task)
        memory_text = mem
        tool_text = 'get_recommendations_by_history'
        tokenized_memory = tokenizer(
            memory_text,
            return_tensors=None,
            truncation=True,
            max_length=mem_length  
        )
        truncated_memory_ids = tokenized_memory["input_ids"]
        memory_text_truncated = tokenizer.decode(truncated_memory_ids, skip_special_tokens=True)
        truncated_full_text = prefix_text.replace('<Memory>', memory_text_truncated).replace('<Tool>', tool_text)
    
    elif task_type == 'review':
        product_text = prettify_product_info(product_info)
        prefix_text = PARAM_PROMPT.replace('<Insturction>', task + product_text)
        memory_text = mem
        tool_text = 'add_product_review'
        tokenized_memory = tokenizer(
            memory_text,
            return_tensors=None,
            truncation=True,
            max_length=mem_length  
        )
        truncated_memory_ids = tokenized_memory["input_ids"]
        memory_text_truncated = tokenizer.decode(truncated_memory_ids, skip_special_tokens=True)
        truncated_full_text = prefix_text.replace('<Memory>', memory_text_truncated).replace('<Tool>', tool_text)

    return truncated_full_text


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

MINI_HISTORY_PROMPT = '''
MEMORY <NUM>:
Product:
- Title: <TITLE>
- Parent Asin: <PARENT_ASIN>
- Main Category: <MAIN_CATEGORY>
Review:
- Rating: <RATING>
- Text: <TEXT>
- Timestamp: <TIMESTAMP>
'''

INTEREC_MEMORY_PROMPT = '''
- Title: <TITLE>
- Parent Asin: <PARENT_ASIN>
- Main Category: <MAIN_CATEGORY>
- Rating: <RATING>
- Text: <TEXT>
'''

REFLECTION_INST = """
You will be given the history of a past experience where you were placed in an environment and given a task to complete. 
In each attempt, you had the option to call tool and provide specific parameters for that tool. 
Your goal is to reflect on the feedback provided by the user regarding your previous attempt.
Reflect on the specific tool you selected and the parameters you provided. 
Think about what went wrong in your decision-making process regarding the tool selection and parameter usage. 
Devise a concise, new plan of action that improves on the mistakes made, with a focus on selecting the right tool and providing the appropriate parameters in future experiences.

Instruction:
<TASK>

Your action:
<ACTIONS>

User feedback:
<FEEDBACK>
"""

REACT_INST = """
# Instruction:
You need to act as an personalized shopping agent that use the above tools to help the user according to the above rules.
At each step, your generation should have exactly the following format and have exactly 6 lines (without ```):

```
Thought:
A single line of reasoning to process the context and inform the decision making. Do not include extra lines.
Action:
The name of the action to take. It has to come from "Available tools", or "respond" to respond to the user.
Arguments:
The arguments to the action in json format. If the action is "respond", the argument is the response to the user.
```

You should not use made-up or placeholder arguments.

For example, if the user says "I want to know the current weather of San Francisco", and there is such a tool available
```json
{
    "type": "function",
    "function": {
        "name": "get_current_weather",
        "description": "Get the current weather",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                },
                "format": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "The temperature unit to use. Infer this from the users location.",
                },
            },
            "required": ["location", "format"],
        },
    }
}
```

Your response can be like this:
```
Thought:
Since the user asks for the weather of San Francisco in USA, the unit should be in fahrenheit. I can query get_current_weather to get the weather.
Action:
get_current_weather
Arguments:
{"location": "San Francisco, CA", "format": "fahrenheit"}
```

Try to be helpful and always follow the rules.
Distingush the task carefully and use corresponding tool.
Wrong tool selection will lead to zero score, even if the result is correct.
"""

RECMIND_ST_PROMPT = '''As a personalized shopping agent, you can help users search for products, recommend products or complete their reviews.

Rules:
- The user will provide user_id and a request.
- You need to use the most appropriate tool to find the product or fill the review that matches the user's request.
- For different requests, you may need to use different tools. Correct tool selection and better tool input will help you get better results.
- You are not allowed to interact with the user. Make the best tool call based on the user's request.
- You are allowed to use get_product_details_by_asin to get more details about products in user's history before making a final task tool call.
- But once you make a tool call to search_product_by_query, get_recommendations_by_history or add_product_review, task will be over.
- The memory about the user is provided, includes all user purchases and reviews, you should use it to help you formulate better tool calls.
'''

RECMIND_MT_PROPMT = '''As a personalized shopping agent, you can help users search for products, recommend products or complete their reviews.

Rules:
- The user will provide user_id and a request.
- You need to use the most appropriate tool to find the product or fill the review that matches the user's request.
- For different requests, you may need to use different tools from search_product_by_query, get_recommendations_by_history or add_product_review. 
- Correct tool selection and better tool input will help you get better results.
- You are allowed to interact with the user or make tool calls, but steps are limited to <NUM>, and less steps are preferred.
- Your main goal is to help the user complete the task as accurately and efficiently as possible, do not keep responding to the user, focus on making the best tool calls.
- You are allowed to use get_product_details_by_asin to get more details about products in user's history before making task tool calls like search_product_by_query, get_recommendations_by_history or add_product_review.
- When you think you have found the best input for the task tool calls, you can end the task by making a 'stop' call. Ending the task early will give you a higher score.
- If the user expresses satisfaction with the results, you should end the task by making a 'stop' call and do not respond to the user anymore.
- You should not make up any information or knowledge not provided from the user or the tools, or give subjective comments or recommendations.
- You should at most make one tool call at a time, and if you take a tool call, you should not respond to the user at the same time. If you respond to the user, you should not make a tool call.
- The memory about the user is provided, includes all user purchases and reviews, you should use it to help you formulate better tool calls.
'''

INTEREC_PROMPT = '''As a personalized shopping agent, you can help users search for products, recommend products or complete their reviews.

Rules:
- The user will provide user_id and a request.
- You need to use the most appropriate tool to find the product or fill the review that matches the user's request.
- For different requests, you may need to use different tools. Correct tool selection and better tool input will help you get better results.
- For every request, only one type of tool is needed.
- You are allowed to interact with the user or make tool calls, but steps are limited to <NUM>, and less steps are preferred.
- Your main goal is to help the user complete the task as accurately and efficiently as possible, do not keep responding to the user, focus on making the best tool calls.
- When you think you have found the best input for the task tool calls, you can end the task by making a 'stop' call. Ending the task early will give you a higher score.
- If the user expresses satisfaction with the results, you should end the task by making a 'stop' call and do not respond to the user anymore.
- You should not make up any information or knowledge not provided from the user or the tools, or give subjective comments or recommendations.
- You should at most make one tool call at a time, and if you take a tool call, you should not respond to the user at the same time. If you respond to the user, you should not make a tool call.
- The memory about the user is provided, includes history, like, dislike and expect, you should use it to help you formulate better tool calls.
'''

INTEREC_UPDATE_MEM_PROMPT = '''Your task is to extract user profile from the conversation.
The profile consists of three parts: like, dislike and expect. Each part is a list.
You should return a json-format string.
Here are some examples.

> Conversations
User: My history is ITEM-1, ITEM-2, ITEM-3. Now I want something new.
Assistent: Based on your preference, I recommend you ITEM-17, ITEM-19, ITEM-30.
User: I don't like those items, give me more options.
Assistent: Based on your feedbacks, I recommend you ITEM-5, ITEM-100.
User: I think ITEM-100 may be very interesting. I may like it. 
> Profiles
{"like": ["ITEM-100"], "dislike": ["ITEM-17", "ITEM-19", "ITEM-30"], "expect": []}

> Conversations
User: I used to enjoy ITEM-89, ITEM-11, ITEM-78, ITEM-67. Now I want something new.
Assistent: Based on your preference, I recommend you ITEM-53, ITEM-10.
User: I think ITEM-10 may be very interesting, but I don't like it.
Assistent: Based on your feedbacks, I recommend you ITEM-88, ITEM-70.
User: I don't like those items, I want something new like books.
> Profiles
{"like": [], "dislike": ["ITEM-10", "ITEM-88", "ITEM-70"], "expect": ["books"]}

Only keep the parent_asin in the like and dislike to represent the product. Expect can contain item names or categories.
Now extract user profiles from below conversation: 
> Conversation
{conversation}
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

TS_AGENT_PROMPT = '''As a personalized shopping agent, you can help users search for products, recommend products or complete their reviews.

Rules:
- The user will provide user_id and a request.
- You need to use the most appropriate tool to find the product or fill the review that matches the user's request.
- You are not allowed to interact with the user. Make the best tool call based on the user's request.
- You have only one chance to make a tool call, so make sure you have the best input for the tool.
- The tool will be provided, you need to use the tool and provide the most appropriate input for the tool. Do not use other tools.
- Formulate the best input for the tool based on the user's request and the memory provided.
'''

TS_AGENT_MT_PROMPT = '''As a personalized shopping agent, you can help users search for products, recommend products or complete their reviews.

Rules:
- The user will provide user_id and a request.
- You need to use the tools to find the product or fill the review that matches the user's request.
- The tool will be provided, you need to use the tool and provide the most appropriate input for the tool. Do not use any other tools.
- Formulate the best input for the tool based on the user's request and the memory provided.
- You are allowed to interact with the user by 'respond' to ask for more information or feedback, but steps are limited to <NUM>, and less steps are preferred.
- Your main goal is to help the user complete the task as accurately and efficiently as possible, do not keep responding to the user, focus on making the better tool calls.
- The evaluation will be based on the ranking of the target product in search and recommendation tasks, and the similarity of the review in the review task.
- When you think you have found the best input for the task tool calls, you can end the task by making a 'stop' call. 
- You should not make up any information or knowledge not provided from the user or the tools, or give subjective comments or recommendations.
- You should at most make one tool call at a time, and if you take a tool call, you should not respond to the user at the same time. If you respond to the user, you should not make a tool call.
'''


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


def mini_pretty_history(item, num):
    res = MINI_HISTORY_PROMPT.replace("<TITLE>", item['product_info']['title'])
    res = res.replace("<PARENT_ASIN>", item['product_info']['parent_asin'])
    res = res.replace("<MAIN_CATEGORY>", str(item['product_info']['main_category']))
    res = res.replace("<RATING>", str(item['review']['rating']))
    res = res.replace("<TEXT>", item['review']['text'])
    res = res.replace("<TIMESTAMP>", str(item['review']['timestamp']))
    res = res.replace("<NUM>", str(num))
    return res


def interecagent_pretty_history(item):
    res = INTEREC_MEMORY_PROMPT.replace("<TITLE>", item['product_info']['title'])
    res = res.replace("<PARENT_ASIN>", item['product_info']['parent_asin'])
    res = res.replace("<MAIN_CATEGORY>", str(item['product_info']['main_category']))
    res = res.replace("<RATING>", str(item['review']['rating']))
    res = res.replace("<TEXT>", item['review']['text'])
    return res


def sup_search_pretty_history(item):
    res = TS_SEARCH_PROMPT.replace("<TITLE>", item['product_info']['title'])
    res = res.replace("<MAIN_CATEGORY>", str(item['product_info']['main_category']))
    res = res.replace("<PRICE>", str(item['product_info']['price']))
    res = res.replace("<STORE>", str(item['product_info']['store']))
    return res


def sup_rec_pretty_history(item):
    res = TS_REC_PROMPT.replace("<TITLE>", item['product_info']['title'])
    res = res.replace("<MAIN_CATEGORY>", str(item['product_info']['main_category']))
    res = res.replace("<ASIN>", str(item['product_info']['parent_asin']))
    return res


def sup_review_pretty_history(item):
    res = TS_REV_PROMPT.replace("<RATING>", str(item['review']['rating']))
    res = res.replace("<TEXT>", item['review']['text'])
    return res


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