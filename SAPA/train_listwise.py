import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from tqdm import tqdm
import os
import numpy as np

# ================= 配置区域 =================
class Config:
    # 模型路径 (可以是本地路径，也可以是 huggingface id)
    model_name = os.getenv('PWAB_BERT_MODEL', 'bert-base-uncased')
    
    # 你的数据文件路径
    data_path = 'data/bert_training_data.json' 
    
    # 输出保存路径
    output_dir = './listwise_reranker_output_lr'
    
    max_length = 512        # 文本最大长度 (如显存不够可调小)
    list_size = 10          # 每次对比多少个候选项 (你的数据是10)
    
    # 【显存关键参数】
    # 物理Batch: 每次喂给GPU多少个Query。
    # 实际进入BERT的句子数 = batch_size * list_size
    # 举例: 2 * 10 = 20 个句子。如果显存只有12G-16G，建议设为 1 或 2
    batch_size = 2          
    
    # 梯度累积: 累积多少步更新一次参数。
    # 等效 Batch Size = batch_size * accumulation_steps
    accumulation_steps = 8  
    
    epochs = 15
    lr = 2e-5
    warmup_steps = 50

# ================= 1. Dataset (包含过滤逻辑) =================
class ListwiseJSONDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512, list_size=10):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.list_size = list_size
        self.data = []
        
        print(f"正在加载数据: {data_path} ...")
        with open(data_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
            
        skipped_count = 0
        
        # 解析数据
        for instruction, candidates_dict in raw_data.items():
            candidates = list(candidates_dict.keys())
            scores = list(candidates_dict.values())
            
            # --- 核心过滤逻辑 ---
            # 1. 如果候选项少于 list_size，跳过 (为了对其batch)
            if len(candidates) < list_size:
                continue
                
            # 2. 截取前10个 (如果多于10个)
            candidates = candidates[:list_size]
            scores = scores[:list_size]
            
            # 3. 【关键】过滤掉全为0或全为1的样本 (最大分 == 最小分)
            # 这种样本会导致模型无法学习"哪个更好"，必须剔除
            score_gap = max(scores) - min(scores)
            # 如果分差太小（说明这一组质量都很接近），过滤掉
            if score_gap < 0.05: 
                skipped_count += 1
                continue
            
            # 4. 找到分最高的索引 (Label)
            # 注意: 即使是 1.0000001 和 0.999 这种微小差距，argmax也能找出来
            best_idx = int(np.argmax(scores))
            
            self.data.append({
                'instruction': instruction,
                'candidates': candidates,
                'label_idx': best_idx
            })
            
        print(f"数据加载完成: 总数 {len(raw_data)}, 有效 {len(self.data)}, 过滤无效/平局样本 {skipped_count}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        instruction = item['instruction']
        candidates = item['candidates']
        
        # 构造 Input Pairs: [[Inst, Cand1], [Inst, Cand2], ...]
        pairs = [[instruction, cand] for cand in candidates]
        
        # Tokenize
        encoding = self.tokenizer(
            pairs,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            # 输出形状: [list_size, seq_len] -> [10, 256]
            'input_ids': encoding['input_ids'],         
            'attention_mask': encoding['attention_mask'],
            'label': torch.tensor(item['label_idx'], dtype=torch.long)
        }

# ================= 2. Model Wrapper (处理 List 维度) =================
class ListwiseReranker(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        # num_labels=1: 我们依然让BERT输出一个标量分数
        self.bert = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)

    def forward(self, input_ids, attention_mask):
        # input_ids shape: [Batch, List, Seq]
        batch_size, list_size, seq_len = input_ids.shape
        
        # 1. 压扁 (Flatten): [B * L, S]
        # BERT 只能吃二维数据，不知道什么是 List
        flat_input_ids = input_ids.view(-1, seq_len)
        flat_mask = attention_mask.view(-1, seq_len)
        
        # 2. BERT Forward
        outputs = self.bert(flat_input_ids, attention_mask=flat_mask)
        
        # 3. 还原 (Reshape): [B * L, 1] -> [B, L]
        # 这样每一行就是一个 Query 的 10 个打分
        scores = outputs.logits.view(batch_size, list_size)
        
        return scores

# ================= 3. 训练主流程 =================
def train():
    cfg = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if not os.path.exists(cfg.output_dir):
        os.makedirs(cfg.output_dir)

    # 1. 准备组件
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    dataset = ListwiseJSONDataset(cfg.data_path, tokenizer, max_length=cfg.max_length, list_size=cfg.list_size)
    
    # 如果数据都被过滤光了，直接退出
    if len(dataset) == 0:
        print("错误: 没有有效的训练数据！请检查数据格式或分数。")
        return

    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=2)
    model = ListwiseReranker(cfg.model_name)
    model.to(device)

    # 2. 优化器
    optimizer = AdamW(model.parameters(), lr=cfg.lr)
    total_steps = len(dataloader) * cfg.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=cfg.warmup_steps, num_training_steps=total_steps)
    
    # 3. Loss Function
    # CrossEntropyLoss 会自动对 inputs 做 softmax，然后计算 label 对应位置的负对数似然
    criterion = nn.CrossEntropyLoss()

    print(f"\n开始训练 | Device: {device} | Batches: {len(dataloader)}")
    print(f"逻辑: 1个 Batch 有 {cfg.batch_size} 个 Query，每个 Query 对比 {cfg.list_size} 个 Candidate。")
    
    model.train()
    
    for epoch in range(cfg.epochs):
        total_loss = 0
        correct_predictions = 0
        total_samples = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{cfg.epochs}")
        
        for step, batch in enumerate(progress_bar):
            # 搬运数据到 GPU
            input_ids = batch['input_ids'].to(device)       # [B, 10, Seq]
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)              # [B] (值在 0-9 之间)

            # 前向传播
            scores = model(input_ids, attention_mask)       # [B, 10]

            # 计算损失
            loss = criterion(scores, labels)
            
            # 梯度累积标准化
            loss = loss / cfg.accumulation_steps
            loss.backward()

            # 统计准确率 (Top-1 Accuracy)
            with torch.no_grad():
                preds = torch.argmax(scores, dim=1)
                correct_predictions += (preds == labels).sum().item()
                total_samples += labels.size(0)
                # 还原用于显示的 Loss
                current_loss_val = loss.item() * cfg.accumulation_steps
                total_loss += current_loss_val

            # 执行优化步 (当累积够了步数)
            if (step + 1) % cfg.accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # 梯度裁剪
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
            progress_bar.set_postfix({'loss': f"{current_loss_val:.4f}", 'acc': f"{correct_predictions/total_samples:.2%}"})

        avg_loss = total_loss / len(dataloader)
        avg_acc = correct_predictions / total_samples
        print(f"\n>>> Epoch {epoch+1} 结束. Avg Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.4f}")
        
        # 保存模型 (保存原始 BERT，方便后续直接加载)
        save_path = os.path.join(cfg.output_dir, f"epoch-{epoch+1}")
        print(f"保存模型至: {save_path}")
        model.bert.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)

if __name__ == "__main__":
    # 请确保目录下有 data.json
    train()
