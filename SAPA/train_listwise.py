import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from tqdm import tqdm
import os
import numpy as np

class Config:
    model_name = os.getenv('PWAB_BERT_MODEL', 'bert-base-uncased')

    data_path = 'data/bert_training_data.json'

    output_dir = './listwise_reranker_output_lr'

    max_length = 512
    list_size = 10

    batch_size = 2

    accumulation_steps = 8

    epochs = 15
    lr = 2e-5
    warmup_steps = 50

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

        for instruction, candidates_dict in raw_data.items():
            candidates = list(candidates_dict.keys())
            scores = list(candidates_dict.values())

            if len(candidates) < list_size:
                continue

            candidates = candidates[:list_size]
            scores = scores[:list_size]

            score_gap = max(scores) - min(scores)
            if score_gap < 0.05:
                skipped_count += 1
                continue

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

        pairs = [[instruction, cand] for cand in candidates]

        encoding = self.tokenizer(
            pairs,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'label': torch.tensor(item['label_idx'], dtype=torch.long)
        }

class ListwiseReranker(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.bert = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)

    def forward(self, input_ids, attention_mask):
        batch_size, list_size, seq_len = input_ids.shape

        flat_input_ids = input_ids.view(-1, seq_len)
        flat_mask = attention_mask.view(-1, seq_len)

        outputs = self.bert(flat_input_ids, attention_mask=flat_mask)

        scores = outputs.logits.view(batch_size, list_size)

        return scores

def train():
    cfg = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists(cfg.output_dir):
        os.makedirs(cfg.output_dir)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    dataset = ListwiseJSONDataset(cfg.data_path, tokenizer, max_length=cfg.max_length, list_size=cfg.list_size)

    if len(dataset) == 0:
        print("错误: 没有有效的训练数据！请检查数据格式或分数。")
        return

    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=2)
    model = ListwiseReranker(cfg.model_name)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=cfg.lr)
    total_steps = len(dataloader) * cfg.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=cfg.warmup_steps, num_training_steps=total_steps)

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
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            scores = model(input_ids, attention_mask)

            loss = criterion(scores, labels)

            loss = loss / cfg.accumulation_steps
            loss.backward()

            with torch.no_grad():
                preds = torch.argmax(scores, dim=1)
                correct_predictions += (preds == labels).sum().item()
                total_samples += labels.size(0)
                current_loss_val = loss.item() * cfg.accumulation_steps
                total_loss += current_loss_val

            if (step + 1) % cfg.accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            progress_bar.set_postfix({'loss': f"{current_loss_val:.4f}", 'acc': f"{correct_predictions/total_samples:.2%}"})

        avg_loss = total_loss / len(dataloader)
        avg_acc = correct_predictions / total_samples
        print(f"\n>>> Epoch {epoch+1} 结束. Avg Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.4f}")

        save_path = os.path.join(cfg.output_dir, f"epoch-{epoch+1}")
        print(f"保存模型至: {save_path}")
        model.bert.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)

if __name__ == "__main__":
    train()
