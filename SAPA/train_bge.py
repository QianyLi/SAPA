import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from tqdm import tqdm
import os
import numpy as np
from sklearn.model_selection import train_test_split


class Config:

    model_name = os.getenv('PWAB_RERANKER_MODEL', 'BAAI/bge-reranker-v2-m3')

    data_path = 'data/bert_training_data.json'
    output_dir = './listwise_reranker_best'

    max_length = 1024
    list_size = 10
    batch_size = 2
    accumulation_steps = 8

    epochs = 15
    lr = 2e-5
    warmup_steps = 50


class ListwiseJSONDataset(Dataset):
    def __init__(self, raw_data_list, tokenizer, max_length=512, list_size=10):
        'Documentation.'
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.list_size = list_size
        self.data = []


        for instruction, candidates_dict in raw_data_list:
            candidates = list(candidates_dict.keys())
            scores = list(candidates_dict.values())


            if len(candidates) < list_size:
                continue

            candidates = candidates[:list_size]
            scores = scores[:list_size]


            score_gap = max(scores) - min(scores)
            if score_gap < 0.05:
                continue

            best_idx = int(np.argmax(scores))

            self.data.append({
                'instruction': instruction,
                'candidates': candidates,
                'label_idx': best_idx
            })

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


def evaluate(model, dataloader, device, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            scores = model(input_ids, attention_mask)
            loss = criterion(scores, labels)

            total_loss += loss.item() * labels.size(0)

            preds = torch.argmax(scores, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total
    acc = correct / total
    return avg_loss, acc


def train():
    cfg = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists(cfg.output_dir):
        os.makedirs(cfg.output_dir)


    print(f"正在读取原始文件: {cfg.data_path}")
    with open(cfg.data_path, 'r', encoding='utf-8') as f:
        raw_dict = json.load(f)


    raw_list = list(raw_dict.items())


    train_raw, val_raw = train_test_split(raw_list, test_size=0.1, random_state=42)
    print(f"数据切分: 训练集 {len(train_raw)} 条, 验证集 {len(val_raw)} 条")


    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    train_dataset = ListwiseJSONDataset(train_raw, tokenizer, max_length=cfg.max_length, list_size=cfg.list_size)
    val_dataset = ListwiseJSONDataset(val_raw, tokenizer, max_length=cfg.max_length, list_size=cfg.list_size)

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=2)

    model = ListwiseReranker(cfg.model_name)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=cfg.lr)
    total_steps = len(train_loader) * cfg.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=cfg.warmup_steps, num_training_steps=total_steps)
    criterion = nn.CrossEntropyLoss()


    best_val_loss = float('inf')

    print(f"\n开始训练...")

    for epoch in range(cfg.epochs):

        model.train()
        total_train_loss = 0
        train_correct = 0
        train_total = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.epochs} [Train]")

        for step, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            scores = model(input_ids, attention_mask)
            loss = criterion(scores, labels)

            loss = loss / cfg.accumulation_steps
            loss.backward()


            current_loss = loss.item() * cfg.accumulation_steps
            total_train_loss += current_loss * labels.size(0)

            with torch.no_grad():
                preds = torch.argmax(scores, dim=1)
                train_correct += (preds == labels).sum().item()
                train_total += labels.size(0)

            if (step + 1) % cfg.accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            progress_bar.set_postfix({'loss': f"{current_loss:.4f}"})

        avg_train_loss = total_train_loss / train_total
        avg_train_acc = train_correct / train_total


        val_loss, val_acc = evaluate(model, val_loader, device, criterion)

        print(f"\n>>> Epoch {epoch+1} Report:")
        print(f"    Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_acc:.2%}")
        print(f"    Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.2%}")


        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"    [★ New Best] Saving model to {cfg.output_dir} ...")

            model.bert.save_pretrained(cfg.output_dir)
            tokenizer.save_pretrained(cfg.output_dir)
        else:
            print(f"    [No Improve] Best Val Loss is {best_val_loss:.4f}")

if __name__ == "__main__":
    train()
