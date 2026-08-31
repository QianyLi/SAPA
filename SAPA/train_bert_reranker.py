import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os

class Config:
    model_path = os.getenv('PWAB_BERT_MODEL', 'bert-base-uncased')
    data_path = 'data/bert_training_data_flatten.json'
    output_dir = './bert_reranker_save'
    max_length = 512
    batch_size = 16
    epochs = 3
    lr = 2e-5
    warmup_steps = 100
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class RerankerDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        encoding = self.tokenizer(
            item['instruction'],
            item['candidate'],
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'token_type_ids': encoding['token_type_ids'].flatten(),
            'labels': torch.tensor(item['score'], dtype=torch.float)
        }

def train():
    cfg = Config()
    if not os.path.exists(cfg.output_dir):
        os.makedirs(cfg.output_dir)

    with open(cfg.data_path, 'r', encoding='utf-8') as f:
        all_data = json.load(f)

    train_data, val_data = train_test_split(all_data, test_size=0.1, random_state=42)
    print(f"训练集大小: {len(train_data)}, 验证集大小: {len(val_data)}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_path)
    model = AutoModelForSequenceClassification.from_pretrained(cfg.model_path, num_labels=1)
    model.to(cfg.device)

    train_dataset = RerankerDataset(train_data, tokenizer, cfg.max_length)
    val_dataset = RerankerDataset(val_data, tokenizer, cfg.max_length)

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size)

    optimizer = AdamW(model.parameters(), lr=cfg.lr)
    total_steps = len(train_loader) * cfg.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=cfg.warmup_steps, num_training_steps=total_steps)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')

    for epoch in range(cfg.epochs):
        model.train()
        total_train_loss = 0
        print(f"\n--- Epoch {epoch+1}/{cfg.epochs} ---")

        for batch in tqdm(train_loader, desc="Training"):
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(cfg.device)
            attention_mask = batch['attention_mask'].to(cfg.device)
            token_type_ids = batch['token_type_ids'].to(cfg.device)
            labels = batch['labels'].to(cfg.device)

            outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            logits = outputs.logits.squeeze(-1)

            loss = criterion(logits, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(cfg.device)
                attention_mask = batch['attention_mask'].to(cfg.device)
                token_type_ids = batch['token_type_ids'].to(cfg.device)
                labels = batch['labels'].to(cfg.device)

                outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
                logits = outputs.logits.squeeze(-1)

                loss = criterion(logits, labels)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Avg Train Loss: {avg_train_loss:.4f}, Avg Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"发现更好的模型，已保存至 {cfg.output_dir}")
            model.save_pretrained(cfg.output_dir)
            tokenizer.save_pretrained(cfg.output_dir)

if __name__ == "__main__":
    train()
