import lightgbm as lgb
from sklearn.model_selection import train_test_split
import numpy as np

# --- CONFIGURATION ---
TRAIN_FULL_FILE = 'ltr_train_full_without_cosine.txt'
MODEL_FILE = 'ltr_model_without_cosine.txt'
VALIDATION_SIZE = 0.1 # <-- 我们将从训练数据中划分 10% 作为验证集
RANDOM_STATE = 42

# --- MAIN SCRIPT ---

print("Loading full training data...")
# LightGBM 不能直接从文件划分，所以我们需要先读入数据
# 为了处理 qid，我们需要一个更手动的加载过程
X_train_full = []
y_train_full = []
q_train_full = []

with open(TRAIN_FULL_FILE, 'r') as f:
    for line in f:
        parts = line.strip().split()
        y_train_full.append(int(parts[0]))
        q_train_full.append(int(parts[1].split(':')[1]))
        
        features = [float(p.split(':')[1]) for p in parts[2:]]
        X_train_full.append(features)

X_train_full = np.array(X_train_full)
y_train_full = np.array(y_train_full)
q_train_full = np.array(q_train_full)

# --- 核心修改：根据 qid 进行分组划分 ---
# 我们不能简单地随机划分行，这会打乱查询组。我们需要按 query_id 来划分。
unique_qids = np.unique(q_train_full)
train_qids, val_qids = train_test_split(unique_qids, test_size=VALIDATION_SIZE, random_state=RANDOM_STATE)

# 根据划分好的 qid，来切分数据集
train_indices = np.isin(q_train_full, train_qids)
val_indices = np.isin(q_train_full, val_qids)

X_train, y_train, q_train = X_train_full[train_indices], y_train_full[train_indices], q_train_full[train_indices]
X_val, y_val, q_val = X_train_full[val_indices], y_train_full[val_indices], q_train_full[val_indices]

# LightGBM 需要知道每个查询组的大小
train_group = np.bincount(q_train)[train_qids]
val_group = np.bincount(q_val)[val_qids]

train_data = lgb.Dataset(X_train, label=y_train, group=train_group)
val_data = lgb.Dataset(X_val, label=y_val, group=val_group, reference=train_data)

print(f"Data split complete: {len(train_qids)} queries for training, {len(val_qids)} for validation.")


# --- 训练过程与之前完全一样 ---
params = {
    'objective': 'lambdarank',
    'metric': 'ndcg',
    # ... 其他参数 ...
    'label_gain': [0, 1, 2, 3, 4]
}

print("\nTraining LightGBM Ranker model...")
model = lgb.train(
    params,
    train_data,
    valid_sets=[val_data],
    callbacks=[lgb.early_stopping(100, verbose=True)]
)

print("\nTraining complete!")
model.save_model(MODEL_FILE)
print(f"Model saved to: {MODEL_FILE}")