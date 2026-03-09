# %% [markdown]
# # 🚀 Hybrid ML Pipeline: XGBoost + CodeBERT (GPU-Accelerated)
# 
# **Architecture**: Tabular branch (XGBoost) + NLP branch (CodeBERT) → Fusion Classifier
# 
# **GPU**: All PyTorch operations run on CUDA (RTX 4060)

# %% [markdown]
# ## Cell 1: Import All Libraries

# %%
# ═══════════════════════════════════════════════════════════
# CELL 1: IMPORTS
# ═══════════════════════════════════════════════════════════
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Scikit-Learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression

# XGBoost
import xgboost as xgb

# PyTorch (GPU)
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset

# Hugging Face
from transformers import AutoTokenizer, AutoModel

# Utilities
from tqdm import tqdm

# ── GPU Setup ──
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️  Using device: {device}")
if device.type == 'cuda':
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    try:
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"   VRAM: {vram:.1f} GB")
    except AttributeError:
        print("   VRAM: (info unavailable)")

print("✅ All libraries imported!")

# %% [markdown]
# ## Cell 2: Load Data & Preprocess

# %%
# ═══════════════════════════════════════════════════════════
# CELL 2: LOAD DATA & PREPROCESS
# ═══════════════════════════════════════════════════════════

df = pd.read_csv('frankenstein_cicd_dataset.csv')
print(f"📦 Dataset: {df.shape[0]} rows × {df.shape[1]} cols")

# ── Encode Target ──
le_target = LabelEncoder()
df['failure_type_encoded'] = le_target.fit_transform(df['failure_type'])
print(f"🎯 Target classes ({len(le_target.classes_)}): {list(le_target.classes_)}")

# ── Save error_message BEFORE dropping it (needed for NLP branch) ──
error_messages = df['error_code'].astype(str) + " - " + df['error_message'].astype(str)

# ── Drop columns ──
drop_cols = [
    'pipeline_id', 'run_id', 'timestamp', 'commit_hash', 'author',  # IDs
    'repository',         # Bias — model memorizes repo→failure
    'severity',           # Data Leakage — assigned after failure
    'failure_type',       # Original target string
    'error_message',      # Text — goes to NLP branch, not tabular
    'error_code',         # Too direct a signal — maps to failure type
    'incident_created',   # Post-hoc — assigned after failure
    'rollback_triggered', # Post-hoc — assigned after failure
]
# Also drop severity_encoded if it was created in prior preprocessing
if 'severity_encoded' in df.columns:
    drop_cols.append('severity_encoded')

df_tabular = df.drop(columns=[c for c in drop_cols if c in df.columns])
print(f"🗑️  Dropped: {[c for c in drop_cols if c in df.columns]}")

# (error_code dropped — too direct a signal for XGBoost)

# ── One-Hot Encode LOW-CARDINALITY categorical columns only ──
# ci_tool(5), branch(4), language(6), os(3), cloud_provider(4), failure_stage(3)
low_card_cols = df_tabular.select_dtypes(include=['object']).columns.tolist()
df_tabular = pd.get_dummies(df_tabular, columns=low_card_cols, drop_first=True, dtype=int)
print(f"   One-Hot Encoded (low cardinality): {low_card_cols}")

# ── Convert booleans to int ──
bool_cols = df_tabular.select_dtypes(include=['bool']).columns.tolist()
for col in bool_cols:
    df_tabular[col] = df_tabular[col].astype(int)
    print(f"   Bool→Int '{col}'")

# ── Split Features / Target ──
X = df_tabular.drop(columns=['failure_type_encoded'])
y = df_tabular['failure_type_encoded']

print(f"\n📊 Feature matrix: {X.shape}")
print(f"   Features: {X.columns.tolist()}")
print(f"   All dtypes numeric: {all(X.dtypes != 'object')}")

# ── Train/Test Split (80/20, stratified) ──
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Keep track of indices for aligning error_messages later
train_indices = X_train.index
test_indices = X_test.index

print(f"\n✅ Split: {X_train.shape[0]} train / {X_test.shape[0]} test")

# ── Z-Score Standardization (StandardScaler) for Neural Network branch ──
# Tree models (XGBoost) don't need scaling, but neural networks DO.
# Scaling to mean=0, std=1 helps gradient descent converge faster.
numeric_cols = ['build_duration_sec', 'test_duration_sec',
                'deploy_duration_sec', 'cpu_usage_pct', 'memory_usage_mb', 'retry_count']
# Only scale columns that exist in X_train
numeric_cols = [c for c in numeric_cols if c in X_train.columns]

scaler = StandardScaler()
# Fit on train, transform both train and test (prevents data leakage)
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])

print(f"\n📏 Z-Score Scaled (for NN branch): {numeric_cols}")
print(f"   Mean after scaling (train): {X_train_scaled[numeric_cols].mean().round(4).tolist()}")
print(f"   Std after scaling (train):  {X_train_scaled[numeric_cols].std().round(4).tolist()}")

# %% [markdown]
# ## Cell 3: Train XGBoost Baseline

# %%
# ═══════════════════════════════════════════════════════════
# CELL 3: XGBOOST TABULAR BASELINE
# ═══════════════════════════════════════════════════════════

# XGBoost does NOT need scaled data — trees split on thresholds, not distances.
# So we use X_train/X_test (unscaled) for XGBoost.
xgb_params = dict(
    objective='multi:softmax',
    num_class=10,
    eval_metric='mlogloss',
    n_estimators=300,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    random_state=42,
    tree_method='hist',
)

# Use GPU for XGBoost if CUDA is available
if torch.cuda.is_available():
    xgb_params['device'] = 'cuda'
    print("🚀 Training XGBoost on GPU...")
else:
    print("🚀 Training XGBoost on CPU...")

xgb_model = xgb.XGBClassifier(**xgb_params)
xgb_model.fit(X_train, y_train)

# 🟢 NEW: Save the XGBoost Expert!
xgb_model.save_model('xgboost_expert.json')
print("💾 XGBoost Tabular Expert saved as 'xgboost_expert.json'!")

y_pred_xgb = xgb_model.predict(X_test)
baseline_acc = accuracy_score(y_test, y_pred_xgb) * 100

print(f"\n📊 XGBoost Tabular Baseline Accuracy: {baseline_acc:.2f}%\n")
print("Classification Report:")
print(classification_report(y_test, y_pred_xgb, target_names=le_target.classes_))

# %% [markdown]
# ## Cell 4: Tokenize Error Messages with CodeBERT

# %%
# ═══════════════════════════════════════════════════════════
# CELL 4: TOKENIZE TEXT WITH CODEBERT
# ═══════════════════════════════════════════════════════════

tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")

# Get error messages aligned with train/test split
texts_train = error_messages.loc[train_indices].tolist()
texts_test  = error_messages.loc[test_indices].tolist()

print(f"Tokenizing {len(texts_train)} training texts...")
train_encodings = tokenizer(
    texts_train,
    truncation=True,
    padding='max_length',
    max_length=512,
    return_tensors='pt'
)

print(f"Tokenizing {len(texts_test)} test texts...")
test_encodings = tokenizer(
    texts_test,
    truncation=True,
    padding='max_length',
    max_length=512,
    return_tensors='pt'
)

print(f"✅ train_encodings input_ids: {train_encodings['input_ids'].shape}")
print(f"✅ test_encodings  input_ids: {test_encodings['input_ids'].shape}")

# %% [markdown]
# ## Cell 5: Define CICDHybridModel + Create DataLoaders

# %%
# ═══════════════════════════════════════════════════════════
# CELL 5: DEFINE HYBRID MODEL + DATALOADERS
# ═══════════════════════════════════════════════════════════

# ────────── Hybrid Model Definition ──────────
class CICDTextModel(nn.Module):
    def __init__(self, num_classes=10): 
        super().__init__()
        # The Reader
        self.bert = AutoModel.from_pretrained("microsoft/codebert-base")
        
        # We freeze the early layers to save VRAM on your RTX 4060
        for param in list(self.bert.parameters())[:-4]:
            param.requires_grad = False
            
        # The Classifier (Only needs 768 dimensions now!)
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )
    
    def forward(self, input_ids, attention_mask):
        # We only pass text in now!
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = bert_output.last_hidden_state[:, 0, :]
        return self.classifier(cls_embedding)

# ────────── Convert to Tensors ──────────
train_labels_tensor  = torch.tensor(y_train.values,  dtype=torch.long)
test_labels_tensor   = torch.tensor(y_test.values,   dtype=torch.long)

# ────────── Create DataLoaders ──────────
batch_size = 32

train_dataset = TensorDataset(
    train_encodings['input_ids'],
    train_encodings['attention_mask'],
    train_labels_tensor
)

test_dataset = TensorDataset(
    test_encodings['input_ids'],
    test_encodings['attention_mask'],
    test_labels_tensor
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

# ────────── Initialize Model on GPU ──────────
model = CICDTextModel(num_classes=10)
model = model.to(device)

print(f"✅ Model on {device}")
print(f"   Train batches: {len(train_loader)}")
print(f"   Test batches:  {len(test_loader)}")

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"   Total params: {total_params:,}")
print(f"   Trainable params: {trainable_params:,}")

# %% [markdown]
# ## Cell 6: Train the Hybrid Model on GPU

# %%
# ═══════════════════════════════════════════════════════════
# CELL 6: TRAIN HYBRID MODEL ON GPU
# ═══════════════════════════════════════════════════════════

optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=2e-5,
    weight_decay=0.01
)
criterion = nn.CrossEntropyLoss()

EPOCHS = 5

print(f"🚀 Training Text Expert on {device} for {EPOCHS} epochs...\n")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=True)
    
    for batch in progress_bar:
        input_ids, attention_mask, labels = batch
        
        # ── Move everything to GPU ──
        input_ids      = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels         = labels.to(device)
        
        # ── Forward pass ──
        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        
        # ── Backward pass ──
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # ── Track metrics ──
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        progress_bar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'acc': f"{correct/total*100:.1f}%"
        })
    
    avg_loss = total_loss / len(train_loader)
    train_acc = correct / total * 100
    print(f"  → Epoch {epoch+1}: Loss={avg_loss:.4f}, Train Acc={train_acc:.2f}%")

    # ── Free GPU cache after each epoch ──
    torch.cuda.empty_cache()

print("\n✅ Training complete!")

# %% [markdown]
# ## Cell 7: Evaluate & Compare Results

# %%
# ═══════════════════════════════════════════════════════════
# CELL 7: EVALUATE HYBRID MODEL & COMPARE
# ═══════════════════════════════════════════════════════════

model.eval()
all_preds = []
all_labels = []

print("🔍 Evaluating Hybrid Model on test set...")

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Testing"):
        input_ids, attention_mask, labels = batch
        
        # Move to GPU
        input_ids      = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        
        logits = model(input_ids, attention_mask)
        preds = torch.argmax(logits, dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

text_acc = accuracy_score(all_labels, all_preds) * 100

# ── Final Comparison ──
print(f"\n{'='*55}")
print(f"  📊 XGBoost Tabular Baseline:    {baseline_acc:.2f}%")
print(f"  📊 CodeBERT Text Expert:        {text_acc:.2f}%")
print(f"  📊 Improvement:                 {text_acc - baseline_acc:+.2f}%")
print(f"{'='*55}")

print("\n📋 CodeBERT Text Expert — Classification Report:")
print(classification_report(all_labels, all_preds, target_names=le_target.classes_))

# ── Save the model ──
torch.save(model.state_dict(), 'codebert_text_expert.pth')
print("💾 CodeBERT Text Expert saved as 'codebert_text_expert.pth'")
