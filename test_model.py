import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import xgboost as xgb
import numpy as np
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

# 1. REBUILD THE EXACT ARCHITECTURE FOR THE TEXT EXPERT
class CICDTextModel(nn.Module):
    def __init__(self, num_classes=10): 
        super().__init__()
        self.bert = AutoModel.from_pretrained("microsoft/codebert-base")
        for param in list(self.bert.parameters())[:-4]:
            param.requires_grad = False
            
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )
    
    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = bert_output.last_hidden_state[:, 0, :]
        return self.classifier(cls_embedding)

# 2. LOAD BOTH EXPERTS
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️ Booting up Two Experts on {device}...")

tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")

# Load CodeBERT Expert
codebert_expert = CICDTextModel()
try:
    codebert_expert.load_state_dict(torch.load('codebert_text_expert.pth', map_location=device))
    codebert_expert.to(device)
    codebert_expert.eval()
    print("✅ CodeBERT Text Expert loaded successfully!")
except Exception as e:
    print(f"❌ Error loading PyTorch model: {e}")
    exit()

# Load XGBoost Expert
xgb_expert = xgb.XGBClassifier()
try:
    xgb_expert.load_model('xgboost_expert.json')
    print("✅ XGBoost Tabular Expert loaded successfully!\n")
except Exception as e:
    print(f"❌ Error loading XGBoost model: {e}")
    exit()

# 3. SIMULATE A LIVE JENKINS CRASH
live_tabular_data = np.array([[
    1.2, -0.5, 0.0, 3.5, 4.2, 0.0,
    0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
]], dtype=np.float32)

target_classes = [
    'Build Failure', 'Configuration Error', 'Dependency Error', 'Deployment Failure', 
    'Network Error', 'Permission Error', 'Resource Exhaustion', 'Security Scan Failure', 
    'Test Failure', 'Timeout'
]

messages = [
    "FATAL ERROR: Ineffective mark-compacts near heap limit Allocation failed - JavaScript heap out of memory",
    "ModuleNotFoundError: No module named 'numpy'",
    "TimeoutError: Connection to AWS RDS database timed out after 30000ms",
    "403 Forbidden: Missing IAM role permissions for s3:PutObject"
]

os.environ["TRANSFORMERS_VERBOSITY"] = "error"

for msg in messages:
    print(f"\n📥 INCOMING WEBHOOK TEXT: '{msg}'")
    
    # 1. Get XGBoost Probabilities
    xgb_probs = xgb_expert.predict_proba(live_tabular_data)[0]
    
    # 2. Get CodeBERT Probabilities
    inputs = tokenizer(msg, return_tensors="pt", max_length=512, truncation=True, padding='max_length').to(device)
    with torch.no_grad():
        logits = codebert_expert(inputs['input_ids'], inputs['attention_mask'])
        bert_probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()[0]
        
    # 3. The Judge (Blending the NumPy Arrays)
    final_probabilities = (0.30 * xgb_probs) + (0.70 * bert_probs)
    
    # 4. Extract the Winner
    predicted_index = np.argmax(final_probabilities)
    confidence = final_probabilities[predicted_index] * 100
    
    print("=========================================")
    print(f"🚨 DIAGNOSIS:  {target_classes[predicted_index]}")
    print(f"📊 CONFIDENCE: {confidence:.2f}%")
    print("=========================================")
    
    # 5. The Threshold Gate
    if confidence >= 85.0:
        print(f"✅ High Confidence (>85%). Approved for Agentic Auto-Fix.")
    else:
        print(f"⚠️ Low Confidence. Routing to Human Engineer.")
