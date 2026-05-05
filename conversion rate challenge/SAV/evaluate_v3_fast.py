import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score

print("="*80)
print("🧬 THEORY V3 : FAST EVALUATION (F1 SCORE ESTIMATION)")
print("="*80)

SEED = 42
N_FOLDS = 5 # 5 folds for speed

# Load
df = pd.read_csv('conversion_data_train.csv')

features = ['country', 'source', 'new_user', 'age', 'total_pages_visited']
for col in ['country', 'source']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

X = df[features]
y = df['converted']

# LIGHTWEIGHT CONFIG (FAST & PROXY FOR FULL MODEL)
PARAMS = {
    'learning_rate': 0.05,
    'max_depth': 6,
    'n_estimators': 150,
    'subsample': 0.8,
    'objective': 'binary:logistic',
    'n_jobs': -1,
    'random_state': SEED,
    'monotone_constraints': '(0, 0, 0, -1, 1)',
    'tree_method': 'hist'
}

US_CODE = 3
PAGES_MIN = 8
PAGES_MAX = 11
AMENDMENT_TH = 0.30
GLOBAL_TH = 0.385 # Fixed Base Threshold

def apply_amendment_cv(X_val, probs):
    preds = (probs >= GLOBAL_TH).astype(int)
    
    # Identify Target Segment
    # X_val is a DataFrame subset
    mask_target = (X_val['country'] == US_CODE) & \
                  (X_val['total_pages_visited'] >= PAGES_MIN) & \
                  (X_val['total_pages_visited'] <= PAGES_MAX)
                  
    hits = (mask_target) & (probs >= AMENDMENT_TH)
    preds[hits] = 1
    return preds

print(f"Starting {N_FOLDS}-Fold CV...")
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
f1_v1 = []
f1_v3 = []

for i, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    
    model = xgb.XGBClassifier(**PARAMS)
    model.fit(X_tr, y_tr)
    
    probs = model.predict_proba(X_val)[:, 1]
    
    # V1 Score
    p_v1 = (probs >= GLOBAL_TH).astype(int)
    s_v1 = f1_score(y_val, p_v1)
    f1_v1.append(s_v1)
    
    # V3 Score
    p_v3 = apply_amendment_cv(X_val, probs)
    s_v3 = f1_score(y_val, p_v3)
    f1_v3.append(s_v3)
    
    print(f"   Fold {i+1}: V1={s_v1:.4f} -> V3={s_v3:.4f} (Delta: {s_v3 - s_v1:+.4f})")

m_v1 = np.mean(f1_v1)
m_v3 = np.mean(f1_v3)

print("-" * 60)
print(f"📊 RESULTATS FINAUX (ESTIMATION)")
print(f"   V1 Mean F1 : {m_v1:.5f}")
print(f"   V3 Mean F1 : {m_v3:.5f}")
print(f"   Impact Net : {m_v3 - m_v1:+.5f}")
