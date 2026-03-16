import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder

print("="*80)
print("🤝 HYBRID ENSEMBLE : V1 (Recall) + REFINED (Precision)")
print("="*80)

SEED = 42
N_FOLDS = 10

# 1. Configurations
PARAMS_V1 = {
    'learning_rate': 0.01,
    'max_depth': 6,
    'n_estimators': 500,
    'subsample': 0.8,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'n_jobs': -1,
    'random_state': SEED,
    'monotone_constraints': '(0, 0, 0, -1, 1)',
    'tree_method': 'hist'
}

PARAMS_REFINED = {
    'learning_rate': 0.01,
    'max_depth': 5,
    'n_estimators': 800,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 50, # The "Precision" constraint
    'reg_lambda': 1.0,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'n_jobs': -1,
    'random_state': SEED,
    'monotone_constraints': '(0, 0, 0, -1, 1)',
    'tree_method': 'hist'
}

# 2. Load & Preprocess
try:
    df = pd.read_csv("conversion_data_train.csv")
except:
    df = pd.read_csv("/home/phil/projetdatascience/conversion rate challenge/conversion_data_train.csv")

features = ['country', 'source', 'new_user', 'age', 'total_pages_visited']
for col in ['country', 'source']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

X = df[features]
y = df['converted']

# 3. CV Loop
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

f1_v1_scores = []
f1_ref_scores = []
f1_hybrid_scores = []

print(f"Starting {N_FOLDS}-Fold CV...")

for i, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
    
    # Train V1
    m1 = xgb.XGBClassifier(**PARAMS_V1)
    m1.fit(X_tr, y_tr)
    p1 = m1.predict_proba(X_val)[:, 1]
    
    # Train Refined
    m2 = xgb.XGBClassifier(**PARAMS_REFINED)
    m2.fit(X_tr, y_tr)
    p2 = m2.predict_proba(X_val)[:, 1]
    
    # Hybrid Prob
    p_hybrid = (p1 + p2) / 2.0
    
    # Evaluate Max F1 for each
    def get_max_f1(probs, true_y):
        best_f, best_t = 0, 0.5
        for th in np.arange(0.3, 0.6, 0.01):
            s = f1_score(true_y, (probs >= th).astype(int))
            if s > best_f: best_f, best_t = s, th
        return best_f
    
    s1 = get_max_f1(p1, y_val)
    s2 = get_max_f1(p2, y_val)
    s_hyb = get_max_f1(p_hybrid, y_val)
    
    f1_v1_scores.append(s1)
    f1_ref_scores.append(s2)
    f1_hybrid_scores.append(s_hyb)
    
    print(f"   Fold {i+1}: V1={s1:.4f} | Ref={s2:.4f} | HYBRID={s_hyb:.4f} ({s_hyb-max(s1,s2):+.4f})")

print("-" * 60)
print(f"📊 RESULTATS FINAUX ({N_FOLDS} Folds)")
print(f"   V1 Mean      : {np.mean(f1_v1_scores):.5f}")
print(f"   Refined Mean : {np.mean(f1_ref_scores):.5f}")
print(f"   HYBRID Mean  : {np.mean(f1_hybrid_scores):.5f}")
print(f"   Best Model   : {'HYBRID 🏆' if np.mean(f1_hybrid_scores) > max(np.mean(f1_v1_scores), np.mean(f1_ref_scores)) else 'Single'}")
