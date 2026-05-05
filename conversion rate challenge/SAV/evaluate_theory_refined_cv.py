import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder

print("="*80)
print("🧬 SINGLE MODEL THEORY : REFINED 10-FOLD CV")
print("="*80)

SEED = 42
N_FOLDS = 10

PARAMS = {
    'learning_rate': 0.01,
    'max_depth': 5,
    'n_estimators': 800,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 50,
    'reg_lambda': 1.0,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'n_jobs': -1,
    'random_state': SEED,
    'monotone_constraints': '(0, 0, 0, -1, 1)'
}

df = pd.read_csv('conversion_data_train.csv')
features = ['country', 'source', 'new_user', 'age', 'total_pages_visited']

for col in ['country', 'source']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

X = df[features]
y = df['converted']

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
f1_scores = []
thresholds = []

print(f"Starting {N_FOLDS}-Fold CV...")

for i, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    model = xgb.XGBClassifier(**PARAMS)
    model.fit(X_tr, y_tr)
    
    probs = model.predict_proba(X_val)[:, 1]
    
    b_f1, b_th = 0, 0.5
    for th in np.arange(0.3, 0.6, 0.01):
        f = f1_score(y_val, (probs >= th).astype(int))
        if f > b_f1: b_f1, b_th = f, th
        
    f1_scores.append(b_f1)
    thresholds.append(b_th)
    
    print(f"   Fold {i+1}: F1={b_f1:.5f} (Th={b_th:.2f})")

print("-" * 60)
print(f"📊 RESULTATS REFINED ({N_FOLDS} Folds)")
print(f"   AVG F1 Score : {np.mean(f1_scores):.5f} (± {np.std(f1_scores):.5f})")
print(f"   AVG Threshold: {np.mean(thresholds):.5f}")
