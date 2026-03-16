import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score

print("="*80)
print("🧠 SELF-TRAINING : 5-FOLD CV EVALUATION")
print("="*80)

SEED = 42
N_FOLDS = 5

# 1. Load Data
df_train = pd.read_csv('conversion_data_train.csv')
df_test = pd.read_csv('conversion_data_test.csv') # Real Test Set for Pseudo-Labeling

features = ['country', 'source', 'new_user', 'age', 'total_pages_visited']
for col in ['country', 'source']:
    le = LabelEncoder()
    full = pd.concat([df_train[col], df_test[col]])
    le.fit(full)
    df_train[col] = le.transform(df_train[col])
    df_test[col] = le.transform(df_test[col])

X = df_train[features]
y = df_train['converted']
X_test_real = df_test[features] # Unlabeled

# Using Fast Params for CV
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

print(f"Starting {N_FOLDS}-Fold CV...")
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
f1_teacher = []
f1_student = []

for i, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    
    # 1. Train Teacher
    teacher = xgb.XGBClassifier(**PARAMS)
    teacher.fit(X_tr, y_tr)
    
    # Evaluate Teacher
    probs_val = teacher.predict_proba(X_val)[:, 1]
    # Simple threshold optimization for teacher
    best_f1_t = 0
    best_th_t = 0.5
    for th in np.arange(0.3, 0.6, 0.05):
        s = f1_score(y_val, (probs_val >= th).astype(int))
        if s > best_f1_t: best_f1_t, best_th_t = s, th
    f1_teacher.append(best_f1_t)
    
    # 2. Pseudo-Label Real Test Set
    probs_test = teacher.predict_proba(X_test_real)[:, 1]
    
    TH_POS = 0.95
    TH_NEG = 0.05
    
    mask_pos = (probs_test >= TH_POS)
    mask_neg = (probs_test <= TH_NEG)
    
    X_pseudo_pos = X_test_real[mask_pos]
    y_pseudo_pos = pd.Series([1]*len(X_pseudo_pos))
    
    X_pseudo_neg = X_test_real[mask_neg]
    y_pseudo_neg = pd.Series([0]*len(X_pseudo_neg))
    
    # 3. Augment Train
    X_aug = pd.concat([X_tr, X_pseudo_pos, X_pseudo_neg])
    y_aug = pd.concat([y_tr, y_pseudo_pos, y_pseudo_neg])
    
    # 4. Train Student
    student = xgb.XGBClassifier(**PARAMS)
    student.fit(X_aug, y_aug)
    
    # Evaluate Student on SAME Validation Set
    probs_val_s = student.predict_proba(X_val)[:, 1]
    
    # Optimize threshold for Student
    best_f1_s = 0
    for th in np.arange(0.3, 0.6, 0.05):
        s = f1_score(y_val, (probs_val_s >= th).astype(int))
        if s > best_f1_s: best_f1_s = s
    f1_student.append(best_f1_s)
    
    print(f"   Fold {i+1}: Teacher={best_f1_t:.4f} -> Student={best_f1_s:.4f} (Delta: {best_f1_s - best_f1_t:+.4f}) "
          f"[Added {len(X_pseudo_pos)} POS, {len(X_pseudo_neg)} NEG]")

m_t = np.mean(f1_teacher)
m_s = np.mean(f1_student)

print("-" * 60)
print(f"📊 RESULTATS SELF-TRAINING CV")
print(f"   Teacher Mean F1 : {m_t:.5f}")
print(f"   Student Mean F1 : {m_s:.5f}")
print(f"   Impact Net      : {m_s - m_t:+.5f}")
