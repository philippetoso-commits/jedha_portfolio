import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score

print("="*80)
print("🧬 SINGLE MODEL THEORY : REFINED (CUSTOM PARAMS)")
print("="*80)

SEED = 42

# User Custom Params
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

# Load
df_train = pd.read_csv('conversion_data_train.csv')
df_test = pd.read_csv('conversion_data_test.csv')

# Preprocessing
features = ['country', 'source', 'new_user', 'age', 'total_pages_visited']

for col in ['country', 'source']:
    le = LabelEncoder()
    full = pd.concat([df_train[col], df_test[col]])
    le.fit(full)
    df_train[col] = le.transform(df_train[col])
    df_test[col] = le.transform(df_test[col])

X = df_train[features]
y = df_train['converted']
X_test = df_test[features]

print(f"Training Full Model (N={len(df_train)})...")
model = xgb.XGBClassifier(**PARAMS)
model.fit(X, y)

# Threshold Optimization on Train
probs_train = model.predict_proba(X)[:, 1]
best_f1 = 0
best_th = 0.5

print("Optimizing Threshold...")
for th in np.arange(0.3, 0.6, 0.005):
    f1 = f1_score(y, (probs_train >= th).astype(int))
    if f1 > best_f1:
        best_f1 = f1
        best_th = th
        
print(f"✅ TRAIN F1 SCORE : {best_f1:.5f}")
print(f"   Seuil Optimal  : {best_th:.3f}")

# Prediction
probs_test = model.predict_proba(X_test)[:, 1]
preds_test = (probs_test >= best_th).astype(int)

submission = pd.DataFrame({'converted': preds_test})
submission.to_csv('submission_THEORY_REFINED.csv', index=False)
print("✅ Submission Saved: submission_THEORY_REFINED.csv")
print(f"   Total Conversions : {submission['converted'].sum()}")
