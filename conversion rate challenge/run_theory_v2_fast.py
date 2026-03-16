import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score

print("="*80)
print("🧬 THEORY V2 FAST : WEIGHTED PREDICTION ONLY")
print("="*80)

SEED = 42

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

# --- CRITICAL ZONE DEFINITION ---
# UK=2, US=3
critical_countries = [2, 3] 
critical_pages = [8, 9, 10, 11]

# Assign Weights
weights = np.ones(len(df_train))
mask_critical = (
    (df_train['converted'] == 1) & 
    (df_train['country'].isin(critical_countries)) &
    (df_train['total_pages_visited'].isin(critical_pages))
)
weight_boost = 1.5
weights[mask_critical] = weight_boost

print(f"Applying Weight Boost x{weight_boost} to {mask_critical.sum()} positive training samples.")

X = df_train[features]
y = df_train['converted']
X_test = df_test[features]

# Model Config
PARAMS = {
    'learning_rate': 0.01,
    'max_depth': 6,
    'n_estimators': 300,
    'subsample': 0.8,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'n_jobs': -1,
    'random_state': SEED,
    'monotone_constraints': '(0, 0, 0, -1, 1)',
    'tree_method': 'hist'
}

print("Training Full Model V2...")
model_v2 = xgb.XGBClassifier(**PARAMS)
model_v2.fit(X, y, sample_weight=weights)

# Optimize Th on Train
probs_train = model_v2.predict_proba(X)[:, 1]
best_f1 = 0
best_th = 0.5
for th in np.arange(0.3, 0.6, 0.005):
    f1 = f1_score(y, (probs_train >= th).astype(int))
    if f1 > best_f1: best_f1, best_th = f1, th

print(f"✅ TRAIN F1 SCORE : {best_f1:.5f}")
print(f"   Seuil Optimal  : {best_th:.3f}")

preds_test = (model_v2.predict_proba(X_test)[:, 1] >= best_th).astype(int)

submission = pd.DataFrame({'converted': preds_test})
submission.to_csv('submission_THEORY_V2_SENSITIVE.csv', index=False)
print("✅ Submission Saved: submission_THEORY_V2_SENSITIVE.csv")
print(f"   Total Conversions : {submission['converted'].sum()}")
