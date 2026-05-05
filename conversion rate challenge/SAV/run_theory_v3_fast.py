import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

print("="*80)
print("🧬 THEORY V3 FAST : GENERATION ONLY")
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

X = df_train[features]
y = df_train['converted']
X_test = df_test[features]

# Model Config
PARAMS = {
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

US_CODE = 3
PAGES_MIN = 8
PAGES_MAX = 11
AMENDMENT_TH = 0.30
GLOBAL_TH = 0.385

def apply_amendment(X_input, probs):
    preds = (probs >= GLOBAL_TH).astype(int)
    
    country_idx = features.index('country')
    pages_idx = features.index('total_pages_visited')
    
    if isinstance(X_input, pd.DataFrame):
        countries = X_input['country'].values
        pages = X_input['total_pages_visited'].values
    else:
        countries = X_input[:, country_idx]
        pages = X_input[:, pages_idx]
        
    mask_target = (countries == US_CODE) & (pages >= PAGES_MIN) & (pages <= PAGES_MAX)
    amendment_hits = (mask_target) & (probs >= AMENDMENT_TH)
    
    # Only upgrade 0->1
    preds[amendment_hits] = 1
    return preds, amendment_hits.sum()

print("Training Full Model...")
model = xgb.XGBClassifier(**PARAMS)
model.fit(X, y)

probs = model.predict_proba(X_test)[:, 1]
preds_v3, n_hits = apply_amendment(X_test, probs)

submission = pd.DataFrame({'converted': preds_v3})
submission.to_csv('submission_THEORY_V3_AMENDMENT.csv', index=False)
print(f"✅ Submission Saved: submission_THEORY_V3_AMENDMENT.csv")
print(f"   Total Conversions : {submission['converted'].sum()}")
print(f"   Amendement Triggered : {n_hits} times (Potential upgrades)")
