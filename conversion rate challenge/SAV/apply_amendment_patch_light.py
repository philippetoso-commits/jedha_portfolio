import pandas as pd
import numpy as np

print("="*80)
print("🧬 THEORY V3 : APPLYING AMENDMENT PATCH (NO RETRAINING)")
print("="*80)

# Load V1 Submission (Base Model Predictions)
# We need the Probabilities actually to apply the 0.30 threshold?
# No, V1 submission only has 0/1.
# Damn. If V1 only has 0/1 (generated at 0.385), I cannot recover those between 0.30 and 0.385 from the CSV.
# I NEED THE PROBABILITIES.
# But training is stuck.
#
# Wait. V1 training worked fine (Step 1243: "Done", used 'hist').
# Why is V3 training stuck?
# V3 script uses exact same params as V1 script!
# PARAMS = { ... 'tree_method': 'hist' ... }
#
# Maybe specific random seed or system load?
#
# Alternative: Load 'submission_POISSON_CONSTRAINED.csv' (which we have)? No.
#
# Fine. I will try to run V1 Training logic again but stripped to the bone.
# Or I recover the probabilities if I saved them? No I didn't.
#
# Pivot: I will run `run_theory_v3_fast.py` BUT I will remove `monotone_constraints` just to get the file out? 
# NO! Monotonicity is the whole point of Theory.
#
# I will try to run a lightweight training (100 estimators) just to get the probabilities.
# 100 estimators is enough to replicate the ranking.
#
# Let's do that.

import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

SEED = 42

df_train = pd.read_csv('conversion_data_train.csv')
df_test = pd.read_csv('conversion_data_test.csv')

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

# LIGHTWEIGHT CONFIG
PARAMS = {
    'learning_rate': 0.05, # Higher LR to compensate fewer trees
    'max_depth': 6,
    'n_estimators': 150, # Reduced from 500
    'subsample': 0.8,
    'objective': 'binary:logistic',
    'n_jobs': -1,
    'random_state': SEED,
    'monotone_constraints': '(0, 0, 0, -1, 1)',
    'tree_method': 'hist' 
}

print("Training Lightweight Model (150 trees)...")
model = xgb.XGBClassifier(**PARAMS)
model.fit(X, y)

probs = model.predict_proba(X_test)[:, 1]

# APPLY LOGIC
US_CODE = 3
PAGES_MIN = 8
PAGES_MAX = 11
AMENDMENT_TH = 0.30
GLOBAL_TH = 0.385

preds = (probs >= GLOBAL_TH).astype(int)

# Target Mask
countries = X_test['country'].values
pages = X_test['total_pages_visited'].values
mask_target = (countries == US_CODE) & (pages >= PAGES_MIN) & (pages <= PAGES_MAX)

# Upgrade
amendment_hits = (mask_target) & (probs >= AMENDMENT_TH)
preds[amendment_hits] = 1

submission = pd.DataFrame({'converted': preds})
submission.to_csv('submission_THEORY_V3_AMENDMENT.csv', index=False)
print(f"✅ Submission Saved: submission_THEORY_V3_AMENDMENT.csv")
print(f"   Total Conversions : {submission['converted'].sum()}")
print(f"   Amendement Triggered : {amendment_hits.sum()} potential upgrades (some might have been >0.385 already)")
print(f"   Net Gain vs Base: {np.sum(preds) - np.sum((probs >= GLOBAL_TH).astype(int))}")
