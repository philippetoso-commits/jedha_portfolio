import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, roc_auc_score

print("="*80)
print("🧬 THEORY V3 : AMENDEMENT TACTIQUE (US 8-11 PAGES / TH=0.30)")
print("="*80)

SEED = 42
N_FOLDS = 10

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

# Model Config (Standard V1 Base)
PARAMS = {
    'learning_rate': 0.01,
    'max_depth': 6,
    'n_estimators': 500,
    'subsample': 0.8,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'n_jobs': -1,
    'random_state': SEED,
    'monotone_constraints': '(0, 0, 0, -1, 1)'
}

# --- DEFINITIONS ---
# US = 3 (checked via previous script assumption, likely alphabetical sorting)
# Assuming alphabetical: China(0), Germany(1), UK(2), US(3).
US_CODE = 3
PAGES_MIN = 8
PAGES_MAX = 11
AMENDMENT_TH = 0.30

def apply_amendment(X_input, probs, global_th):
    """
    Applies Global Threshold first, then overrides for US 8-11 pages with Amendment Threshold.
    """
    # 1. Base Prediction
    preds = (probs >= global_th).astype(int)
    
    # 2. Identify Target Segment
    # We need to access columns by name. X_input might be numpy or DataFrame.
    if isinstance(X_input, pd.DataFrame):
        countries = X_input['country'].values
        pages = X_input['total_pages_visited'].values
    else:
        # Assuming column order matches 'features' list
        # features = ['country', 'source', 'new_user', 'age', 'total_pages_visited']
        country_idx = features.index('country')
        pages_idx = features.index('total_pages_visited')
        countries = X_input[:, country_idx]
        pages = X_input[:, pages_idx]
        
    mask_target = (countries == US_CODE) & (pages >= PAGES_MIN) & (pages <= PAGES_MAX)
    
    # 3. Apply Amendment
    # Override: If in target and prob >= 0.30, set to 1
    # Note: If prob was >= global_th (e.g. 0.38), it's already 1.
    # We only care about cases where 0.30 <= prob < global_th.
    
    amendment_hits = (mask_target) & (probs >= AMENDMENT_TH)
    preds[amendment_hits] = 1
    
    return preds

# 1. 10-Fold CV Validation
print(f"Starting {N_FOLDS}-Fold CV with Amendment Logic...")
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
f1_scores = []
base_f1_scores = [] # For V1 vs V3 comparison inside folds

for i, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    model = xgb.XGBClassifier(**PARAMS)
    model.fit(X_tr, y_tr)
    
    probs = model.predict_proba(X_val)[:, 1]
    
    # Optimize Base Threshold on Validation (Simulator behavior)
    # usually done on Train, but here we want "Best Possible V1" vs "Best Possible V3"
    # Actually, fairness requires optimizing on Train part of fold or fixing it.
    # Let's fix a "standard good" threshold to compare apple to apple, e.g. 0.385 (found in V1).
    GLOBAL_TH = 0.385
    
    # Base V1
    preds_v1 = (probs >= GLOBAL_TH).astype(int)
    score_v1 = f1_score(y_val, preds_v1)
    base_f1_scores.append(score_v1)
    
    # V3 with Amendment
    preds_v3 = apply_amendment(X_val, probs, GLOBAL_TH)
    score_v3 = f1_score(y_val, preds_v3)
    f1_scores.append(score_v3)
    
    # print(f"   Fold {i+1}: V1={score_v1:.4f} -> V3={score_v3:.4f}")

mean_v1 = np.mean(base_f1_scores)
mean_v3 = np.mean(f1_scores)

print("-" * 60)
print(f"📊 RESULTATS CV (10 Folds)")
print(f"   V1 Mean F1 : {mean_v1:.5f}")
print(f"   V3 Mean F1 : {mean_v3:.5f}")
print(f"   Delta      : {mean_v3 - mean_v1:+.5f}")

if mean_v3 > mean_v1:
    print("👉 SUCCESS: L'amendement améliore le score !")
else:
    print("👉 WARNING: L'amendement dégrade le score (Precision drop > Recall gain).")

# 2. Generation Submission
print("-" * 60)
print("Training Full Model V3...")
model_full = xgb.XGBClassifier(**PARAMS)
model_full.fit(X, y)

probs_test = model_full.predict_proba(X_test)[:, 1]

# Apply same logic
GLOBAL_TH_FINAL = 0.385 # Using the robust V1 threshold
preds_test_v3 = apply_amendment(X_test, probs_test, GLOBAL_TH_FINAL)

submission = pd.DataFrame({'converted': preds_test_v3})
submission.to_csv('submission_THEORY_V3_AMENDMENT.csv', index=False)
print(f"✅ Submission Saved: submission_THEORY_V3_AMENDMENT.csv")
print(f"   Total Conversions : {submission['converted'].sum()}")

# Quick check of impact on Test
preds_test_v1 = (probs_test >= GLOBAL_TH_FINAL).astype(int)
diff = np.sum(preds_test_v3) - np.sum(preds_test_v1)
print(f"   Impact sur Test : +{diff} conversions ajoutées par l'amendement.")
