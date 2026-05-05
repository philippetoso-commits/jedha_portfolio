import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, roc_auc_score

print("="*80)
print("🧬 THEORY V2 : WEIGHTED SENSITIVITY (BOOST ZONE 8-11 PAGES)")
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

# --- CRITICAL ZONE DEFINITION ---
# US (3), UK (2) usually (check mapping, but let's use string lookup to be safe if we had it, 
# here we used fit encoded. 
# Let's re-map to be sure: China=0, Germany=1, UK=2, US=3 usually alphabetic.
# Let's double check standard LabelEncoder behavior: Sorted Alphabetical.
# China=0, Germany=1, UK=2, US=3.
critical_countries = [2, 3] # UK, US
critical_pages = [8, 9, 10, 11, 12] # The "Transition" Zone

# Assign Weights
# Default = 1.0
# If Converted=1 AND is in Critical Zone -> Weight = 2.0 (To force learning)
weights = np.ones(len(df_train))

mask_critical = (
    (df_train['converted'] == 1) & 
    (df_train['country'].isin(critical_countries)) &
    (df_train['total_pages_visited'].isin(critical_pages))
)

weight_boost = 1.5 # Moderate boost. Too high = Overfitting noise.
weights[mask_critical] = weight_boost

print(f"Applying Weight Boost x{weight_boost} to {mask_critical.sum()} positive training samples.")

X = df_train[features]
y = df_train['converted']
X_test = df_test[features]

# Model Config (Same as V1)
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

# 1. Evaluate CV (To see if F1 holds)
print(f"Starting {N_FOLDS}-Fold CV...")
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
f1_scores = []
thresholds = []

for i, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
    w_tr = weights[train_idx]
    
    model = xgb.XGBClassifier(**PARAMS)
    model.fit(X_tr, y_tr, sample_weight=w_tr)
    
    probs = model.predict_proba(X_val)[:, 1]
    
    b_f1, b_th = 0, 0.5
    for th in np.arange(0.3, 0.6, 0.01):
        f = f1_score(y_val, (probs >= th).astype(int))
        if f > b_f1: b_f1, b_th = f, th
        
    f1_scores.append(b_f1)
    thresholds.append(b_th)
    
    # print(f"   Fold {i+1}: F1={b_f1:.5f}")

print(f"📊 AVG F1 Score (V2) : {np.mean(f1_scores):.5f} (± {np.std(f1_scores):.5f})")
print(f"   AVG Threshold     : {np.mean(thresholds):.3f}")

# 2. Generate Submission using Full Train
print("-" * 60)
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
