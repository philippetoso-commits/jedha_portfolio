import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score, brier_score_loss
from sklearn.preprocessing import LabelEncoder

print("="*80)
print("🎓 PROFESSOR MODEL : 5-FOLD CV EVALUATION")
print("   (Target Encoding applied INSIDE folds to prevent leakage)")
print("="*80)

SEED = 42
N_FOLDS = 5

# Configs (Constrained Params)
PARAMS = {
    'learning_rate': 0.03, 'max_depth': 6, 'n_estimators': 600,
    'objective': 'reg:tweedie', 'tweedie_variance_power': 1.5,
    'n_jobs': -1, 'random_state': SEED,
    'monotone_constraints': '(0, 0, 0, -1, 1, 1, 1)' # Adjusted for new feature at end
}

# Load Data
df = pd.read_csv('conversion_data_train.csv')

def preprocessing(df):
    df_c = df.copy()
    df_c['age_bin'] = pd.cut(df_c['age'], bins=[0, 18, 25, 30, 35, 40, 45, 50, 60, 100], labels=False).fillna(-1).astype(int)
    df_c['pages_bin'] = pd.cut(df_c['total_pages_visited'], bins=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 25, 100], labels=False).fillna(-1).astype(int)
    df_c['pages_age_ratio'] = df_c['total_pages_visited'] / (df_c['age'] + 1)
    
    # PROFESSOR FEATURE (Raw String)
    df_c['combinaison'] = (
        df_c['country'].astype(str) + "_" + 
        df_c['age'].astype(str) + "_" + 
        df_c['new_user'].astype(str) + "_" + 
        df_c['source'].astype(str) + "_" + 
        df_c['total_pages_visited'].astype(str)
    )
    return df_c

df = preprocessing(df)

# Label Encoding for standard cats
cat_cols = ['country', 'source']
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

features_base = ['country', 'source', 'new_user', 'age_bin', 'pages_bin', 'pages_age_ratio']
# 'TE_combinaison' will be added dynamically

# CV Loop
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
f1_scores = []
auc_scores = []
best_thresholds = []

print(f"Starting {N_FOLDS}-Fold CV...")

for i, (train_idx, val_idx) in enumerate(skf.split(df, df['converted'])):
    X_tr = df.iloc[train_idx].copy()
    X_val = df.iloc[val_idx].copy()
    y_tr, y_val = X_tr['converted'], X_val['converted']
    
    # --- TARGET ENCODING INSIDE FOLD ---
    # Calc means on Train
    means = X_tr.groupby('combinaison')['converted'].mean()
    global_mean = y_tr.mean()
    
    # Map to Train (with smoothing/noise could be better, but simple map for now)
    # Ideally we use K-Fold inside Train for Train TE, but standard TE is often just:
    # Train = LOO or K-Fold. Here let's do simple leave-one-out approximation or just map.
    # Simple map on Train overfits massively.
    # Let's use K-Fold TE on Train for the Train set.
    
    # Inner K-Fold for Train Encoding
    skf_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED+i)
    X_tr['TE_combinaison'] = np.nan
    for tr_in, val_in in skf_inner.split(X_tr, y_tr):
        X_inner_tr = X_tr.iloc[tr_in]
        X_inner_val = X_tr.iloc[val_in]
        inner_means = X_inner_tr.groupby('combinaison')['converted'].mean()
        X_tr.loc[X_tr.index[val_in], 'TE_combinaison'] = X_inner_val['combinaison'].map(inner_means)
    
    X_tr['TE_combinaison'] = X_tr['TE_combinaison'].fillna(global_mean)

    # Map to Val (Standard Map)
    X_val['TE_combinaison'] = X_val['combinaison'].map(means)
    X_val['TE_combinaison'] = X_val['TE_combinaison'].fillna(global_mean)
    
    # Prepare Features
    features_final = features_base + ['TE_combinaison']
    
    # Train
    model = xgb.XGBRegressor(**PARAMS)
    model.fit(X_tr[features_final], y_tr)
    
    # Predict
    dval = X_val[features_final]
    probs = model.predict(dval)
    
    # Evaluate AUC
    auc = roc_auc_score(y_val, probs)
    auc_scores.append(auc)
    
    # Optimize Threshold
    best_f1_fold = 0
    best_th_fold = 0
    for th in np.arange(0.3, 0.6, 0.01):
        f1 = f1_score(y_val, (probs >= th).astype(int))
        if f1 > best_f1_fold: 
            best_f1_fold = f1
            best_th_fold = th
            
    f1_scores.append(best_f1_fold)
    best_thresholds.append(best_th_fold)
    
    print(f"   Fold {i+1}: F1={best_f1_fold:.5f} (Th={best_th_fold:.2f}) | AUC={auc:.5f}")

print("-" * 60)
print(f"📊 RESULTATS MOYENS ({N_FOLDS} Folds)")
print(f"   AVG F1 Score : {np.mean(f1_scores):.5f} (± {np.std(f1_scores):.5f})")
print(f"   AVG AUC      : {np.mean(auc_scores):.5f}")
print(f"   AVG Threshold: {np.mean(best_thresholds):.3f}")

# Compare with Baseline (Hardcoded reference from previous runs)
print("-" * 60)
print("⚖️ BENCHMARK")
print(f"   Base Poisson Constrained (10-Fold) : ~0.7670")
print(f"   Audit Recovery (Estimated)         : ~0.7670+")
if np.mean(f1_scores) > 0.7670:
    print("   ✅ VERDICT : L'Hypothèse Professeur AMÉLIORE le score CV.")
else:
    print("   ❌ VERDICT : L'Hypothèse Professeur DÉGRADE (ou n'améliore pas) le score CV.")
