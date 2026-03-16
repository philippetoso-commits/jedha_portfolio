import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score

print("="*80)
print("📊 EVALUATION 10-FOLD CV : POISSON CONSTRAINED")
print("="*80)

SEED = 42
N_FOLDS = 10

# Configs
ROBUST_PARAMS = {
    'learning_rate': 0.03, 'max_depth': 6, 'n_estimators': 600,
    'objective': 'reg:tweedie', 'tweedie_variance_power': 1.5,
    'n_jobs': -1, 'random_state': SEED,
    'monotone_constraints': '(0, 0, 0, -1, 1, 1)'
}
OVERFIT_PARAMS = {
    'learning_rate': 0.05, 'max_depth': 10, 'min_child_weight': 1,
    'subsample': 1.0, 'colsample_bytree': 1.0, 'n_estimators': 1200,
    'objective': 'reg:tweedie', 'tweedie_variance_power': 1.5,
    'n_jobs': -1, 'random_state': SEED,
    'monotone_constraints': '(0, 0, 0, -1, 1, 1)'
}
features = ['country', 'source', 'new_user', 'age_bin', 'pages_bin', 'pages_age_ratio']

# Load Data
df = pd.read_csv('conversion_data_train.csv')

def preprocessing(df):
    df_c = df.copy()
    df_c['age_bin'] = pd.cut(df_c['age'], bins=[0, 18, 25, 30, 35, 40, 45, 50, 60, 100], labels=False).fillna(-1).astype(int)
    df_c['pages_bin'] = pd.cut(df_c['total_pages_visited'], bins=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 25, 100], labels=False).fillna(-1).astype(int)
    df_c['pages_age_ratio'] = df_c['total_pages_visited'] / (df_c['age'] + 1)
    return df_c

df = preprocessing(df)

# Label Encoding Global (Safe for cat vars with few levels)
cat_cols = ['country', 'source']
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# CV Loop
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
scores_f1 = []
scores_auc = []
thresholds = []

print(f"Starting {N_FOLDS}-Fold CV...")

for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['converted'])):
    # Split
    X_tr = df.iloc[train_idx].copy()
    X_val = df.iloc[val_idx].copy()
    y_tr = df['converted'].iloc[train_idx]
    y_val = df['converted'].iloc[val_idx]
    
    # 1. Aggregation (Train only)
    agg_tr = X_tr.groupby(features)['converted'].agg(['mean', 'count']).reset_index()
    agg_tr.rename(columns={'mean': 'conversion_rate', 'count': 'weight'}, inplace=True)
    
    # 2. Asymmetric Weights (Train only)
    boost_mask = (agg_tr['conversion_rate'] > 0.5) & (agg_tr['pages_bin'] >= 10)
    agg_tr.loc[boost_mask, 'weight'] *= 5
    
    # 3. Training
    model_robust = xgb.XGBRegressor(**ROBUST_PARAMS)
    model_robust.fit(agg_tr[features], agg_tr['conversion_rate'], sample_weight=agg_tr['weight'])
    
    model_overfit = xgb.XGBRegressor(**OVERFIT_PARAMS)
    model_overfit.fit(agg_tr[features], agg_tr['conversion_rate'], sample_weight=agg_tr['weight'])
    
    # 4. Predict on TRAIN (for Threshold Opt)
    # Map features to raw train rows
    # Efficiency: Unnecessary to merge, just predict X_tr
    p_tr_rob = model_robust.predict(X_tr[features])
    p_tr_over = model_overfit.predict(X_tr[features])
    
    # Blend Train
    p_tr_final = p_tr_over.copy()
    mask_tr = (p_tr_rob > 0.60) & (p_tr_over < p_tr_rob)
    p_tr_final[mask_tr] = p_tr_rob[mask_tr]
    
    # Optimize Threshold
    best_th_fold = 0.5
    best_f1_fold_tr = 0
    for th in np.arange(0.3, 0.6, 0.005):
        score = f1_score(y_tr, (p_tr_final >= th).astype(int))
        if score > best_f1_fold_tr:
            best_f1_fold_tr = score
            best_th_fold = th
    thresholds.append(best_th_fold)
    
    # 5. Predict on VAL
    p_val_rob = model_robust.predict(X_val[features])
    p_val_over = model_overfit.predict(X_val[features])
    
    # Blend Val
    p_val_final = p_val_over.copy()
    mask_val = (p_val_rob > 0.60) & (p_val_over < p_val_rob)
    p_val_final[mask_val] = p_val_rob[mask_val]
    
    # Score
    val_preds = (p_val_final >= best_th_fold).astype(int)
    f1 = f1_score(y_val, val_preds)
    auc = roc_auc_score(y_val, p_val_final)
    
    scores_f1.append(f1)
    scores_auc.append(auc)
    
    print(f"  Fold {fold+1}/{N_FOLDS} | F1: {f1:.5f} | AUC: {auc:.5f} | TH: {best_th_fold:.3f}")

print("-" * 60)
print(f"🏆 AVERAGE F1 SCORE  : {np.mean(scores_f1):.5f} (+/- {np.std(scores_f1):.5f})")
print(f"📈 AVERAGE ROC AUC   : {np.mean(scores_auc):.5f}")
print(f"⚙️ AVERAGE THRESHOLD : {np.mean(thresholds):.3f}")
print("-" * 60)
