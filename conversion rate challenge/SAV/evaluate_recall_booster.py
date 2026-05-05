import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score

print("="*80)
print("🚀 EVALUATION RECALL BOOSTER (Gray Zone Bias)")
print("="*80)

SEED = 42
N_FOLDS = 10

scores_baseline = []
scores_boosted = []
conversions_baseline = []
conversions_boosted = []

# Configs (Constrained Model Base)
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
    # Re-use granular bins for model input, but use raw 'total_pages_visited' for Booster Logic
    df_c['pages_bin'] = pd.cut(df_c['total_pages_visited'], bins=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 25, 100], labels=False).fillna(-1).astype(int)
    df_c['pages_age_ratio'] = df_c['total_pages_visited'] / (df_c['age'] + 1)
    return df_c

df = preprocessing(df)

cat_cols = ['country', 'source']
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# CV Loop
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

pass_idx = 0
for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['converted'])):
    pass_idx += 1
    # Split
    X_tr = df.iloc[train_idx].copy()
    X_val = df.iloc[val_idx].copy()
    y_tr = df['converted'].iloc[train_idx]
    y_val = df['converted'].iloc[val_idx]
    
    # 1. Aggregation & Weights
    agg_tr = X_tr.groupby(features)['converted'].agg(['mean', 'count']).reset_index()
    agg_tr.rename(columns={'mean': 'conversion_rate', 'count': 'weight'}, inplace=True)
    boost_mask = (agg_tr['conversion_rate'] > 0.5) & (agg_tr['pages_bin'] >= 10)
    agg_tr.loc[boost_mask, 'weight'] *= 5
    
    # 2. Train
    model_robust = xgb.XGBRegressor(**ROBUST_PARAMS)
    model_robust.fit(agg_tr[features], agg_tr['conversion_rate'], sample_weight=agg_tr['weight'])
    
    model_overfit = xgb.XGBRegressor(**OVERFIT_PARAMS)
    model_overfit.fit(agg_tr[features], agg_tr['conversion_rate'], sample_weight=agg_tr['weight'])
    
    # 3. Predict Train & Opt Threshold
    p_tr_rob = model_robust.predict(X_tr[features])
    p_tr_over = model_overfit.predict(X_tr[features])
    p_tr_final = p_tr_over.copy()
    mask_tr = (p_tr_rob > 0.60) & (p_tr_over < p_tr_rob)
    p_tr_final[mask_tr] = p_tr_rob[mask_tr]
    
    best_th = 0.5
    best_f1_tr = 0
    for th in np.arange(0.3, 0.6, 0.005):
        score = f1_score(y_tr, (p_tr_final >= th).astype(int))
        if score > best_f1_tr: best_f1_tr, best_th = score, th

    # 4. Predict Val (Baseline)
    p_val_rob = model_robust.predict(X_val[features])
    p_val_over = model_overfit.predict(X_val[features])
    p_val_final = p_val_over.copy()
    mask_val = (p_val_rob > 0.60) & (p_val_over < p_val_rob)
    p_val_final[mask_val] = p_val_rob[mask_val]
    
    base_preds = (p_val_final >= best_th).astype(int)
    f1_base = f1_score(y_val, base_preds)
    scores_baseline.append(f1_base)
    conversions_baseline.append(base_preds.sum())
    
    # 5. RECALL BOOSTER LOGIC
    # We apply a bias to probabilities where p < threshold but p > 0.30 (Gray Zone)
    p_boosted = p_val_final.copy()
    
    # Need access to raw features for logic
    rows_val = X_val
    
    # Definition of Bias
    # Tier 1: Pages >= 12 (Very Safe) -> +0.10
    mask_tier1 = (rows_val['total_pages_visited'] >= 12)
    p_boosted[mask_tier1] += 0.10
    
    # Tier 2: Pages >= 9 & Age >= 25 (Safe) -> +0.05
    mask_tier2 = (rows_val['total_pages_visited'] >= 9) & (rows_val['age'] >= 25)
    p_boosted[mask_tier2] += 0.05

    # Tier 3: Pages >= 7 (Soft) -> +0.02
    mask_tier3 = (rows_val['total_pages_visited'] >= 7)
    p_boosted[mask_tier3] += 0.02
    
    # Re-Threshold
    boosted_preds = (p_boosted >= best_th).astype(int)
    f1_boost = f1_score(y_val, boosted_preds)
    scores_boosted.append(f1_boost)
    conversions_boosted.append(boosted_preds.sum())
    
    print(f"  Fold {pass_idx} | Base F1: {f1_base:.5f} -> Boost F1: {f1_boost:.5f} | Delta: {f1_boost - f1_base:+.5f} | Conv: {base_preds.sum()}->{boosted_preds.sum()}")

print("-" * 60)
mean_base = np.mean(scores_baseline)
mean_boost = np.mean(scores_boosted)
print(f"🏆 MEAN BASELINE F1 : {mean_base:.5f}")
print(f"🚀 MEAN BOOSTED F1  : {mean_boost:.5f}")
print(f"📈 GAIN F1          : {mean_boost - mean_base:+.5f}")
print(f"📊 AVG CONV GAIN    : {np.mean(conversions_boosted) - np.mean(conversions_baseline):.1f}")
print("-" * 60)
