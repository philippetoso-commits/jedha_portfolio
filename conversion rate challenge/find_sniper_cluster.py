import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

print("="*80)
print("🎯 OPERATION SNIPER : RECHERCHE DE CLUSTERS CACHÉS")
print("="*80)

SEED = 42
N_FOLDS = 5

# Configs (Constrained Model)
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
    # Granular pages for precision
    df_c['pages_bin'] = pd.cut(df_c['total_pages_visited'], bins=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 25, 100], labels=False).fillna(-1).astype(int)
    df_c['pages_age_ratio'] = df_c['total_pages_visited'] / (df_c['age'] + 1)
    return df_c

df = preprocessing(df)

cat_cols = ['country', 'source']
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# OOF Generation
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
oof_preds = np.zeros(len(df))

print("Génération OOF (Patience)...")
for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['converted'])):
    X_tr = df.iloc[train_idx].copy()
    X_val = df.iloc[val_idx].copy()
    
    # Aggregation
    agg_tr = X_tr.groupby(features)['converted'].agg(['mean', 'count']).reset_index()
    agg_tr.rename(columns={'mean': 'conversion_rate', 'count': 'weight'}, inplace=True)
    boost_mask = (agg_tr['conversion_rate'] > 0.5) & (agg_tr['pages_bin'] >= 10)
    agg_tr.loc[boost_mask, 'weight'] *= 5
    
    # Train
    model_robust = xgb.XGBRegressor(**ROBUST_PARAMS)
    model_robust.fit(agg_tr[features], agg_tr['conversion_rate'], sample_weight=agg_tr['weight'])
    model_overfit = xgb.XGBRegressor(**OVERFIT_PARAMS)
    model_overfit.fit(agg_tr[features], agg_tr['conversion_rate'], sample_weight=agg_tr['weight'])
    
    # Predict Val
    p_val_rob = model_robust.predict(X_val[features])
    p_val_over = model_overfit.predict(X_val[features])
    p_val_final = p_val_over.copy()
    mask_val = (p_val_rob > 0.60) & (p_val_over < p_val_rob)
    p_val_final[mask_val] = p_val_rob[mask_val]
    
    oof_preds[val_idx] = p_val_final

# Determine Threshold
best_f1 = 0
best_th = 0
for th in np.arange(0.3, 0.6, 0.005):
    f1 = f1_score(df['converted'], (oof_preds >= th).astype(int))
    if f1 > best_f1: best_f1, best_th = f1, th

print(f"OOF F1: {best_f1:.5f} | Threshold Global: {best_th:.3f}")

# ANALYSIS OF REJECTED
print("\n🕵️ Analyse des Rejetés...")
df['prob'] = oof_preds
rejected = df[df['prob'] < best_th].copy()

# Define Clusters (Country + Source + NewUser + AgeBin)
# We ignore Pages because clear MONOTONE signal usually comes from Pages.
# So we want a cluster defined by DEMOGRAPHICS, inside which HIGH PAGES are ignored.
cluster_cols = ['country', 'source', 'new_user', 'age_bin']
grouped = rejected.groupby(cluster_cols)

print(f"Nombre de clusters rejetés : {len(grouped)}")

candidates = []

for name, group in grouped:
    if len(group) < 50: continue # Noise
    
    # Check Tail (Top 5% pages or Max Pages)
    max_pages = group['total_pages_visited'].max()
    if max_pages < 8: continue # No high signal
    
    # Tail logic: Pages >= Max - 1 (The very top)
    tail = group[group['total_pages_visited'] >= max_pages - 1]
    
    if len(tail) < 5: continue # Too small
    
    tail_conv_rate = tail['converted'].mean()
    tail_volume = len(tail)
    
    # CRITERION: Tail Conv Rate > 0.50 (Should be accepted)
    if tail_conv_rate > 0.60:
        candidates.append({
            'cluster': name,
            'cluster_size': len(group),
            'tail_cut': max_pages - 1,
            'tail_size': tail_volume,
            'tail_rate': tail_conv_rate,
            'mean_prob': group['prob'].mean()
        })

print("\n🏆 CANDIDATS SNIPER (Queue Droite Explosive)")
print("-" * 80)
if len(candidates) == 0:
    print("❌ AUCUN CANDIDAT VIABLE IDENTIFIÉ.")
else:
    candidates_df = pd.DataFrame(candidates).sort_values('tail_rate', ascending=False)
    print(candidates_df.to_string())
    print("-" * 80)
    
    # Gain Estimation
    total_gain = 0
    total_new_fp = 0
    for cand in candidates:
        tp = cand['tail_size'] * cand['tail_rate']
        fp = cand['tail_size'] * (1 - cand['tail_rate'])
        total_gain += tp
        total_new_fp += fp
        
    print(f"Potentiel Gain TP : +{total_gain:.1f}")
    print(f"Potentiel Coût FP : +{total_new_fp:.1f}")
