import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score

print("="*80)
print("🛡️ GENERATION : CONSTRAINED OVERFIT (Safe & Deep)")
print("="*80)

SEED = 42

# 1. Configs
# A. Robust Config (For Safety Net)
ROBUST_PARAMS = {
    'learning_rate': 0.03, 'max_depth': 6, 'n_estimators': 600,
    'objective': 'reg:tweedie', 'tweedie_variance_power': 1.5,
    'n_jobs': -1, 'random_state': SEED,
    'monotone_constraints': '(0, 0, 0, -1, 1, 1)' # Enforce structure even here
}

# B. Overfit Config (Deep but Constrained)
OVERFIT_PARAMS = {
    'learning_rate': 0.05,        
    'max_depth': 10,              # Deep
    'min_child_weight': 1,        # Micro-structures
    'subsample': 1.0, 'colsample_bytree': 1.0,
    'n_estimators': 1200,         
    'objective': 'reg:tweedie', 
    'tweedie_variance_power': 1.5,
    'n_jobs': -1, 'random_state': SEED,
    'monotone_constraints': '(0, 0, 0, -1, 1, 1)' # Crucial: Force Logic
}
# Feature order: ['country', 'source', 'new_user', 'age_bin', 'pages_bin', 'pages_age_ratio']
# Constraints:  (0,        0,        0,         -1(Age),   +1(Pages),  +1(Ratio))

# 2. Data
df_train = pd.read_csv('conversion_data_train.csv')
df_test = pd.read_csv('conversion_data_test.csv')

def preprocessing(df):
    df_c = df.copy()
    df_c['age_bin'] = pd.cut(df_c['age'], bins=[0, 18, 25, 30, 35, 40, 45, 50, 60, 100], labels=False).fillna(-1).astype(int)
    # Granular pages for Overfit precision
    df_c['pages_bin'] = pd.cut(df_c['total_pages_visited'], bins=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 25, 100], labels=False).fillna(-1).astype(int)
    df_c['pages_age_ratio'] = df_c['total_pages_visited'] / (df_c['age'] + 1)
    return df_c

df_train = preprocessing(df_train)
df_test = preprocessing(df_test)

cat_cols = ['country', 'source']
for col in cat_cols:
    le = LabelEncoder()
    full = pd.concat([df_train[col], df_test[col]])
    le.fit(full)
    df_train[col] = le.transform(df_train[col])
    df_test[col] = le.transform(df_test[col])

features = ['country', 'source', 'new_user', 'age_bin', 'pages_bin', 'pages_age_ratio']

# 3. Aggregation & Asymmetric Weights
print("Agrégation & Pondération Asymétrique...")
agg_train = df_train.groupby(features)['converted'].agg(['mean', 'count']).reset_index()
agg_train.rename(columns={'mean': 'conversion_rate', 'count': 'weight'}, inplace=True)

# Asymmetric Logic:
# If profile is "Strong" (Rate > 0.5 and Pages > 10), we BOOST the weight.
# This tells the model: "It is CRITICAL to fit this profile correctly."
# Penalty for False Negative becomes huge.
boost_mask = (agg_train['conversion_rate'] > 0.5) & (agg_train['pages_bin'] >= 10) # Bin 10 is roughly 10 pages depending on cut
agg_train.loc[boost_mask, 'weight'] *= 5 

print(f"  Profils Boostés : {boost_mask.sum()}")

# 4. Training (Double Model)
print("Entraînement Modèle ROBUSTE (Structure)...")
model_robust = xgb.XGBRegressor(**ROBUST_PARAMS)
model_robust.fit(agg_train[features], agg_train['conversion_rate'], sample_weight=agg_train['weight']) # Normal weights? Or boosted? Let's use boosted for safety too.

print("Entraînement Modèle OVERFIT (Deep)...")
model_overfit = xgb.XGBRegressor(**OVERFIT_PARAMS)
model_overfit.fit(agg_train[features], agg_train['conversion_rate'], sample_weight=agg_train['weight'])

# 5. Prediction (Train) - For Threshold Opt
print("Optimisation Seuil (Train)...")
# Map profiles
train_mapped = pd.merge(df_train, agg_train[features], on=features, how='left') # Just to be sure we have features aligned
# Predict Overfit on Train
prob_train_overfit = model_overfit.predict(df_train[features])
prob_train_robust = model_robust.predict(df_train[features])

# Safety Blend Logic (Train)
# If Robust says "Very Likely" (>0.8), we trust it minimaly to be above threshold
# Directional Shrinkage:
# P_final = P_overfit
# if P_robust > 0.6 (Solid) and P_overfit < P_robust: P_final = P_robust
# This effectively fills the "holes" created by overfit.
prob_train_final = prob_train_overfit.copy()
safety_mask = (prob_train_robust > 0.60) & (prob_train_overfit < prob_train_robust)
prob_train_final[safety_mask] = prob_train_robust[safety_mask]

# Optimized Threshold on Blended Prob
best_f1 = 0
best_th = 0
for th in np.arange(0.3, 0.6, 0.005):
    f1 = f1_score(df_train['converted'], (prob_train_final >= th).astype(int))
    if f1 > best_f1:
        best_f1 = f1
        best_th = th

print(f"🔥 TRAIN F1 SCORE (Blended) : {best_f1:.5f}")
print(f"   Seuil Optimal           : {best_th:.3f}")

# 6. Prediction (Test)
print("Prédiction Test & Blending...")
prob_test_overfit = model_overfit.predict(df_test[features])
prob_test_robust = model_robust.predict(df_test[features])

# Appy Blend
prob_test_final = prob_test_overfit.copy()
safety_mask_test = (prob_test_robust > 0.60) & (prob_test_overfit < prob_test_robust)
prob_test_final[safety_mask_test] = prob_test_robust[safety_mask_test]

print(f"  🔧 Corrections de Sécurité appliquées : {safety_mask_test.sum()}")

test_bin = (prob_test_final >= best_th).astype(int)

# 7. Export
sub_name = 'submission_POISSON_CONSTRAINED.csv'
sub = pd.DataFrame({'converted': test_bin})
sub.to_csv(sub_name, index=False)

print(f"✅ Soumission générée : {sub_name}")
print(f"   Total Conversions : {test_bin.sum()}")
print("-" * 60)
