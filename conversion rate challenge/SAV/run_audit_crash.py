import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score

print("="*80)
print("🚑 AUDIT CRASH : STRESS TEST SANS LABELS")
print("="*80)

SEED = 42

# --- RECONSTRUCTION DU MODÈLE (Identique à run_poisson_constrained.py) ---
print("1. Reconstruction du Modèle Constrained...")

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

df_train = pd.read_csv('conversion_data_train.csv')
df_test = pd.read_csv('conversion_data_test.csv')

def preprocessing(df):
    df_c = df.copy()
    df_c['age_bin'] = pd.cut(df_c['age'], bins=[0, 18, 25, 30, 35, 40, 45, 50, 60, 100], labels=False).fillna(-1).astype(int)
    # Granular pages for Overfit precision match
    df_c['pages_bin'] = pd.cut(df_c['total_pages_visited'], bins=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 25, 100], labels=False).fillna(-1).astype(int)
    df_c['pages_age_ratio'] = df_c['total_pages_visited'] / (df_c['age'] + 1)
    return df_c

df_train = preprocessing(df_train)
df_test = preprocessing(df_test)

cat_cols = ['country', 'source']
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    full = pd.concat([df_train[col], df_test[col]])
    le.fit(full)
    df_train[col] = le.transform(df_train[col])
    df_test[col] = le.transform(df_test[col])
    label_encoders[col] = le

features = ['country', 'source', 'new_user', 'age_bin', 'pages_bin', 'pages_age_ratio']

# Aggregation & Weights
agg_train = df_train.groupby(features)['converted'].agg(['mean', 'count']).reset_index()
agg_train.rename(columns={'mean': 'conversion_rate', 'count': 'weight'}, inplace=True)
boost_mask = (agg_train['conversion_rate'] > 0.5) & (agg_train['pages_bin'] >= 10)
agg_train.loc[boost_mask, 'weight'] *= 5 

# Training
model_robust = xgb.XGBRegressor(**ROBUST_PARAMS)
model_robust.fit(agg_train[features], agg_train['conversion_rate'], sample_weight=agg_train['weight'])

model_overfit = xgb.XGBRegressor(**OVERFIT_PARAMS)
model_overfit.fit(agg_train[features], agg_train['conversion_rate'], sample_weight=agg_train['weight'])

# Threshold Opt (Train)
prob_train_overfit = model_overfit.predict(df_train[features])
prob_train_robust = model_robust.predict(df_train[features])
prob_train_final = prob_train_overfit.copy()
safety_mask = (prob_train_robust > 0.60) & (prob_train_overfit < prob_train_robust)
prob_train_final[safety_mask] = prob_train_robust[safety_mask]

best_f1 = 0
best_th = 0
for th in np.arange(0.3, 0.6, 0.005):
    f1 = f1_score(df_train['converted'], (prob_train_final >= th).astype(int))
    if f1 > best_f1: best_f1, best_th = f1, th
print(f"   Modèle Reconstruit. Seuil Op: {best_th:.3f}")


# --- PREDICTION FUNCTION (Wrapper) ---
def get_scores(df_in):
    # Ensure features exist
    if 'pages_age_ratio' not in df_in.columns:
         df_in = preprocessing(df_in) # simplistic re-prep if needed
    
    # Predict
    p_over = model_overfit.predict(df_in[features])
    p_rob = model_robust.predict(df_in[features])
    p_final = p_over.copy()
    mask = (p_rob > 0.60) & (p_over < p_rob)
    p_final[mask] = p_rob[mask]
    return p_final

# Get Baseline Test Scores
scores_test = get_scores(df_test)

# --- TEST 1: STABILITÉ FRONTIÈRE ---
print("\n🧪 Test 1: Stabilité de la Frontière")
margin = np.abs(scores_test - best_th)
density_near_border = (margin < 0.02).mean()
print(f"   Marge Moyenne : {margin.mean():.4f}")
print(f"   Densité Critique (<0.02 du seuil) : {density_near_border:.2%}")
if density_near_border < 0.05:
    print("   ✅ VERDICT : Frontière Nette (Peu d'hésitation)")
elif density_near_border < 0.10:
    print("   ⚠️ VERDICT : Frontière Modérée")
else:
    print("   ❌ VERDICT : Frontière Floue (Instabilité)")

# --- TEST 2: MONOTONIE LOCALE (Contrefactuels) ---
print("\n🧪 Test 2: Monotonie Locale (Contrefactuels)")
# Case A: +1 Page
df_test_plus_page = df_test.copy()
df_test_plus_page['total_pages_visited'] += 1
df_test_plus_page = preprocessing(df_test_plus_page) # Update bins/ratios
scores_plus_page = get_scores(df_test_plus_page)
violations_page = (scores_plus_page < scores_test - 1e-5).sum() # Tolerance epsilon

# Case B: -1 Age (Younger)
df_test_minus_age = df_test.copy()
df_test_minus_age['age'] = (df_test_minus_age['age'] - 1).clip(lower=17)
df_test_minus_age = preprocessing(df_test_minus_age)
scores_minus_age = get_scores(df_test_minus_age)
violations_age = (scores_minus_age < scores_test - 1e-5).sum()

print(f"   Violations (+1 Page) : {violations_page} / {len(df_test)}")
print(f"   Violations (-1 Age)  : {violations_age} / {len(df_test)}")

if violations_page + violations_age == 0:
    print("   ✅ VERDICT : Monotonie Parfaite")
else:
    print("   ❌ VERDICT : Violations de Monotonie détectées !")

# --- TEST 3: SYMÉTRIE TRAIN/TEST ---
print("\n🧪 Test 3: Symétrie Train / Test")
stats_train = pd.Series(prob_train_final).describe(percentiles=[0.1, 0.5, 0.9])
stats_test = pd.Series(scores_test).describe(percentiles=[0.1, 0.5, 0.9])

print(f"   Mean: Train={stats_train['mean']:.3f} vs Test={stats_test['mean']:.3f}")
print(f"   P50 : Train={stats_train['50%']:.3f} vs Test={stats_test['50%']:.3f}")
print(f"   P90 : Train={stats_train['90%']:.3f} vs Test={stats_test['90%']:.3f}")

diff_mean = abs(stats_train['mean'] - stats_test['mean'])
if diff_mean < 0.01:
    print("   ✅ VERDICT : Distributions Alignées")
else:
    print("   ⚠️ VERDICT : Décalage Distribution (Drift potentiel)")

# --- TEST 4: SENSIBILITÉ (Perturbation) ---
print("\n🧪 Test 4: Sensibilité (Perturbation Random)")
# Permute Country
df_perm = df_test.copy()
df_perm['country'] = np.random.permutation(df_perm['country'])
df_perm = preprocessing(df_perm) # Just in case
scores_perm = get_scores(df_perm)
mean_diff = np.abs(scores_test - scores_perm).mean()
print(f"   Impact Permutation Pays (Mean Abs Delta) : {mean_diff:.4f}")

if mean_diff < 0.1: # Threshold arbitrary
    print("   ✅ VERDICT : Modèle Robuste (Pas uniquement dépendant du Pays)")
else:
    print("   ℹ️ Note : Modèle sensible au Pays (Normal si c'est une feature clé)")

# --- TEST 5: AUDIT ZONE GRISE ---
print("\n🧪 Test 5: Audit Zone Grise")
mask_gray = (scores_test >= best_th - 0.02) & (scores_test <= best_th + 0.02)
gray_zone = df_test[mask_gray]
print(f"   Population Zone Grise : {len(gray_zone)}")

if len(gray_zone) > 0:
    std_age = gray_zone['age'].std()
    std_pages = gray_zone['total_pages_visited'].std()
    print(f"   Diversité Age (Std)   : {std_age:.2f}")
    print(f"   Diversité Pages (Std) : {std_pages:.2f}")
    
    if std_age < 1.0 and std_pages < 1.0:
        print("   ❌ ALERT : La frontière est un Cluster Pathologique Unique (Manque de Robustesse ?)")
    else:
        print("   ✅ VERDICT : Zone Grise Diversifiée (Frontière Saine)")
else:
    print("   ✅ Zone vide (Séparation parfaite)")

print("\n" + "="*80)
print("🏁 CONCLUSION GLOBALE")
if violations_page == 0 and violations_age == 0 and density_near_border < 0.1:
    print("   🟢 FEU VERT : LE MODÈLE EST STRUCTURELLEMENT SAIN.")
else:
    print("   🔴 FEU ROUGE : RISQUES D'INSTABILITÉ IDENTIFIÉS.")
