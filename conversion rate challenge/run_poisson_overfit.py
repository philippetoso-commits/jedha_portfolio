import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score

print("="*80)
print("🧨 GENERATION : POISSON OVERFIT (Target Train F1 >= 0.78)")
print("="*80)

SEED = 42

# 1. Config "Overfit"
OVERFIT_PARAMS = {
    'learning_rate': 0.05,        # Slightly higher to fit faster
    'max_depth': 10,              # Deep trees
    'min_child_weight': 1,        # Learn from small buckets
    'subsample': 1.0,             # No randomness
    'colsample_bytree': 1.0,      # Use all features
    'n_estimators': 1200,         # Many trees
    'objective': 'reg:tweedie', 
    'tweedie_variance_power': 1.5,
    'n_jobs': -1, 
    'random_state': SEED
}
print(f"Params: {OVERFIT_PARAMS}")

# 2. Data Load
df_train = pd.read_csv('conversion_data_train.csv')
df_test = pd.read_csv('conversion_data_test.csv')

def preprocessing(df):
    df_c = df.copy()
    df_c['age_bin'] = pd.cut(df_c['age'], bins=[0, 18, 25, 30, 35, 40, 45, 50, 60, 100], labels=False).fillna(-1).astype(int)
    # Granular pages bin for overfit
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

# 3. Aggregation & Structural Memory
features_base = ['country', 'source', 'new_user', 'age_bin', 'pages_bin', 'pages_age_ratio']

print("Agrégation et Création Mémoire Structurelle...")
agg_train = df_train.groupby(features_base)['converted'].agg(['mean', 'count']).reset_index()
agg_train.rename(columns={'mean': 'conversion_rate', 'count': 'weight'}, inplace=True)

# Add Memory Features (Train only, model learns to trust big weights)
agg_train['log_weight'] = np.log1p(agg_train['weight'])
agg_train['weight_rank'] = agg_train['weight'].rank(pct=True)

print(f"  Profils Train : {len(agg_train)}")

# 4. Training (Full Train)
features_train = features_base + ['log_weight', 'weight_rank']

print("Entraînement Muscle...")
model = xgb.XGBRegressor(**OVERFIT_PARAMS)
# On entraine sur les profils agrégés
model.fit(
    agg_train[features_train],
    agg_train['conversion_rate'],
    sample_weight=agg_train['weight']
)

# 5. Threshold Optimization (ON FULL TRAIN PREDICTION)
# Pour prédire sur le train bruts, on doit mapper les features mémoire
# Mais les lignes individuelles n'ont pas de 'weight'.
# ASTUCE : On doit mapper les propriétés du PROFIL à chaque ligne.
print("Calcul des prédictions Train (Mapping Profils)...")
# Merge train brut avec ses infos de profil (weight, log_weight, etc.)
df_train_mapped = pd.merge(df_train, agg_train[features_base + ['log_weight', 'weight_rank']], on=features_base, how='left')

train_preds = model.predict(df_train_mapped[features_train])

# Find Best Threshold
best_f1 = 0
best_th = 0
for th in np.arange(0.3, 0.6, 0.005):
    f1 = f1_score(df_train['converted'], (train_preds >= th).astype(int))
    if f1 > best_f1:
        best_f1 = f1
        best_th = th

print(f"🔥 TRAIN F1 SCORE : {best_f1:.5f}")
print(f"🎯 Classification Train - Seuil Optimal : {best_th:.3f}")

if best_f1 < 0.78:
    print("⚠️ Attention : Target 0.78 non atteinte. Essayez d'augmenter depth ou n_estimators.")
else:
    print("✅ Target 0.78 atteinte !")

# 6. Prediction Test
print("Prédiction Test...")
# On doit d'abord calculer les features mémoire pour le Test
# HYPOTHÈSE FORTE : On utilise le poids du PROFIL DANS LE TEST ? 
# NON ! L'Overfit structurel suppose que la confiance dépend de la taille du profil DANS L'APPRENTISSAGE.
# Mais si un profil n'existe pas en train ?
# Approche : On map les stats du TRAIN sur le TEST.
# Si un profil test n'existait pas en train, il a weight=0 (ou nan -> 0).

# Calcul des stats profils train (déjà fait dans agg_train)
# On merge agg_train sur df_test
df_test_mapped = pd.merge(df_test, agg_train[features_base + ['log_weight', 'weight_rank']], on=features_base, how='left')

# Fillna pour les nouveaux profils (Inconnus au bataillon)
df_test_mapped['log_weight'] = df_test_mapped['log_weight'].fillna(0)
df_test_mapped['weight_rank'] = df_test_mapped['weight_rank'].fillna(0)

test_preds = model.predict(df_test_mapped[features_train])
test_bin = (test_preds >= best_th).astype(int)

# 7. Export
sub_name = 'submission_POISSON_OVERFIT.csv'
sub = pd.DataFrame({'converted': test_bin})
sub.to_csv(sub_name, index=False)

print(f"✅ Soumission générée : {sub_name}")
print(f"   Total Conversions : {test_bin.sum()}")
print("-" * 60)
