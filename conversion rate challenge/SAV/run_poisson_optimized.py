import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

print("="*80)
print("🏆 POISSON SUPREMACY OPTIMIZED (V3 | lr=0.03 d=6 n=600)")
print("="*80)

# 1. Config Gagnante
# V3 (Ratio) | lr=0.03 d=6 n=600 | TH=0.3840
BEST_PARAMS = {
    'learning_rate': 0.03,
    'max_depth': 6,
    'n_estimators': 600,
    'objective': 'reg:tweedie',
    'tweedie_variance_power': 1.5,
    'n_jobs': -1,
    'random_state': 42
}
BEST_TH = 0.3840

# 2. Data Load & Prep
print("Chargement et Préparation...")
df_train = pd.read_csv('conversion_data_train.csv')
df_test = pd.read_csv('conversion_data_test.csv')

def preprocessing(df):
    df_c = df.copy()
    # Bins
    df_c['age_bin'] = pd.cut(df_c['age'], bins=[0, 18, 25, 30, 35, 40, 45, 50, 60, 100], labels=False).fillna(-1).astype(int)
    df_c['pages_bin'] = pd.cut(df_c['total_pages_visited'], bins=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 25, 100], labels=False).fillna(-1).astype(int)
    
    # V3 Specific: Ratio
    df_c['pages_age_ratio'] = df_c['total_pages_visited'] / (df_c['age'] + 1)
    
    return df_c

df_train = preprocessing(df_train)
df_test = preprocessing(df_test)

# Encodage
cat_cols = ['country', 'source']
for col in cat_cols:
    le = LabelEncoder()
    full = pd.concat([df_train[col], df_test[col]])
    le.fit(full)
    df_train[col] = le.transform(df_train[col])
    df_test[col] = le.transform(df_test[col])

# Features utilisées
features = ['country', 'source', 'new_user', 'age_bin', 'pages_bin', 'pages_age_ratio']

# 3. Aggregation (Training Set)
print("Agrégation par Profils...")
agg_train = df_train.groupby(features)['converted'].agg(['mean', 'count']).reset_index()
agg_train.rename(columns={'mean': 'conversion_rate', 'count': 'weight'}, inplace=True)

print(f"  Nombre de profils uniques (Train) : {len(agg_train)}")

# 4. Training
print(f"Entraînement avec params: {BEST_PARAMS}...")
model = xgb.XGBRegressor(**BEST_PARAMS)
model.fit(
    agg_train[features],
    agg_train['conversion_rate'],
    sample_weight=agg_train['weight']
)

# 5. Prediction
print("Prédiction sur Test...")
# On prédit directement sur les lignes du test (qui ont les features du profil)
test_preds_proba = model.predict(df_test[features])

# Application Seuil
test_preds_bin = (test_preds_proba >= BEST_TH).astype(int)

# 6. Export
sub_name = 'submission_POISSON_OPTIMIZED_V3.csv'
sub = pd.DataFrame({'converted': test_preds_bin})
sub.to_csv(sub_name, index=False)

print(f"✅ Soumission générée : {sub_name}")
print(f"   Total Conversions : {test_preds_bin.sum()}")
print(f"   Seuil utilisé     : {BEST_TH}")
print("-" * 60)
