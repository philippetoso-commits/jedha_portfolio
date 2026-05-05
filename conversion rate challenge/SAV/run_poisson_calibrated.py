import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold

print("="*80)
print("🧪 GENERATION : POISSON V3 + ISOTONIC CALIBRATION")
print("="*80)

SEED = 42
N_FOLDS = 5

# 1. Config (V3 Optimized)
BEST_PARAMS = {
    'learning_rate': 0.03, 'max_depth': 6, 'n_estimators': 600,
    'objective': 'reg:tweedie', 'tweedie_variance_power': 1.5,
    'n_jobs': -1, 'random_state': SEED
}
features = ['country', 'source', 'new_user', 'age_bin', 'pages_bin', 'pages_age_ratio']

# 2. Load Data
df_train = pd.read_csv('conversion_data_train.csv')
df_test = pd.read_csv('conversion_data_test.csv')

def preprocessing(df):
    df_c = df.copy()
    df_c['age_bin'] = pd.cut(df_c['age'], bins=[0, 18, 25, 30, 35, 40, 45, 50, 60, 100], labels=False).fillna(-1).astype(int)
    df_c['pages_bin'] = pd.cut(df_c['total_pages_visited'], bins=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 25, 100], labels=False).fillna(-1).astype(int)
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

# 3. Fit Calibrator via CV (Strict OOF)
print("Calibration (Training OOF)...")
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
oof_preds = np.zeros(len(df_train))

for fold, (train_idx, val_idx) in enumerate(skf.split(df_train, df_train['converted'])):
    X_tr = df_train.iloc[train_idx]
    y_tr = df_train['converted'].iloc[train_idx]
    
    # Aggregation
    tr_agg = X_tr.groupby(features)['converted'].agg(['mean', 'count']).reset_index()
    tr_agg.rename(columns={'mean': 'conversion_rate', 'count': 'weight'}, inplace=True)
    
    # Train
    model = xgb.XGBRegressor(**BEST_PARAMS)
    model.fit(tr_agg[features], tr_agg['conversion_rate'], sample_weight=tr_agg['weight'])
    
    # Predict Raw
    oof_preds[val_idx] = model.predict(df_train.iloc[val_idx][features])

# Fit Isotonic on OOF
iso = IsotonicRegression(out_of_bounds='clip')
iso.fit(oof_preds, df_train['converted'])

# 4. Final Training (Full Data)
print("Entraînement Final (Full Train)...")
agg_full = df_train.groupby(features)['converted'].agg(['mean', 'count']).reset_index()
agg_full.rename(columns={'mean': 'conversion_rate', 'count': 'weight'}, inplace=True)

model_full = xgb.XGBRegressor(**BEST_PARAMS)
model_full.fit(agg_full[features], agg_full['conversion_rate'], sample_weight=agg_full['weight'])

# 5. Prediction & Calibration
print("Prédiction & Calibration Test...")
test_raw_preds = model_full.predict(df_test[features])
test_calib_preds = iso.transform(test_raw_preds)

# Threshold (from evolution notebook)
BEST_TH_CALIB = 0.335
test_bin_preds = (test_calib_preds >= BEST_TH_CALIB).astype(int)

# 6. Export
sub_name = 'submission_POISSON_CALIBRATED.csv'
sub = pd.DataFrame({'converted': test_bin_preds})
sub.to_csv(sub_name, index=False)

print(f"✅ Soumission : {sub_name}")
print(f"   Total Conversions : {test_bin_preds.sum()}")
print("-" * 60)
