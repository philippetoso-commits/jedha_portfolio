import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

print("="*80)
print("🎓 PROFESSOR MODEL : INTERACTION TARGET ENCODING")
print("="*80)

SEED = 42
N_FOLDS_TE = 5 # Folds for Target Encoding

# Configs (Same as Constrained)
PARAMS = {
    'learning_rate': 0.03, 'max_depth': 6, 'n_estimators': 600,
    'objective': 'reg:tweedie', 'tweedie_variance_power': 1.5,
    'n_jobs': -1, 'random_state': SEED,
    'monotone_constraints': '(0, 0, 0, -1, 1, 1, 1)' # Added 1 constraint for new feature (positive impact expected)
}

# Load Data
df_train = pd.read_csv('conversion_data_train.csv')
df_test = pd.read_csv('conversion_data_test.csv')

def preprocessing(df):
    df_c = df.copy()
    df_c['age_bin'] = pd.cut(df_c['age'], bins=[0, 18, 25, 30, 35, 40, 45, 50, 60, 100], labels=False).fillna(-1).astype(int)
    df_c['pages_bin'] = pd.cut(df_c['total_pages_visited'], bins=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 25, 100], labels=False).fillna(-1).astype(int)
    df_c['pages_age_ratio'] = df_c['total_pages_visited'] / (df_c['age'] + 1)
    
    # PROFESSOR FEATURE
    df_c['combinaison'] = (
        df_c['country'].astype(str) + "_" + 
        df_c['age'].astype(str) + "_" + 
        df_c['new_user'].astype(str) + "_" + 
        df_c['source'].astype(str) + "_" + 
        df_c['total_pages_visited'].astype(str)
    )
    return df_c

df_train = preprocessing(df_train)
df_test = preprocessing(df_test)

# --- TARGET ENCODING (Regularized) ---
print("Applying Target Encoding on 'combinaison'...")

# 1. Train Set (K-Fold to prevent leakage)
skf = StratifiedKFold(n_splits=N_FOLDS_TE, shuffle=True, random_state=SEED)
df_train['TE_combinaison'] = np.nan

for train_idx, val_idx in skf.split(df_train, df_train['converted']):
    X_tr, X_val = df_train.iloc[train_idx], df_train.iloc[val_idx]
    
    # Calculate means on X_tr
    means = X_tr.groupby('combinaison')['converted'].mean()
    
    # Map to X_val
    df_train.loc[val_idx, 'TE_combinaison'] = X_val['combinaison'].map(means)

# Fill NaNs in Train (New combinations in fold) with Global Mean
global_mean = df_train['converted'].mean()
df_train['TE_combinaison'].fillna(global_mean, inplace=True)

# 2. Test Set (Use Full Train)
full_means = df_train.groupby('combinaison')['converted'].mean()
df_test['TE_combinaison'] = df_test['combinaison'].map(full_means)
df_test['TE_combinaison'].fillna(global_mean, inplace=True) # Unknown profiles get global mean

print("Target Encoding Complete.")

# Label Encoding for other cats
cat_cols = ['country', 'source']
for col in cat_cols:
    le = LabelEncoder()
    full = pd.concat([df_train[col], df_test[col]])
    le.fit(full)
    df_train[col] = le.transform(df_train[col])
    df_test[col] = le.transform(df_test[col])

features = ['country', 'source', 'new_user', 'age_bin', 'pages_bin', 'pages_age_ratio', 'TE_combinaison']

# Aggregation & Weights Check
# Note: Aggregation is tricky with TE because TE is row-specific (mostly unique).
# WE CANNOT AGGREGATE EASILY if we use TE.
# XGBoost is fast enough for 300k rows. Let's train on FULL ROWS to assume the TE signal is key.
# Or we aggregate by 'combinaison' but that's basically row-level for the interesting ones.
# Let's train row-level to be safe and precise.

print(f"Training XGBoost on {len(df_train)} rows with features: {features}")
model = xgb.XGBRegressor(**PARAMS)
model.fit(df_train[features], df_train['converted'])

# Threshold Opt
prob_train = model.predict(df_train[features])
best_f1 = 0
best_th = 0
for th in np.arange(0.3, 0.6, 0.005):
    f1 = f1_score(df_train['converted'], (prob_train >= th).astype(int))
    if f1 > best_f1: best_f1, best_th = f1, th
    
print(f"✅ TRAIN F1 SCORE : {best_f1:.5f}")
print(f"   Seuil Optimal  : {best_th:.3f}")

# Prediction
prob_test = model.predict(df_test[features])
final_preds = (prob_test >= best_th).astype(int)

submission = pd.DataFrame({'converted': final_preds})
submission.to_csv('submission_PROFESSOR_MODEL.csv', index=False)
print("✅ Submission Saved: submission_PROFESSOR_MODEL.csv")
print(f"   Total Conversions : {submission['converted'].sum()}")
