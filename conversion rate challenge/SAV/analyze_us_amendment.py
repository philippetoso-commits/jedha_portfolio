import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, precision_recall_fscore_support

print("="*80)
print("🦅 AMENDEMENT US : ANALYSE TACTIQUE (ZONE 8-11 PAGES)")
print("="*80)

SEED = 42

# 1. Load & Preprocess
df = pd.read_csv('conversion_data_train.csv')

features = ['country', 'source', 'new_user', 'age', 'total_pages_visited']
for col in ['country', 'source']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# Note: In standard encoding (alphabetical):
# China=0, Germany=1, UK=2, US=3.
# Let's verify by checking the counts or just trusting standard behavior.
# US is usually the largest, let's grab the code for US.
us_code = df[df['country'] == 3]['country'].iloc[0] # Assuming 3 is US, but logic holds if 3 exists
print(f"Code Pays US (estimé): {us_code}")

X = df[features]
y = df['converted']

# 2. Get Out-Of-Fold Predictions (Clean Probabilities)
print("Generating OOF Predictions (5-Fold)...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
oof_preds = np.zeros(len(df))

PARAMS = {
    'learning_rate': 0.01,
    'max_depth': 6,
    'n_estimators': 500,
    'subsample': 0.8,
    'objective': 'binary:logistic',
    'n_jobs': -1,
    'random_state': SEED,
    'monotone_constraints': '(0, 0, 0, -1, 1)'
}

for tr_idx, val_idx in skf.split(X, y):
    X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
    X_val = X.iloc[val_idx]
    
    model = xgb.XGBClassifier(**PARAMS)
    model.fit(X_tr, y_tr)
    oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]

df['prob'] = oof_preds

# 3. Focus on the TARGET SEGMENT
# Country = US, Pages = [8, 9, 10, 11]
target_mask = (df['country'] == 3) & (df['total_pages_visited'].between(8, 11))
df_target = df[target_mask].copy()

print(f"\n📊 ZOOM SUR LE SEGMENT CIBLE (US / 8-11 Pages)")
print(f"Volume Total       : {len(df_target)}")
print(f"Vrais Convertis    : {df_target['converted'].sum()}")
print(f"Taux de Conversion : {df_target['converted'].mean():.2%}")

# 4. Find Optimal Local Threshold for this Segment
best_f1 = 0
best_th = 0
best_prec = 0
best_rec = 0

print("\n--- Optimisation du Seuil LOCAL ---")
print(f"{'Seuil':<10} | {'F1':<10} | {'Prec':<10} | {'Rec':<10} | {'TP':<5} | {'FP':<5}")

stats = []
for th in np.arange(0.1, 0.6, 0.02):
    preds = (df_target['prob'] >= th).astype(int)
    f1 = f1_score(df_target['converted'], preds, zero_division=0)
    p, r, _, _ = precision_recall_fscore_support(df_target['converted'], preds, average='binary', zero_division=0)
    
    tp = np.sum((preds == 1) & (df_target['converted'] == 1))
    fp = np.sum((preds == 1) & (df_target['converted'] == 0))
    
    stats.append((th, f1, p, r, tp, fp))
    
    if f1 > best_f1:
        best_f1, best_th = f1, th
        best_prec, best_rec = p, r

# Print range around optimal
for s in stats:
    print(f"{s[0]:.2f}       | {s[1]:.4f}     | {s[2]:.4f}     | {s[3]:.4f}     | {s[4]:<5} | {s[5]:<5}")

print("-" * 60)
print(f"🏆 MEILLEUR SEUIL LOCAL : {best_th:.2f}")
print(f"   F1 Local : {best_f1:.4f}")
print(f"   Gain (TP): {int(np.sum((df_target['prob'] >= best_th) & (df_target['converted'] == 1)))}")
print(f"   Coût (FP): {int(np.sum((df_target['prob'] >= best_th) & (df_target['converted'] == 0)))}")

# Compare with Global Threshold (approx 0.40)
global_tp = np.sum((df_target['prob'] >= 0.40) & (df_target['converted'] == 1))
global_fp = np.sum((df_target['prob'] >= 0.40) & (df_target['converted'] == 0))

print(f"\nVS Seuil Global (~0.40) :")
print(f"   TP : {global_tp}")
print(f"   FP : {global_fp}")
print(f"👉 Différentiel : +{int(np.sum((df_target['prob'] >= best_th) & (df_target['converted'] == 1))) - global_tp} Conversions / +{int(np.sum((df_target['prob'] >= best_th) & (df_target['converted'] == 0))) - global_fp} Faux Positifs")
