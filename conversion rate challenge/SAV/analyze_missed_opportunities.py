import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

print("="*80)
print("🕵️ ANALYSE DES 'POSITIFS MANQUÉS' (FALSE NEGATIVES)")
print("="*80)

SEED = 42
N_FOLDS = 5

# Config Theory Model
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

# Load
df = pd.read_csv('conversion_data_train.csv')

# Preprocessing
features = ['country', 'source', 'new_user', 'age', 'total_pages_visited']
for col in ['country', 'source']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

X = df[features]
y = df['converted']

# Generate Out-of-Fold Predictions (to simulate Test behavior)
print(f"Generating CV Predictions (5-Fold)...")
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
oof_preds = np.zeros(len(df))

for tr_idx, val_idx in skf.split(X, y):
    X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
    X_val = X.iloc[val_idx]
    
    model = xgb.XGBClassifier(**PARAMS)
    model.fit(X_tr, y_tr)
    oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]

# Apply Threshold (using the optimal one found previously ~0.38-0.42)
# Let's verify the optimal threshold on these OOF preds first
best_f1 = 0
best_th = 0
for th in np.arange(0.3, 0.6, 0.01):
    f1 = 2 * (np.sum((oof_preds >= th) & (y == 1))) / (np.sum(oof_preds >= th) + np.sum(y == 1))
    if f1 > best_f1: best_f1, best_th = f1, th

print(f"Optimal CV Threshold used: {best_th:.3f}")
binary_preds = (oof_preds >= best_th).astype(int)

# Identify False Negatives
df['pred_prob'] = oof_preds
df['pred_class'] = binary_preds
df['type'] = 'Correct'
df.loc[(y == 1) & (binary_preds == 0), 'type'] = 'FN (Manqué)'
df.loc[(y == 0) & (binary_preds == 1), 'type'] = 'FP (Hallucination)'

missed = df[df['type'] == 'FN (Manqué)'].copy()
print(f"Total Conversions Réelles : {y.sum()}")
print(f"Total Conversions Trouvées: {binary_preds.sum()}")
print(f"Total Manqués (FN)        : {len(missed)}")
print("-" * 60)

# Analyze Categories
if len(missed) > 0:
    print("📊 QUI SONT LES MANQUÉS ?")
    
    # 1. By Country
    print("\n[Par Pays]")
    print(missed['country'].apply(lambda x: ['China','Germany','UK','US'][x]).value_counts(normalize=True).to_string())
    
    # 2. By Source
    print("\n[Par Source]")
    print(missed['source'].apply(lambda x: ['Ads','Direct','Seo'][x]).value_counts(normalize=True).to_string())
    
    # 3. By Age (Binning)
    missed['age_group'] = pd.cut(missed['age'], bins=[0, 20, 30, 40, 50, 100])
    print("\n[Par Age]")
    print(missed['age_group'].value_counts(normalize=True).sort_index().to_string())
    
    # 4. By Pages
    print("\n[Par Pages Visitées]")
    print(missed['total_pages_visited'].value_counts().sort_index().head(10).to_string())
    
    # 5. The "Gray Zone" Check
    print("\n[Zone Grise (Prob 0.2 - Threshold)]")
    gray_zone_misses = missed[(missed['pred_prob'] > 0.2)]
    print(f"FN avec proba 'décente' (>0.20) : {len(gray_zone_misses)} / {len(missed)}")
    if len(gray_zone_misses) > 0:
        print("Exemple de profils manqués de peu :")
        print(gray_zone_misses[['country', 'age', 'total_pages_visited', 'source', 'pred_prob']].head(10))

else:
    print("Incroyable, 0 FN. Le modèle est parfait (ou il y a un bug).")
