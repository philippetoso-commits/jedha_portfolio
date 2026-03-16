# ============================================================
# 🎓 PROFESSOR TRUE METHOD — DUPLICATE SUPER-FEATURE (ONE CELL)
# ============================================================

import pandas as pd
import numpy as np
import xgboost as xgb

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score

SEED = 42
N_FOLDS = 10

# -----------------------
# Load data
# -----------------------
try:
    df = pd.read_csv("conversion_data_train.csv")
except FileNotFoundError:
    df = pd.read_csv("/home/phil/projetdatascience/conversion rate challenge/conversion_data_train.csv")

# -----------------------
# Exact combination (NO binning, NO transform)
# -----------------------
df['combinaison'] = (
    df['country'].astype(str) + "_" +
    df['age'].astype(str) + "_" +
    df['new_user'].astype(str) + "_" +
    df['source'].astype(str) + "_" +
    df['total_pages_visited'].astype(str)
)

# -----------------------
# Exact duplicate analysis (PROF CORE IDEA)
# -----------------------
stats = (
    df
    .groupby('combinaison')['converted']
    .agg(['count', 'min', 'max', 'mean'])
    .reset_index()
)

stats['has_positive'] = stats['max']                     # at least one 1
stats['is_ambiguous'] = (stats['min'] != stats['max'])   # mix 0 / 1
stats['exact_rate'] = stats['mean']                      # exact historical rate

# -----------------------
# Merge super-features
# -----------------------
df = df.merge(
    stats[['combinaison', 'has_positive', 'is_ambiguous', 'exact_rate']],
    on='combinaison',
    how='left'
)

# -----------------------
# Minimal encoding (as in class)
# -----------------------
for col in ['country', 'source']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# -----------------------
# Final feature set (PROF STYLE)
# -----------------------
FEATURES = [
    'country',
    'source',
    'new_user',
    'age',
    'total_pages_visited',
    'has_positive',
    'is_ambiguous',
    'exact_rate'
]

X = df[FEATURES]
y = df['converted']

# -----------------------
# Simple model (no tricks)
# -----------------------
model = xgb.XGBClassifier(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='logloss',
    random_state=SEED,
    n_jobs=-1
)

# -----------------------
# Cross-validation F1
# -----------------------
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
scores = []

print("="*70)
print("🎓 PROFESSOR TRUE METHOD — CV RESULTS")
print("="*70)

for i, (tr, val) in enumerate(skf.split(X, y), 1):
    model.fit(X.iloc[tr], y.iloc[tr])
    preds = model.predict(X.iloc[val])
    f1 = f1_score(y.iloc[val], preds)
    scores.append(f1)
    print(f"Fold {i:2d} | F1 = {f1:.4f}")

print("-"*70)
print(f"F1 MEAN : {np.mean(scores):.5f}")
print(f"F1 STD  : {np.std(scores):.5f}")
print("="*70)
