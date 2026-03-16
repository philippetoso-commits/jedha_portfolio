# ============================================================
# 🎓 PROFESSOR TRUE METHOD — DUPLICATE SUPER-FEATURE (NO LEAK)
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
# Exact combination key (NO binning)
# -----------------------
df['combinaison'] = (
    df['country'].astype(str) + "_" +
    df['age'].astype(str) + "_" +
    df['new_user'].astype(str) + "_" +
    df['source'].astype(str) + "_" +
    df['total_pages_visited'].astype(str)
)

# Prepare OOF columns
df['has_positive'] = np.nan
df['is_ambiguous'] = np.nan
df['exact_rate'] = np.nan

# -----------------------
# OOF duplicate analysis (NO LEAKAGE)
# -----------------------
print("Generating OOF Features (No Leakage)...")
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

for train_idx, val_idx in skf.split(df, df['converted']):
    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]

    stats = (
        train_df
        .groupby('combinaison')['converted']
        .agg(['min', 'max', 'mean'])
    )

    df.loc[val_idx, 'has_positive'] = val_df['combinaison'].map(stats['max'])
    # ambiguous if we have seen both 0 and 1 in TRAINING history for this combo
    df.loc[val_idx, 'is_ambiguous'] = (
        (val_df['combinaison'].map(stats['min']) == 0) & 
        (val_df['combinaison'].map(stats['max']) == 1)
    )
    df.loc[val_idx, 'exact_rate'] = val_df['combinaison'].map(stats['mean'])

# Fill unseen combinations with safe defaults
# If unseen in training, we know nothing -> fill with global mean or 0 signal
global_mean = df['converted'].mean()
df['has_positive'].fillna(0, inplace=True)
df['is_ambiguous'].fillna(False, inplace=True) # Bool to int conversion later
df['exact_rate'].fillna(global_mean, inplace=True)

df['is_ambiguous'] = df['is_ambiguous'].astype(int)

# -----------------------
# Encode categorical vars
# -----------------------
for col in ['country', 'source']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# -----------------------
# Feature set (PROF PURE)
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
# Cross-validation
# -----------------------
scores = []
print("="*70)
print("🎓 PROFESSOR TRUE METHOD — NO LEAKAGE CV")
print("="*70)

# We can reuse the same SKF split for evaluation to match the OOF loop if we want perfect consistency,
# but using a fresh SKF is also fine to test robustness (OOF features were generated ONCE globally, 
# but effectively 'leak-free' relative to their specific folds. 
# Wait, if we use X (which has OOF features from Split A) and we split using Split B, 
# then for a validation sample in Split B, its 'exact_rate' might have come from a training fold in Split A 
# that INCLUDED itself? 
# NO! The OOF generation loop iterates over the WHOLE dataframe.
# For each sample in the dataframe, its 'exact_rate' was calculated using 90% of OTHER data.
# So 'exact_rate' is a valid feature.
# Then when we CV-evaluate the Model, we can split however we want. The feature is "valid historical rate from other data".

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
