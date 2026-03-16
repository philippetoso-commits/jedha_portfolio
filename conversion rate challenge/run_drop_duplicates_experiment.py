
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder

print("🎓 PROFESSOR'S THEORY: THE DUPLICATE PARADOX 🎓")

# 1. Load Data
df = pd.read_csv('conversion rate challenge/conversion_data_train.csv')

# Features for Duplication Check
feature_cols = ['country', 'age', 'new_user', 'source', 'total_pages_visited']
# Note: We do NOT include target 'converted' in duplication check yet.

# 2. Analyze Duplicates
print(f"📥 Original Data Shape: {df.shape}")
n_dupes = df.duplicated(subset=feature_cols, keep=False).sum()
print(f"   Rows involved in duplicates: {n_dupes} ({n_dupes/len(df)*100:.1f}%)")

# Check Conflicts (Same Features, Different Converted)
# Group by Features, measure Variance of Target
conflicts = df.groupby(feature_cols)['converted'].agg(['mean', 'count', 'std'])
n_conflict_groups = conflicts[conflicts['std'] > 0].shape[0]
n_conflict_rows = conflicts[conflicts['std'] > 0]['count'].sum()

print(f"\n⚡ CONTRADICTIONS FOUND:")
print(f"   Groups with conflicting outcomes (0 AND 1): {n_conflict_groups}")
print(f"   Total rows in conflicting groups: {n_conflict_rows}")
print(conflicts[conflicts['std'] > 0].head())

# 3. Strategy Setup
# Strategy A: Baseline (Keep All)
# Strategy B: Drop Duplicates (Keep First) - Dangerous?
# Strategy C: Drop Conflicting Groups Entirely (Purist)
# Strategy D: Consensus (Replace Group with Mean Probability / Weighted Sample?) -> XGB handles this naturally, effectively weighting by count.
# Wait, if XGB handles weighting naturally, then duplicates are just weights.
# The Professor implies 'drop' is better. Maybe getting rid of the noise allows the tree to see the signal better?

def train_eval(data, name):
    X = data[feature_cols].copy()
    y = data['converted']
    
    # Encode
    for c in ['country', 'source']:
        le = LabelEncoder()
        X[c] = le.fit_transform(X[c])
        
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        eval_metric='logloss',
        use_label_encoder=False,
        n_jobs=-1,
        random_state=42
    )
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring='f1')
    print(f"   👉 {name:<20} | F1: {scores.mean():.5f} (+/- {scores.std():.4f})")
    return scores.mean()

print("\n🏆 BENCHMARKING STRATEGIES:")

# A. Baseline
train_eval(df, "Baseline (All)")

# B. Drop Duplicates (Keep First) - Effectively de-weighting frequent profiles
df_drop_first = df.drop_duplicates(subset=feature_cols, keep='first')
print(f"   (Data Size: {len(df_drop_first)})")
train_eval(df_drop_first, "Drop Dupes (First)")

# C. Deduplicate and Keep Majority Vote (Consensus)
# If mean > 0.5 -> 1, else 0
# Create a new dataset designated by unique profiles
grouped = df.groupby(feature_cols, as_index=False)['converted'].mean()
grouped['converted'] = (grouped['converted'] >= 0.5).astype(int)
print(f"   (Data Size: {len(grouped)})")
train_eval(grouped, "Consensus (Majority)")

# D. Feature Engineered Conflict Flag (Maybe the conflict IS the signal?)
# We assume XGBoost loves weights, so Baseline should win.
# Unless the noise is harmful.

print("\n🤔 VERDICT:")
# We will see output.
