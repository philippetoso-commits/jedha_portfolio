
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import HistGradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

print("🧪 SYNDICATE FINAL CUT: FULL PIPELINE CV EVALUATION 🧪")

# 1. SETUP
df = pd.read_csv('conversion rate challenge/conversion_data_train.csv')
y_true = df['converted']

# Feature Engineering Function (Shared)
def engineered_features(df):
    d = df.copy()
    d['pages_per_age'] = d['total_pages_visited'] / (d['age'] + 0.1)
    d['interaction'] = d['total_pages_visited'] * d['age']
    d['pages_sq'] = d['total_pages_visited'] ** 2
    d['age_sq'] = d['age'] ** 2
    return d

numeric_features = ['age', 'total_pages_visited', 'pages_per_age', 'interaction', 'pages_sq', 'age_sq']
categorical_features = ['country', 'source', 'new_user']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
    ]
)

# 2. DEFINING THE PIPELINE LOGIC INSIDE CV
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
f1_scores = []
conversions_counts = []

print(f"   Starting 5-Fold CV on {len(df)} rows...")

for fold, (train_idx, val_idx) in enumerate(skf.split(df, y_true)):
    # Split
    X_train_raw = df.iloc[train_idx].copy()
    y_train_raw = y_true.iloc[train_idx]
    X_val = df.iloc[val_idx].copy()
    y_val = y_true.iloc[val_idx]
    
    # A. CONSENSUS TRAINING (The Professor's Cleaning)
    # Group duplicates in TRAIN only
    feature_cols = ['country', 'age', 'new_user', 'source', 'total_pages_visited']
    train_combined = X_train_raw.copy()
    train_combined['converted'] = y_train_raw
    
    consensus = train_combined.groupby(feature_cols, as_index=False)['converted'].mean()
    consensus['converted'] = (consensus['converted'] >= 0.5).astype(int)
    
    X_train_clean = engineered_features(consensus.drop('converted', axis=1))
    y_train_clean = consensus['converted']
    
    # B. TRAIN SENATE (With Reduced Data)
    clf_xgb = xgb.XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=6, random_state=42, n_jobs=-1, eval_metric='logloss')
    clf_lgb = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, num_leaves=31, random_state=42, n_jobs=-1, verbose=-1)
    clf_hgb = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.05, max_depth=6, random_state=42)
    
    ensemble = VotingClassifier(estimators=[('xgb', clf_xgb), ('lgb', clf_lgb), ('hgb', clf_hgb)], voting='soft')
    pipeline_model = Pipeline([('preprocessor', preprocessor), ('model', ensemble)])
    
    pipeline_model.fit(X_train_clean, y_train_clean)
    
    # Predict Base
    X_val_eng = engineered_features(X_val)
    base_probs = pipeline_model.predict_proba(X_val_eng)[:, 1]
    
    # Threshold 0.45 (Aggressive Base)
    preds = (base_probs > 0.45).astype(int)
    
    # C. APPLY RULES (The Amendments)
    # American Hustle
    mask_us = ((X_val['country'] == 'US') & (X_val['age'] >= 20) & (X_val['age'] <= 30) & (X_val['total_pages_visited'] >= 12))
    preds[mask_us] = 1
    
    # Erasmus
    mask_erasmus = ((X_val['new_user'] == 1) & (X_val['total_pages_visited'] >= 8) & (X_val['total_pages_visited'] <= 16) & (X_val['country'].isin(['Germany', 'UK'])) & (X_val['age'] < 25))
    preds[mask_erasmus] = 1
    
    # Mariage Frères
    mask_mariage = ((X_val['new_user'] == 0) & (X_val['total_pages_visited'] >= 12) & (X_val['total_pages_visited'] <= 16))
    preds[mask_mariage] = 1
    
    # D. FORENSIC AUDIT (The Cleaner)
    # We need a Formula model trained on FULL RAW TRAIN (for stats volume)
    formula = LogisticRegression(solver='lbfgs', max_iter=1000, C=1e9)
    pipe_formula = Pipeline([('preprocessor', preprocessor), ('model', formula)])
    pipe_formula.fit(engineered_features(X_train_raw), y_train_raw)
    
    f_probs = pipe_formula.predict_proba(X_val_eng)[:, 1]
    
    # Kill False Positives
    hallucinations = (preds == 1) & (f_probs < 0.10)
    preds[hallucinations] = 0
    
    # SCORE
    score = f1_score(y_val, preds)
    f1_scores.append(score)
    conversions_counts.append(preds.sum())
    
    print(f"   Fold {fold+1}: F1 = {score:.5f} (Conversions: {preds.sum()})")

mean_f1 = np.mean(f1_scores)
print(f"\n🏆 ESTIMATED FINAL F1: {mean_f1:.5f} (+/- {np.std(f1_scores):.4f})")
print(f"   Avg Conversions per Fold: {np.mean(conversions_counts):.1f}")
