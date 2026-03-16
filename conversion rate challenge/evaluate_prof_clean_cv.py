import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score

print("="*80)
print("🧑‍🏫 PROF CLEARED : EVALUATION CV (5-FOLD RIGOYREUX)")
print("="*80)

SEED = 42
N_FOLDS = 5
ALPHA = 100

# Load
df = pd.read_csv("conversion_data_train.csv")

# Profiling Function
def profiling(df):
    df = df.copy()
    df["age_bin"] = pd.cut(df["age"], bins=[0, 18, 25, 30, 35, 40, 45, 50, 60, 100], labels=False)
    df["pages_bin"] = pd.cut(df["total_pages_visited"], bins=[0, 1, 2, 3, 4, 5, 6, 7, 9, 12, 16, 25, 100], labels=False)
    df["profile_key"] = (
        df["country"].astype(str) + "_" +
        df["source"].astype(str) + "_" +
        df["new_user"].astype(str) + "_" +
        df["age_bin"].astype(str) + "_" +
        df["pages_bin"].astype(str)
    )
    return df

df = profiling(df)

# Encode Categoricals globally (Label Encoding is safeish if consistent)
for col in ["country", "source"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

features = ["country", "source", "new_user", "age", "total_pages_visited", "prof_feature"]
X_base = df[["country", "source", "new_user", "age", "total_pages_visited", "profile_key", "converted"]] # Keep key and target for manual splitting

skf_outer = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
f1_scores = []
thresholds = []

print(f"Starting {N_FOLDS}-Fold CV with Nested Target Encoding...")

for i, (train_idx, val_idx) in enumerate(skf_outer.split(X_base, X_base["converted"])):
    print(f"Processing Fold {i+1}...")
    
    # Split
    tr, val = X_base.iloc[train_idx].copy(), X_base.iloc[val_idx].copy()
    global_mean_tr = tr["converted"].mean()
    
    # --- INNER TARGET ENCODING (OOF for Train part seems overkill if we just want valid features for the model) ---
    # Actually, for the XGBoost model to learn to use 'prof_feature', it needs 'prof_feature' populated on its Training set.
    # To avoid overfitting, 'prof_feature' on True Training Set should be OOF-generated (Inner CV).
    # 'prof_feature' on Validation Set should be standard mapping from True Training Set.
    
    # 1. Generate 'prof_feature' for TR using Inner OOF
    # This matches the user's script logic "df_train['prof_feature'] = ..."
    tr["prof_feature"] = np.nan
    skf_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    
    for in_tr_idx, in_val_idx in skf_inner.split(tr, tr["converted"]):
        in_tr = tr.iloc[in_tr_idx]
        in_val = tr.iloc[in_val_idx]
        
        stats = in_tr.groupby("profile_key")["converted"].agg(["mean", "count"])
        stats["prof_prob"] = (stats["count"] * stats["mean"] + ALPHA * global_mean_tr) / (stats["count"] + ALPHA)
        
        tr.iloc[in_val_idx, tr.columns.get_loc("prof_feature")] = in_val["profile_key"].map(stats["prof_prob"])
        
    tr["prof_feature"].fillna(global_mean_tr, inplace=True)
    
    # 2. Generate 'prof_feature' for VAL using Full TR stats
    stats_full = tr.groupby("profile_key")["converted"].agg(["mean", "count"])
    stats_full["prof_prob"] = (stats_full["count"] * stats_full["mean"] + ALPHA * global_mean_tr) / (stats_full["count"] + ALPHA)
    
    val["prof_feature"] = val["profile_key"].map(stats_full["prof_prob"])
    val["prof_feature"].fillna(global_mean_tr, inplace=True)
    
    # Prepare Arrays
    X_tr_final = tr[features]
    y_tr_final = tr["converted"]
    X_val_final = val[features]
    y_val_final = val["converted"]
    
    # Train Model
    model = xgb.XGBClassifier(
        n_estimators=400,
        learning_rate=0.03,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=SEED,
        n_jobs=-1
    )
    model.fit(X_tr_final, y_tr_final)
    
    # Evaluate
    probs = model.predict_proba(X_val_final)[:, 1]
    
    b_f1, b_th = 0, 0.5
    for th in np.arange(0.3, 0.6, 0.01):
        f = f1_score(y_val_final, (probs >= th).astype(int))
        if f > b_f1: b_f1, b_th = f, th
        
    f1_scores.append(b_f1)
    thresholds.append(b_th)
    print(f"   -> F1={b_f1:.5f} (Th={b_th:.2f})")

print("-" * 60)
print(f"📊 RESULTATS PROF CLEAN ({N_FOLDS} Folds)")
print(f"   AVG F1 Score : {np.mean(f1_scores):.5f} (± {np.std(f1_scores):.5f})")
print(f"   AVG Threshold: {np.mean(thresholds):.5f}")
