import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score

print("="*80)
print("🧑‍🏫 PROF CLEARED : TARGET ENCODING ROBUSTE")
print("="*80)

# =========================
# CONFIG
# =========================
SEED = 42
N_FOLDS = 5
ALPHA = 100   # Régularisation FORTE (clé académique)

# =========================
# LOAD DATA
# =========================
try:
    df_train = pd.read_csv("conversion_data_train.csv")
    df_test  = pd.read_csv("conversion_data_test.csv")
except FileNotFoundError:
    # Fallback to absolute path if needed, but we expect to run in the correct cwd
    df_train = pd.read_csv("/home/phil/projetdatascience/conversion rate challenge/conversion_data_train.csv")
    df_test  = pd.read_csv("/home/phil/projetdatascience/conversion rate challenge/conversion_data_test.csv")

# =========================
# PROFILING (STRUCTUREL)
# =========================
def profiling(df):
    df = df.copy()
    
    df["age_bin"] = pd.cut(
        df["age"],
        bins=[0, 18, 25, 30, 35, 40, 45, 50, 60, 100],
        labels=False
    )

    df["pages_bin"] = pd.cut(
        df["total_pages_visited"],
        bins=[0, 1, 2, 3, 4, 5, 6, 7, 9, 12, 16, 25, 100],
        labels=False
    )

    df["profile_key"] = (
        df["country"].astype(str) + "_" +
        df["source"].astype(str) + "_" +
        df["new_user"].astype(str) + "_" +
        df["age_bin"].astype(str) + "_" +
        df["pages_bin"].astype(str)
    )

    return df

df_train = profiling(df_train)
df_test  = profiling(df_test)

# =========================
# TARGET ENCODING OOF (PROF)
# =========================
print("Calculating OOF Target Encoding...")
df_train["prof_feature"] = np.nan
global_mean = df_train["converted"].mean()

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

for tr_idx, val_idx in skf.split(df_train, df_train["converted"]):
    tr, val = df_train.iloc[tr_idx], df_train.iloc[val_idx]
    
    stats = tr.groupby("profile_key")["converted"].agg(["mean", "count"])
    stats["prof_prob"] = (
        stats["count"] * stats["mean"] + ALPHA * global_mean
    ) / (stats["count"] + ALPHA)

    # Map safely
    df_train.loc[val_idx, "prof_feature"] = (
        val["profile_key"].map(stats["prof_prob"])
    )

df_train["prof_feature"].fillna(global_mean, inplace=True)

# Test = stats full train
print("Encoding Test Set...")
stats_full = df_train.groupby("profile_key")["converted"].agg(["mean", "count"])
stats_full["prof_prob"] = (
    stats_full["count"] * stats_full["mean"] + ALPHA * global_mean
) / (stats_full["count"] + ALPHA)

df_test["prof_feature"] = (
    df_test["profile_key"].map(stats_full["prof_prob"])
)
df_test["prof_feature"].fillna(global_mean, inplace=True)

# =========================
# ENCODAGE CAT SIMPLE
# =========================
for col in ["country", "source"]:
    le = LabelEncoder()
    le.fit(pd.concat([df_train[col], df_test[col]]))
    df_train[col] = le.transform(df_train[col])
    df_test[col]  = le.transform(df_test[col])

# =========================
# MODEL (SIMPLE & PROPRE)
# =========================
features = [
    "country",
    "source",
    "new_user",
    "age",
    "total_pages_visited",
    "prof_feature"
]

X = df_train[features]
y = df_train["converted"]
X_test = df_test[features]

print("Training XGBoost...")
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

model.fit(X, y)

# =========================
# THRESHOLD OPT
# =========================
probs = model.predict_proba(X)[:, 1]

best_f1, best_th = 0, 0.5
print("Optimizing Threshold on Train (Bias Check)...")
for th in np.arange(0.30, 0.55, 0.01):
    f1 = f1_score(y, (probs >= th).astype(int))
    if f1 > best_f1:
        best_f1, best_th = f1, th

print(f"F1 TRAIN : {best_f1:.4f} | TH = {best_th:.2f}")

# =========================
# SUBMISSION
# =========================
preds_test = (model.predict_proba(X_test)[:, 1] >= best_th).astype(int)
pd.DataFrame({"converted": preds_test}).to_csv(
    "submission_PROF_CLEAN.csv",
    index=False
)

print(f"✅ submission_PROF_CLEAN.csv générée")
print(f"   Total Conversions: {preds_test.sum()}")
