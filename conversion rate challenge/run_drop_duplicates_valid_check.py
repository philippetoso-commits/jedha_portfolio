
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder

print("⚖️ THE ARBITER: CHATGPT VS THE PROFESSOR ⚖️")

# 1. Load Data
df = pd.read_csv('conversion rate challenge/conversion_data_train.csv')
feature_cols = ['country', 'age', 'new_user', 'source', 'total_pages_visited']

# Encode
for c in ['country', 'source']:
    le = LabelEncoder()
    df[c] = le.fit_transform(df[c])

X = df[feature_cols]
y = df['converted']

# 2. Split Data (Fixed Validation Set)
# We must evaluate on a representative sample of reality (Full Val)
X_train_full, X_val, y_train_full, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"   Train Size (Full): {len(X_train_full)}")
print(f"   Val Size (Full):   {len(X_val)}")

# 3. Create Training Variants
# Variant A: Full (Control)
dtrain_full = xgb.DMatrix(X_train_full, label=y_train_full)

# Variant B: Unique (Professor's Idea - Drop Duplicates)
train_full_combined = X_train_full.copy()
train_full_combined['converted'] = y_train_full
train_unique = train_full_combined.drop_duplicates(subset=feature_cols, keep='first')
X_train_unique = train_unique[feature_cols]
y_train_unique = train_unique['converted']
dtrain_unique = xgb.DMatrix(X_train_unique, label=y_train_unique)
print(f"   Train Size (Unique): {len(X_train_unique)} (Dropping duplicates from Train)")

# Variant C: Consensus (Majority Vote / Mean)
# Group by features, take mean target.
# If we treat mean as label, XGBoost does regression? Or we threshold?
# Let's threshold for classification 0/1. >= 0.5 -> 1.
train_consensus = train_full_combined.groupby(feature_cols, as_index=False)['converted'].mean()
train_consensus['converted'] = (train_consensus['converted'] >= 0.5).astype(int)
X_train_consensus = train_consensus[feature_cols]
y_train_consensus = train_consensus['converted']
dtrain_consensus = xgb.DMatrix(X_train_consensus, label=y_train_consensus)
print(f"   Train Size (Consensus): {len(X_train_consensus)}")

# Validation Set (ALWAYS THE SAME)
dval = xgb.DMatrix(X_val, label=y_val)

# 4. Train & Evaluate
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'max_depth': 6,
    'learning_rate': 0.1,
    'seed': 42
}

def evaluate_model(name, dtrain):
    print(f"\n🥊 Round: {name}")
    model = xgb.train(params, dtrain, num_boost_round=100)
    
    # Predict on Fixed Validation Set
    probs = model.predict(dval)
    preds = (probs > 0.5).astype(int)
    score = f1_score(y_val, preds)
    
    print(f"   F1 Score on Full Val: {score:.5f}")
    return score

score_a = evaluate_model("Full Data (ChatGPT Bias)", dtrain_full)
score_b = evaluate_model("Unique Data (Drop Dupes)", dtrain_unique)
score_c = evaluate_model("Consensus Data (Cleaned)", dtrain_consensus)

print("\n🏆 FINAL VERDICT:")
best_score = max(score_a, score_b, score_c)
if best_score == score_a:
    print("   ChatGPT WINS! (Volume information is crucial. Dropping duplicates hurts.)")
elif best_score == score_b:
    print("   PROFESSOR WINS! (Noise is toxic. Dropping duplicates cleans the signal.)")
else:
    print("   CONSENSUS WINS! (Cleaning contradictions is better than blind dropping.)")
