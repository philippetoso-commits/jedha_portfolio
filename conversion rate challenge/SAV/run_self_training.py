import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score

print("="*80)
print("🧠 SELF-TRAINING (PSEUDO-LABELING) : DOMAIN ADAPTATION")
print("="*80)

SEED = 42

# 1. Load Data
df_train = pd.read_csv('conversion_data_train.csv')
df_test = pd.read_csv('conversion_data_test.csv')

# Preprocessing
features = ['country', 'source', 'new_user', 'age', 'total_pages_visited']
for col in ['country', 'source']:
    le = LabelEncoder()
    full = pd.concat([df_train[col], df_test[col]])
    le.fit(full)
    df_train[col] = le.transform(df_train[col])
    df_test[col] = le.transform(df_test[col])

X_train = df_train[features]
y_train = df_train['converted']
X_test = df_test[features]

# Model Config (Theory V1 Base)
PARAMS = {
    'learning_rate': 0.01,
    'max_depth': 6,
    'n_estimators': 500,
    'subsample': 0.8,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'n_jobs': -1,
    'random_state': SEED,
    'monotone_constraints': '(0, 0, 0, -1, 1)',
    'tree_method': 'hist'
}

# 2. Train Teacher Model (Base V1)
print("Step 1: Training Teacher Model (V1)...")
teacher = xgb.XGBClassifier(**PARAMS)
teacher.fit(X_train, y_train)

# 3. Predict on Test
print("Step 2: Generating Pseudo-Labels on Test...")
probs_test = teacher.predict_proba(X_test)[:, 1]

# 4. Filter High Confidence Selection
TH_POS = 0.95
TH_NEG = 0.05

mask_pos = (probs_test >= TH_POS)
mask_neg = (probs_test <= TH_NEG)

pseudo_pos = X_test[mask_pos].copy()
pseudo_pos['converted'] = 1

pseudo_neg = X_test[mask_neg].copy()
pseudo_neg['converted'] = 0

print(f"   High Confidence Positives (> {TH_POS}) : {len(pseudo_pos)}")
print(f"   High Confidence Negatives (< {TH_NEG}) : {len(pseudo_neg)}")

# 5. Augment Training Set
print("Step 3: Augmenting Training Set...")
X_pseudo = pd.concat([pseudo_pos[features], pseudo_neg[features]])
y_pseudo = pd.concat([pseudo_pos['converted'], pseudo_neg['converted']])

X_aug = pd.concat([X_train, X_pseudo])
y_aug = pd.concat([y_train, y_pseudo])

print(f"   Original Size : {len(X_train)}")
print(f"   New Size      : {len(X_aug)} (+{len(X_pseudo)})")

# 6. Train Student Model (Self-Trained)
print("Step 4: Training Student Model (Self-Trained)...")
student = xgb.XGBClassifier(**PARAMS)
student.fit(X_aug, y_aug)

# 7. Generate Final Submission
# We must optimize threshold again? Or assume similar distribution?
# Safer to optimize threshold on ORIGINAL Train set to be robust.
probs_train_check = student.predict_proba(X_train)[:, 1]
best_f1 = 0
best_th = 0.5
for th in np.arange(0.3, 0.6, 0.005):
    f1 = f1_score(y_train, (probs_train_check >= th).astype(int))
    if f1 > best_f1: best_f1, best_th = f1, th

print(f"✅ STUDENT TRAIN F1 : {best_f1:.5f}")
print(f"   Seuil Optimal  : {best_th:.3f}")

final_probs = student.predict_proba(X_test)[:, 1]
final_preds = (final_probs >= best_th).astype(int)

submission = pd.DataFrame({'converted': final_preds})
submission.to_csv('submission_SELF_TRAINING.csv', index=False)
print(f"✅ Submission Saved: submission_SELF_TRAINING.csv")
print(f"   Total Conversions : {submission['converted'].sum()}")

# Comparison vs Teacher (Approximation)
# Using teacher with same threshold
teacher_preds = (probs_test >= best_th).astype(int) 
diff = np.sum(final_preds) - np.sum(teacher_preds)
print(f"   Diff vs Teacher : {diff:+} conversions")
