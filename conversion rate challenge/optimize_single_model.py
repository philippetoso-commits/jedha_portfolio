import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import f1_score, roc_auc_score, make_scorer
from sklearn.preprocessing import LabelEncoder

print("="*80)
print("🧬 SINGLE MODEL THEORY : XGBOOST MONOTONIC (BERNOULLI MLE)")
print("="*80)

SEED = 42
N_FOLDS = 5

# 1. Loading & "Minimal" Preprocessing
# "Pas de feature supervisée", "Pas de réorganisation"
df_train = pd.read_csv('conversion_data_train.csv')
df_test = pd.read_csv('conversion_data_test.csv')

# Feature Engineering: Strictly limited to raw + standard ratio (implied by 'pages' vs 'age')
# To adhere strictly to "Pas de feature supervisée", we stick to basic transforms.
df_train['age_bin'] = pd.cut(df_train['age'], bins=[0, 18, 25, 30, 35, 40, 45, 50, 60, 100], labels=False).fillna(-1).astype(int)
df_test['age_bin'] = pd.cut(df_test['age'], bins=[0, 18, 25, 30, 35, 40, 45, 50, 60, 100], labels=False).fillna(-1).astype(int)

# Encoding (Standard Label Encoding for Cats)
for col in ['country', 'source']:
    le = LabelEncoder()
    full = pd.concat([df_train[col], df_test[col]])
    le.fit(full)
    df_train[col] = le.transform(df_train[col])
    # df_test[col] = le.transform(df_test[col]) # Not used for optimization but good practice

features = ['country', 'source', 'new_user', 'age', 'total_pages_visited']
# Note: 'age_bin' is useful for trees to find cuts easier, but 'age' raw is fine too. 
# We'll use the raw features + constraints.

X = df_train[features]
y = df_train['converted']

# 2. Model Configuration
# Theory:
# - Generator: Bernoulli (0/1) -> Objective: binary:logistic
# - Structure: Monotonic (More pages -> More prob)
# - Regularization: Essential to prevent fitting noise (ceiling at 0.767)

# Monotonic Constraints:
# country (0), source (0), new_user (?), age (-1?), total_pages_visited (+1)
# 0: No constraint, 1: Increasing, -1: Decreasing
# Features: ['country', 'source', 'new_user', 'age', 'total_pages_visited']
# Constraints: (0, 0, 0, -1, 1) -> Age decreases conv, Pages increases conv.
feature_monotony = '(0, 0, 0, -1, 1)'

model_base = xgb.XGBClassifier(
    objective='binary:logistic',
    n_jobs=-1,
    random_state=SEED,
    monotone_constraints=feature_monotony,
    tree_method='hist' # Faster
)

# 3. Hyperparameter Space (Reasonable & Useful)
# "Liste courte d’hyperparamètres réellement utiles"
param_grid = {
    'learning_rate': [0.01, 0.03, 0.05],   # Step size
    'max_depth': [3, 4, 5, 6],             # Complexity (Low for stability)
    'n_estimators': [300, 500, 800],       # Duration
    'subsample': [0.8, 1.0],               # Resilience
    # 'min_child_weight': [10, 20]         # Leaf size (Control overfitting)
}

# 4. Optimization Strategy (Grid Search with Stratified CV)
print(f"Features used: {features}")
print(f"Constraints: {feature_monotony}")
print("Starting Grid Search...")

def custom_f1_eval(y_true, y_pred_proba):
    # Optimize threshold for this specific fold
    best_f1 = 0
    # best_th = 0 # unused
    thresholds = np.linspace(0.3, 0.6, 31)
    for th in thresholds:
        f1 = f1_score(y_true, (y_pred_proba >= th).astype(int))
        if f1 > best_f1: best_f1 = f1
    return best_f1

# We need a custom scoring function for GridSearch that optimizes threshold implicitly?
# Standard 'f1' uses th=0.5. We need strict evaluation.
# Let's run a manual loop for clarity and control over reporting.

best_score = 0
best_params = {}
results = []

import itertools
keys, values = zip(*param_grid.items())
combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

print(f"Testing {len(combinations)} combinations...")

for i, params in enumerate(combinations):
    cv_f1 = []
    cv_auc = []
    cv_th = []
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    
    for tr_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
        
        model = xgb.XGBClassifier(**params, 
                                  objective='binary:logistic', 
                                  n_jobs=-1, 
                                  random_state=SEED,
                                  monotone_constraints=feature_monotony,
                                  eval_metric='logloss')
        model.fit(X_tr, y_tr)
        probs = model.predict_proba(X_val)[:, 1]
        
        auc = roc_auc_score(y_val, probs)
        
        # Th optimization
        b_f1 = 0
        b_th = 0.5
        for th in np.arange(0.3, 0.6, 0.02):
            f = f1_score(y_val, (probs >= th).astype(int))
            if f > b_f1: b_f1, b_th = f, th
            
        cv_f1.append(b_f1)
        cv_auc.append(auc)
        cv_th.append(b_th)
        
    mean_f1 = np.mean(cv_f1)
    std_f1 = np.std(cv_f1)
    mean_auc = np.mean(cv_auc)
    mean_th = np.mean(cv_th)
    
    # Progress (limited output)
    if mean_f1 > best_score:
        best_score = mean_f1
        best_params = params
        print(f"🌟 New Best: F1={mean_f1:.5f} (±{std_f1:.5f}) | AUC={mean_auc:.5f} | Params={params}")

print("="*80)
print("🏆 CONFIGURATION FINALE")
print(f"Params: {best_params}")
print(f"Avg F1 (5-Fold): {best_score:.5f}")
