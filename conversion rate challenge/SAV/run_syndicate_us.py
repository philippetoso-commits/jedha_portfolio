
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb

print("🇺🇸 SYNDICATE US: 'THE AMERICAN HUSTLE' EVALUATION 🇺🇸")

# 1. Chargement & Préparation Train pour Evaluation
print("📥 Loading Data...")
train_df = pd.read_csv('conversion rate challenge/conversion_data_train.csv')
y_true = train_df['converted']

# On va simuler un modèle de base (XGBoost simple) pour avoir des probas de base
# Puis on applique le Boost US pour voir si le F1 monte.
print("⚙️ Training Base Model for Baseline Probas...")
X = train_df.drop('converted', axis=1)
# Feature Eng minimal
X['pages_per_age'] = X['total_pages_visited'] / (X['age'] + 0.1)
for col in ['country', 'source']:
    X[col] = X[col].astype('category')

# XGB Params
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'enable_categorical': True,
    'learning_rate': 0.1,
    'max_depth': 6
}

# 5-Fold CV pour évaluer la règle
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
base_f1s = []
boosted_f1s = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_true)):
    X_tr, y_tr = X.iloc[train_idx], y_true.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y_true.iloc[val_idx]
    
    # Train Base
    dtrain = xgb.DMatrix(X_tr, label=y_tr, enable_categorical=True)
    dval = xgb.DMatrix(X_val, enable_categorical=True)
    model = xgb.train(params, dtrain, num_boost_round=100)
    
    # Predict Proba
    proba = model.predict(dval)
    
    # Base Threshold (0.5 standard)
    pred_base = (proba > 0.5).astype(int)
    base_f1 = f1_score(y_val, pred_base)
    base_f1s.append(base_f1)
    
    # APPLY US RULE
    # Logic: If (US & 20-30 & Pages >= 12), Boost Proba
    # C'est un peu Tricky car dval est une DMatrix. On reprend X_val.
    
    # Identify Target Rows in Validation Set
    mask_us_target = (X_val['country'] == 'US') & (X_val['age'] >= 20) & (X_val['age'] <= 30) & (X_val['total_pages_visited'] >= 12)
    
    # Boost Logic: Probability + 0.15 (Force le passage de 0.35 à 0.50)
    proba_boosted = proba.copy()
    proba_boosted[mask_us_target] += 0.15 
    
    pred_boosted = (proba_boosted > 0.5).astype(int)
    boost_f1 = f1_score(y_val, pred_boosted)
    boosted_f1s.append(boost_f1)

print(f"\n📊 --- LOCAL CV RESULTS (5-Fold) ---")
print(f"   Base Model F1:    {np.mean(base_f1s):.5f}")
print(f"   US Boosted F1:    {np.mean(boosted_f1s):.5f}")
delta = np.mean(boosted_f1s) - np.mean(base_f1s)
print(f"   Impact of Rule:   {'+' if delta > 0 else ''}{delta:.5f}")

if delta > 0:
    print("✅ CONCLUSION: The US Rule improves F1!")
else:
    print("❌ CONCLUSION: The US Rule degrades F1 (or is neutral).")

# 2. GENERATION DU FICHIER DE SOUMISSION
print("\n🚀 Génération du Fichier 'submission_SYNDICATE_USA.csv'...")
# On part du Syndicate Optimal existant comme base solide
df_syn = pd.read_csv('conversion rate challenge/submission_SYNDICATE_OPTIMAL.csv')
test_df = pd.read_csv('conversion rate challenge/conversion_data_test.csv')

# Application de la règle "Hard" sur le fichier
# Rule: If US, 20-30, Pages >= 13 (Soyons sûrs avec 13 = 51% chance), FORCE 1.
# On prend 13 pour être safe (High Potential confirmed).

mask_us_force = (test_df['country'] == 'US') & (test_df['age'] >= 20) & (test_df['age'] <= 30) & (test_df['total_pages_visited'] >= 13)

initial_ones = df_syn['converted'].sum()
# Force conversion
df_syn.loc[mask_us_force, 'converted'] = 1
final_ones = df_syn['converted'].sum()

print(f"   Base Conversions: {initial_ones}")
print(f"   New Conversions:  {final_ones}")
print(f"   Added by US Rule: {final_ones - initial_ones}")

df_syn.to_csv('conversion rate challenge/submission_SYNDICATE_USA.csv', index=False)
print("✅ Fichier généré.")
