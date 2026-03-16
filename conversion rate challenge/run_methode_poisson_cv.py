import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
import os

# Set working directory to project root if needed
# os.chdir('/home/phil/projetdatascience/conversion rate challenge/')

print("🚀 Démarrage du script Methode Poisson 5-Fold CV...")

# 1. Chargement des données
try:
    train_data = pd.read_csv('conversion_data_train.csv')
    test_data = pd.read_csv('conversion_data_test.csv')
except FileNotFoundError:
    print("❌ Erreur: Fichiers CSV introuvables. Vérifiez le chemin.")
    exit()

# 2. Préparation des variables
target = 'converted'
X = train_data.drop(target, axis=1)
y = train_data[target]

# Encodage des catégories
categorical_cols = ['country', 'source']
X_encoded = X.copy()
X_test_encoded = test_data.copy()
cat_indices = [X.columns.get_loc(col) for col in categorical_cols]

for col in categorical_cols:
    le = LabelEncoder()
    # Fit sur train + test pour gérer tout le scope
    le.fit(pd.concat([X[col], test_data[col]]))
    X_encoded[col] = le.transform(X[col])
    X_test_encoded[col] = le.transform(test_data[col])

# 3. Cross-Validation 5 Folds Stratifiée
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

f1_scores = []
thresholds = []
auc_scores = []

print(f"\ndataset shape: {X_encoded.shape}")
print("-" * 30)

for fold, (train_idx, val_idx) in enumerate(skf.split(X_encoded, y)):
    X_train_fold, X_val_fold = X_encoded.iloc[train_idx], X_encoded.iloc[val_idx]
    y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
    
    # Entraînement
    model = HistGradientBoostingRegressor(
        loss='poisson',
        learning_rate=0.05,
        max_iter=500,
        max_depth=8,
        l2_regularization=0.1,
        categorical_features=cat_indices,
        random_state=42
    )
    model.fit(X_train_fold, y_train_fold)
    
    # Prédiction & Optimisation Seuil
    val_scores = model.predict(X_val_fold)
    
    best_f1_fold = 0
    best_t_fold = 0
    # On cherche le seuil sur les prédictions du fold
    ts = np.linspace(val_scores.min(), val_scores.max(), 100)
    
    for t in ts:
        score = f1_score(y_val_fold, (val_scores >= t).astype(int))
        if score > best_f1_fold:
            best_f1_fold = score
            best_t_fold = t
            
    f1_scores.append(best_f1_fold)
    thresholds.append(best_t_fold)
    auc_scores.append(roc_auc_score(y_val_fold, val_scores))
    print(f"Fold {fold+1}: F1={best_f1_fold:.5f} | Seuil={best_t_fold:.5f} | AUC={auc_scores[-1]:.5f}")

avg_f1 = np.mean(f1_scores)
avg_threshold = np.mean(thresholds)
avg_auc = np.mean(auc_scores)

print(f"\n--- RÉSULTATS MOYENS 5-FOLD ---")
print(f"Mean F1-Score : {avg_f1:.5f}")
print(f"Mean Threshold: {avg_threshold:.6f}")
print(f"Mean ROC-AUC  : {avg_auc:.5f}")

# 4. Entraînement Final sur TOUT le dataset
print("\n🚀 Entraînement final sur l'ensemble du dataset...")
final_model = HistGradientBoostingRegressor(
    loss='poisson',
    learning_rate=0.05,
    max_iter=500,
    max_depth=8,
    l2_regularization=0.1,
    categorical_features=cat_indices,
    random_state=42
)
final_model.fit(X_encoded, y)

# 5. Génération Soumission avec Seuil Moyen
test_scores = final_model.predict(X_test_encoded)
test_preds = (test_scores >= avg_threshold).astype(int)

submission = pd.DataFrame({'converted': test_preds})
filename = 'submission_POISSON_MONSTER_CV.csv'
submission.to_csv(filename, index=False)
print(f"✅ Fichier '{filename}' généré ({test_preds.sum()} conversions).")
