
import pandas as pd
import numpy as np
from scipy.stats import wasserstein_distance
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
import warnings

warnings.filterwarnings('ignore')

print("📥 Chargement des données...")
train_df = pd.read_csv('conversion rate challenge/conversion_data_train.csv')
test_df = pd.read_csv('conversion rate challenge/conversion_data_test.csv')

print(f"Train shape: {train_df.shape}")
print(f"Test shape:  {test_df.shape}")

# Feature Engineering minimaliste
for df in [train_df, test_df]:
    df['pages_per_age'] = df['total_pages_visited'] / (df['age'] + 0.1)

# --- 1. Distances Statistiques ---
print("\n📊 Distances de Wasserstein (Distributions Numériques) :")
features_num = ['age', 'total_pages_visited', 'pages_per_age']
for col in features_num:
    wd = wasserstein_distance(train_df[col], test_df[col])
    print(f"   - {col:<20} : {wd:.6f}")
    if wd > 0.1:
        print(f"     ⚠️ ATTENTION : Dérive significative sur {col} !")

# --- 2. Adversarial Validation ---
print("\n🛡️ Lancement de l'Adversarial Validation...")
train_df['is_test'] = 0
test_df['is_test'] = 1

full_df = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)
target_adv = full_df['is_test']
# On retire 'converted' (cible) et 'is_test'
X_adv = full_df.drop(['converted', 'is_test'], axis=1)

# Encoding pour le modèle
for col in ['country', 'source']:
    le = LabelEncoder()
    # Fit sur tout pour éviter erreurs
    X_adv[col] = le.fit_transform(X_adv[col].astype(str))

# Modèle Détecteur
clf_adv = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)

# Cross-Validation ROC AUC
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(clf_adv, X_adv, target_adv, cv=cv, scoring='roc_auc')

auc_mean = scores.mean()
print(f"   👉 Adversarial AUC Score : {auc_mean:.4f} (+/- {scores.std():.4f})")

print("\n🔍 ANALYSE DU RESULTAT :")
if auc_mean < 0.55:
    print("   ✅ AUC ≈ 0.50 : Aucune dérive. Le Train et le Test sont indiscernables.")
    print("      Le modèle peut généraliser en toute sécurité.")
elif auc_mean < 0.70:
    print("   ⚠️ AUC entre 0.55 et 0.70 : Légère dérive possible, mais pas critique.")
else:
    print("   🚨 AUC > 0.70 : DÉRIVE MAJEURE ! Le Test est structurellement différent.")
    
    # Feature Importance si dérive
    clf_adv.fit(X_adv, target_adv)
    imp = pd.Series(clf_adv.feature_importances_, index=X_adv.columns).sort_values(ascending=False)
    print("\n   Top Features responsables du Drift :")
    print(imp.head(3))
