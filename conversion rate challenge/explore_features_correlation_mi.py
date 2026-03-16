"""
explore_features_correlation_mi.py
Analyse des corrélations (Spearman) et de l'information mutuelle (MI) entre les features et la cible.
"""

import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif

# ------------------------------------------------------------
# Chemins absolus des données (même répertoire que le script)
# ------------------------------------------------------------
from pathlib import Path
SCRIPT_DIR = Path.cwd()
TRAIN_PATH = SCRIPT_DIR / 'conversion_data_train.csv'

print("🔍 Chargement du jeu d'entraînement...")
train_df = pd.read_csv(TRAIN_PATH)

# ------------------------------------------------------------
# Feature engineering (identique à celui utilisé dans les modèles)
# ------------------------------------------------------------
def feature_engineering(df):
    df = df.copy()
    df['is_active'] = (df['total_pages_visited'] > 2).astype(int)
    df['interaction_age_pages'] = df['age'] * df['total_pages_visited']
    df['pages_per_age'] = df['total_pages_visited'] / (df['age'] + 0.1)
    return df

X = feature_engineering(train_df.drop('converted', axis=1))
y = train_df['converted']

# ------------------------------------------------------------
# Encodage des variables catégorielles (country, source)
# ------------------------------------------------------------
categorical_cols = ['country', 'source']
for col in categorical_cols:
    le = LabelEncoder()
    le.fit(pd.concat([X[col], train_df[col]]))  # même encodage que le pipeline
    X[col] = le.transform(X[col])

print(f"✅ Dataset préparé : {X.shape[0]} lignes, {X.shape[1]} colonnes")

# ------------------------------------------------------------
# 1️⃣ Corrélation (Spearman) entre chaque feature et la cible
# ------------------------------------------------------------
print("\n📊 Calcul de la corrélation (Spearman) avec la cible...")
correlations = X.apply(lambda col: col.corr(y, method='spearman')).rename('spearman_corr')
correlations = correlations.sort_values(ascending=False)
print(correlations.head(10))

# Sauvegarde de la corrélation complète
corr_path = os.path.join(SCRIPT_DIR, 'feature_correlations.csv')
correlations.to_csv(corr_path, header=True)
print(f"✅ Corrélations sauvegardées → {corr_path}")

# ------------------------------------------------------------
# 2️⃣ Information mutuelle (MI) entre chaque feature et la cible
# ------------------------------------------------------------
print("\n🔎 Calcul de l'information mutuelle (MI) avec la cible...")
mi = mutual_info_classif(X, y, discrete_features='auto')
mi_series = pd.Series(mi, index=X.columns, name='mutual_info')
mi_series = mi_series.sort_values(ascending=False)
print(mi_series.head(10))

# Sauvegarde de la MI complète
mi_path = os.path.join(SCRIPT_DIR, 'feature_mutual_information.csv')
mi_series.to_csv(mi_path, header=True)
print(f"✅ Information mutuelle sauvegardée → {mi_path}")

# ------------------------------------------------------------
# Résumé des top‑5 features selon chaque critère
# ------------------------------------------------------------
summary = pd.DataFrame({
    'Spearman': correlations.head(5),
    'MutualInfo': mi_series.head(5)
})
summary_path = os.path.join(SCRIPT_DIR, 'top_features_summary.csv')
summary.to_csv(summary_path)
print(f"✅ Résumé top‑5 sauvegardé → {summary_path}")

print("\n🎉 Analyse terminée ! Vous pouvez visualiser les CSV générés avec votre outil préféré.")
