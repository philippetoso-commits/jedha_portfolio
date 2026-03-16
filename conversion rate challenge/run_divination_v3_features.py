"""
DIVINATION V3 - FEATURE EDITION
Base : Divination V1 (poids optimisés)
Nouveauté : Feature Engineering avancé basé sur l'analyse de corrélation
"""

import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import VotingClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

print("="*80)
print("🎯 DIVINATION V3 - FEATURE EDITION")
print("="*80)
print("\n💡 STRATÉGIE :")
print("   base: Divination V1 (Champion actuel)")
print("   + Feature: pages_relative_to_country (Normalisation par pays)")
print("   + Feature: new_user_pages (Interaction)")
print("   + Feature: is_hyper_active (>10 pages)\n")

# Chemins absolus
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
TRAIN_PATH = os.path.join(SCRIPT_DIR, 'conversion_data_train.csv')
TEST_PATH  = os.path.join(SCRIPT_DIR, 'conversion_data_test.csv')

print("⏳ Chargement des données...")
train_data = pd.read_csv(TRAIN_PATH)
test_data  = pd.read_csv(TEST_PATH)

def advanced_feature_engineering(df, train_stats=None):
    df_eng = df.copy()
    
    # 1. Features de base (V1)
    df_eng['is_active'] = (df_eng['total_pages_visited'] > 2).astype(int)
    df_eng['interaction_age_pages'] = df_eng['age'] * df_eng['total_pages_visited']
    df_eng['pages_per_age'] = df_eng['total_pages_visited'] / (df_eng['age'] + 0.1)
    
    # 2. Nouvelles Features (V3)
    # Interaction New User * Pages
    df_eng['new_user_pages'] = df_eng['new_user'] * df_eng['total_pages_visited']
    
    # Is Hyper Active (le seuil de 10 est empirique, basé sur l'analyse)
    df_eng['is_hyper_active'] = (df_eng['total_pages_visited'] > 12).astype(int)
    
    # Pages relatives à la moyenne du pays
    # Si on est en train (train_stats=None), on calcule les moyennes
    # Si on est en test (train_stats existe), on utilise les moyennes du train pour éviter le data leakage
    if train_stats is None:
        country_means = df_eng.groupby('country')['total_pages_visited'].mean()
        train_stats = {'country_means': country_means}
    
    # Appliquer les moyennes
    means = train_stats['country_means']
    df_eng['country_mean_pages'] = df_eng['country'].map(means).fillna(means.mean())
    df_eng['pages_relative_to_country'] = df_eng['total_pages_visited'] / df_eng['country_mean_pages']
    
    return df_eng, train_stats

# Feature Engineering sur Train
print("🔧 Création des nouvelles features...")
X, train_stats = advanced_feature_engineering(train_data.drop('converted', axis=1))
y = train_data['converted']

# Feature Engineering sur Test (avec stats du train)
X_test, _ = advanced_feature_engineering(test_data, train_stats)

categorical_cols = ['country', 'source']
X_encoded = X.copy()

for col in categorical_cols:
    le = LabelEncoder()
    combined = pd.concat([X[col], X_test[col]])
    le.fit(combined)
    X_encoded[col] = le.transform(X[col])
    # Note: On garde X et X_test originaux pour le pipeline qui a son propre OneHotEncoder

print(f"✅ Dataset prêt : {len(X)} lignes")
print(f"   Nouvelles colonnes : {list(X.columns)}")

print("\n🔧 Configuration du pipeline...")
# Mise à jour de la liste des features numériques
numeric_features = [
    'age', 'total_pages_visited', 
    'interaction_age_pages', 'pages_per_age',
    'new_user_pages', 'is_hyper_active', 'pages_relative_to_country'
]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols)
    ],
    remainder='passthrough'
)

# Modèles de base (V1 Composition)
clf_xgb1 = XGBClassifier(
    n_estimators=350, max_depth=4, learning_rate=0.05,
    subsample=0.9, colsample_bytree=0.9, eval_metric='logloss',
    random_state=42, n_jobs=1
)

clf_xgb2 = XGBClassifier(
    n_estimators=350, max_depth=4, learning_rate=0.05,
    subsample=0.9, colsample_bytree=0.9, eval_metric='logloss',
    random_state=2025, n_jobs=1
)

clf_lgbm = LGBMClassifier(
    n_estimators=350, max_depth=4, learning_rate=0.05,
    subsample=0.9, colsample_bytree=0.9, verbose=-1,
    random_state=42, n_jobs=1
)

clf_gb = GradientBoostingClassifier(
    n_estimators=350, max_depth=4, learning_rate=0.05,
    subsample=0.9, random_state=42
)

clf_logreg = LogisticRegression(
    max_iter=1000, class_weight={0: 1, 1: 80},
    solver='lbfgs', n_jobs=1
)

# VotingClassifier V1 (Champion)
voting_model = VotingClassifier(
    estimators=[
        ('xgb1', clf_xgb1),
        ('xgb2', clf_xgb2),
        ('lgbm', clf_lgbm),
        ('gb', clf_gb),
        ('logreg', clf_logreg)
    ],
    voting='soft',
    weights=[0.7, 0.7, 1.2, 0.7, 0.5],  # Poids V1
    n_jobs=-1
)

pipeline_v3 = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', voting_model)
])

print("\n⏳ Entraînement en 10-Fold CV (Test ultime)...")

# Cross-validation
n_folds = 10
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

oof_predictions = np.zeros(len(X))
fold_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    print(f"   Fold {fold}/{n_folds}...", end=" ", flush=True)
    
    X_train_fold = X.iloc[train_idx]
    y_train_fold = y.iloc[train_idx]
    X_val_fold = X.iloc[val_idx]
    y_val_fold = y.iloc[val_idx]
    
    pipeline_v3.fit(X_train_fold, y_train_fold)
    val_proba = pipeline_v3.predict_proba(X_val_fold)[:, 1]
    oof_predictions[val_idx] = val_proba
    
    fold_f1 = f1_score(y_val_fold, (val_proba >= 0.5).astype(int))
    fold_scores.append(fold_f1)
    print(f"F1: {fold_f1:.5f}")

print(f"\n✅ Cross-validation terminée")
print(f"   F1 moyen par fold : {np.mean(fold_scores):.5f} ± {np.std(fold_scores):.5f}")

# Optimisation du seuil
print("\n🎯 Optimisation du seuil...")
best_f1 = 0
best_threshold = 0

for threshold in np.linspace(oof_predictions.min(), oof_predictions.max(), 1000):
    preds = (oof_predictions >= threshold).astype(int)
    score = f1_score(y, preds)
    if score > best_f1:
        best_f1 = score
        best_threshold = threshold

final_predictions = (oof_predictions >= best_threshold).astype(int)

metrics_v3 = {
    'f1': best_f1,
    'roc_auc': roc_auc_score(y, oof_predictions),
    'precision': precision_score(y, final_predictions),
    'recall': recall_score(y, final_predictions),
    'accuracy': accuracy_score(y, final_predictions),
    'threshold': best_threshold
}

print(f"   Seuil optimal : {best_threshold:.6f}")
print(f"\n{'='*70}")
print("🏆 RÉSULTATS DIVINATION V3 (FEATURES)")
print(f"{'='*70}")
print(f"   F1-Score  : {metrics_v3['f1']:.5f}")
print(f"   ROC-AUC   : {metrics_v3['roc_auc']:.5f}")
print(f"   Precision : {metrics_v3['precision']:.5f}")
print(f"   Recall    : {metrics_v3['recall']:.5f}")
print(f"   Accuracy  : {metrics_v3['accuracy']:.5f}")
print(f"{'='*70}")

# Comparaison avec V1
v1_f1 = 0.77022
improvement = (metrics_v3['f1'] - v1_f1) * 100

print(f"\n📈 COMPARAISON vs V1 :")
print(f"   V1 (Original)      : F1 = {v1_f1:.5f}")
print(f"   V3 (New Features)  : F1 = {metrics_v3['f1']:.5f}")
print(f"   Δ                  : {improvement:+.3f} points")

if metrics_v3['f1'] > v1_f1:
    print(f"\n🎉 VICTOIRE ! Les nouvelles features font la différence !")
elif abs(improvement) < 0.01:
    print(f"\n➡️ ÉGALITÉ : Pas de gain significatif.")
else:
    print(f"\n➡️ V1 reste meilleur : Les nouvelles features introduisent peut-être du bruit.")

# Entraînement final et prédictions
print(f"\n🚀 Entraînement du modèle final sur tout le dataset...\n")

pipeline_v3.fit(X, y)
test_proba = pipeline_v3.predict_proba(X_test)[:, 1]
test_predictions = (test_proba >= best_threshold).astype(int)

submission = pd.DataFrame({'converted': test_predictions})
submission_path = os.path.join(SCRIPT_DIR, 'divination_v3_features_predictions.csv')
submission.to_csv(submission_path, index=False)

print(f"✅ Prédictions générées : {submission_path}")
print(f"   Taux de conversion prédit : {test_predictions.mean():.4%}")
print(f"   Nombre de conversions : {test_predictions.sum()} / {len(test_predictions)}")

print(f"\n{'='*70}")
print("🏆 DIVINATION V3 - TERMINÉ")
print(f"{'='*70}")
