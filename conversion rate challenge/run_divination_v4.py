"""
DIVINATION V4 - SNIPER ÉLITE OPTIMISÉ
Stratégie : Remplacer XGBoost #2 (redondant) par Ultimate Mix
Conserver LogisticRegression pour la diversité linéaire
"""

import pandas as pd
import numpy as np
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
from ultimate_mix_wrapper import UltimateMixClassifier

print("="*80)
print("🎯 DIVINATION V4 - SNIPER ÉLITE OPTIMISÉ")
print("="*80)
print("\n💡 STRATÉGIE :")
print("   ❌ Retiré  : XGBoost #2 (seed=2025) - redondance")
print("   ✅ Ajouté  : Ultimate Mix (Poisson+LogLoss)")
print("   ✅ Conservé: LogisticRegression - diversité linéaire\n")

print("⏳ Chargement des données...")

# Chargement des données
train_data = pd.read_csv('conversion_data_train.csv')
test_data = pd.read_csv('conversion_data_test.csv')

def feature_engineering(df):
    df_eng = df.copy()
    df_eng['is_active'] = (df_eng['total_pages_visited'] > 2).astype(int)
    df_eng['interaction_age_pages'] = df_eng['age'] * df_eng['total_pages_visited']
    df_eng['pages_per_age'] = df_eng['total_pages_visited'] / (df_eng['age'] + 0.1)
    return df_eng

X = feature_engineering(train_data.drop('converted', axis=1))
y = train_data['converted']
X_test = feature_engineering(test_data)

# Encodage
categorical_cols = ['country', 'source']
X_encoded = X.copy()
X_test_encoded = X_test.copy()

for col in categorical_cols:
    le = LabelEncoder()
    combined = pd.concat([X[col], X_test[col]])
    le.fit(combined)
    X_encoded[col] = le.transform(X[col])
    X_test_encoded[col] = le.transform(X_test[col])

print(f"✅ Dataset prêt : {len(X)} lignes\n")

# Preprocessing
print("🔧 Configuration du pipeline...")
numeric_features = ['age', 'total_pages_visited', 'interaction_age_pages', 'pages_per_age']
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols)
    ],
    remainder='passthrough'
)

# Les 5 modèles - COMPOSITION OPTIMISÉE
clf_xgb = XGBClassifier(
    n_estimators=350, max_depth=4, learning_rate=0.05,
    subsample=0.9, colsample_bytree=0.9, eval_metric='logloss',
    random_state=42, n_jobs=1
)

clf_ultimate = UltimateMixClassifier(
    learning_rate=0.05, max_iter=500, max_depth=8,
    l2_regularization=0.1, random_state=42
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

# VotingClassifier avec composition optimisée
voting_model_v4 = VotingClassifier(
    estimators=[
        ('xgb', clf_xgb),                    # Un seul XGBoost
        ('ultimate_mix', clf_ultimate),      # 🆕 Remplace XGBoost #2
        ('lgbm', clf_lgbm),
        ('gb', clf_gb),
        ('logreg', clf_logreg)               # ✅ Conservé !
    ],
    voting='soft',
    weights=[0.7, 1.0, 1.2, 0.7, 0.5],      # Ultimate Mix: poids 1.0
    n_jobs=-1
)

pipeline_v4 = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', voting_model_v4)
])

print("📊 Configuration V4 :")
print("   - XGBoost (seed=42)        : poids 0.7")
print("   - Ultimate Mix (Poi+Log)   : poids 1.0 🆕")
print("   - LightGBM                 : poids 1.2 ⭐")
print("   - GradientBoosting         : poids 0.7")
print("   - LogisticRegression       : poids 0.5 ✅")
print("\n🎯 Diversité maximale :")
print("   • Linéaire    : LogisticRegression")
print("   • Arbres      : XGBoost, LightGBM, GradientBoosting")
print("   • Hybride     : Ultimate Mix (Régression + Classification)")
print("\n⏳ Entraînement en 10-Fold CV (peut prendre 3-4 minutes)...\n")

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
    
    # Entraînement
    pipeline_v4.fit(X_train_fold, y_train_fold)
    
    # Prédictions
    val_proba = pipeline_v4.predict_proba(X_val_fold)[:, 1]
    oof_predictions[val_idx] = val_proba
    
    # Score du fold
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

# Prédictions finales
final_predictions = (oof_predictions >= best_threshold).astype(int)

# Métriques
metrics_v4 = {
    'f1': best_f1,
    'roc_auc': roc_auc_score(y, oof_predictions),
    'precision': precision_score(y, final_predictions),
    'recall': recall_score(y, final_predictions),
    'accuracy': accuracy_score(y, final_predictions),
    'threshold': best_threshold
}

print(f"   Seuil optimal : {best_threshold:.6f}")
print(f"\n{'='*70}")
print("🏆 RÉSULTATS DIVINATION V4")
print(f"{'='*70}")
print(f"   F1-Score  : {metrics_v4['f1']:.5f}")
print(f"   ROC-AUC   : {metrics_v4['roc_auc']:.5f}")
print(f"   Precision : {metrics_v4['precision']:.5f}")
print(f"   Recall    : {metrics_v4['recall']:.5f}")
print(f"   Accuracy  : {metrics_v4['accuracy']:.5f}")
print(f"{'='*70}")

# Comparaison avec V1 et V3
metrics_v1 = {
    'f1': 0.77022,
    'roc_auc': 0.98577,
    'precision': 0.81007,
    'recall': 0.73410,
    'accuracy': 0.98587
}

metrics_v3 = {
    'f1': 0.76948,
    'roc_auc': 0.98545,
    'precision': 0.81970,
    'recall': 0.72505,
    'accuracy': 0.98599
}

print("\n" + "="*90)
print("📊 COMPARAISON COMPLÈTE : V1 vs V3 vs V4")
print("="*90)
print(f"\n{'Métrique':<15} {'V1 (Original)':<20} {'V3 (Ult-LogReg)':<20} {'V4 (Ult-XGB2)':<20}")
print("-"*90)

for metric in ['f1', 'roc_auc', 'precision', 'recall', 'accuracy']:
    v1_val = metrics_v1[metric]
    v3_val = metrics_v3[metric]
    v4_val = metrics_v4[metric]
    
    # Trouver le meilleur
    best_val = max(v1_val, v3_val, v4_val)
    
    v1_str = f"{v1_val:.5f}" + (" ⭐" if v1_val == best_val else "")
    v3_str = f"{v3_val:.5f}" + (" ⭐" if v3_val == best_val else "")
    v4_str = f"{v4_val:.5f}" + (" ⭐" if v4_val == best_val else "")
    
    print(f"{metric.upper():<15} {v1_str:<20} {v3_str:<20} {v4_str:<20}")

print("\n" + "="*90)

# Verdict
all_f1 = [
    ('V1 (Original)', metrics_v1['f1']),
    ('V3 (Ultimate Mix remplace LogReg)', metrics_v3['f1']),
    ('V4 (Ultimate Mix remplace XGB2)', metrics_v4['f1'])
]
all_f1_sorted = sorted(all_f1, key=lambda x: x[1], reverse=True)

print(f"\n🏆 CLASSEMENT F1-Score :")
for i, (name, score) in enumerate(all_f1_sorted, 1):
    emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉"
    print(f"   {emoji} {i}. {name:<40} F1 = {score:.5f}")

winner_name, winner_score = all_f1_sorted[0]

if winner_name.startswith('V4'):
    improvement_v1 = (metrics_v4['f1'] - metrics_v1['f1']) * 100
    improvement_v3 = (metrics_v4['f1'] - metrics_v3['f1']) * 100
    print(f"\n🎉 VICTOIRE V4 !")
    print(f"   Amélioration vs V1 : +{improvement_v1:.3f} points")
    print(f"   Amélioration vs V3 : +{improvement_v3:.3f} points")
    print(f"\n🚀 RECOMMANDATION : Soumettre Divination V4 sur Kaggle !")
elif winner_name.startswith('V1'):
    print(f"\n➡️ V1 reste champion avec F1 = {winner_score:.5f}")
    if metrics_v4['f1'] > metrics_v3['f1']:
        print(f"   Mais V4 bat V3 ! Meilleure stratégie que de remplacer LogReg.")
else:
    print(f"\n➡️ V3 est champion avec F1 = {winner_score:.5f}")

print("\n" + "="*90)

# Entraînement final et prédictions
print("\n🚀 Entraînement du modèle final sur tout le dataset...\n")

pipeline_v4.fit(X, y)

test_proba = pipeline_v4.predict_proba(X_test)[:, 1]
test_predictions = (test_proba >= best_threshold).astype(int)

submission = pd.DataFrame({
    'converted': test_predictions
})

submission.to_csv('divination_v4_predictions.csv', index=False)

print("✅ Prédictions générées : divination_v4_predictions.csv")
print(f"   Taux de conversion prédit : {test_predictions.mean():.4%}")
print(f"   Nombre de conversions : {test_predictions.sum()} / {len(test_predictions)}")
print(f"\n🎯 Seuil utilisé : {best_threshold:.6f}")
print(f"\n{'='*70}")
print("🏆 DIVINATION V4 - PRÊT POUR SOUMISSION")
print(f"{'='*70}")
