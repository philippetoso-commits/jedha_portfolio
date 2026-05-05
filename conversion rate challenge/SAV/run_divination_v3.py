"""
Script pour exécuter Divination V3 et comparer avec V1
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
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from ultimate_mix_wrapper import UltimateMixClassifier

print("="*80)
print("🎯 DIVINATION V3 - SNIPER ÉLITE ULTIME")
print("="*80)
print("\n⏳ Chargement des données...")

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

# Les 5 modèles
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

clf_ultimate = UltimateMixClassifier(
    learning_rate=0.05, max_iter=500, max_depth=8,
    l2_regularization=0.1, random_state=42
)

# VotingClassifier
voting_model_v3 = VotingClassifier(
    estimators=[
        ('xgb1', clf_xgb1),
        ('xgb2', clf_xgb2),
        ('lgbm', clf_lgbm),
        ('gb', clf_gb),
        ('ultimate_mix', clf_ultimate)
    ],
    voting='soft',
    weights=[0.7, 0.7, 1.2, 0.7, 1.0],
    n_jobs=-1
)

pipeline_v3 = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', voting_model_v3)
])

print("📊 Configuration :")
print("   - XGBoost #1 (seed=42)     : poids 0.7")
print("   - XGBoost #2 (seed=2025)   : poids 0.7")
print("   - LightGBM                 : poids 1.2 ⭐")
print("   - GradientBoosting         : poids 0.7")
print("   - Ultimate Mix (Poi+Log)   : poids 1.0 🆕")
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
    pipeline_v3.fit(X_train_fold, y_train_fold)
    
    # Prédictions
    val_proba = pipeline_v3.predict_proba(X_val_fold)[:, 1]
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
print("🏆 RÉSULTATS DIVINATION V3")
print(f"{'='*70}")
print(f"   F1-Score  : {metrics_v3['f1']:.5f}")
print(f"   ROC-AUC   : {metrics_v3['roc_auc']:.5f}")
print(f"   Precision : {metrics_v3['precision']:.5f}")
print(f"   Recall    : {metrics_v3['recall']:.5f}")
print(f"   Accuracy  : {metrics_v3['accuracy']:.5f}")
print(f"{'='*70}")

# Comparaison avec V1
metrics_v1 = {
    'f1': 0.77022,
    'roc_auc': 0.98577,
    'precision': 0.81007,
    'recall': 0.73410,
    'accuracy': 0.98587
}

print("\n" + "="*80)
print("📊 COMPARAISON DIVINATION V1 vs V3")
print("="*80)
print(f"\n{'Métrique':<15} {'V1 (Original)':<20} {'V3 (Ultimate Mix)':<20} {'Δ':<15}")
print("-"*80)

for metric in ['f1', 'roc_auc', 'precision', 'recall', 'accuracy']:
    v1_val = metrics_v1[metric]
    v3_val = metrics_v3[metric]
    delta = v3_val - v1_val
    delta_str = f"{delta:+.5f}" if delta != 0 else "="
    emoji = "📈" if delta > 0 else "📉" if delta < 0 else "➡️"
    
    print(f"{metric.upper():<15} {v1_val:<20.5f} {v3_val:<20.5f} {emoji} {delta_str}")

print("\n" + "="*80)

# Verdict
if metrics_v3['f1'] > metrics_v1['f1']:
    improvement = (metrics_v3['f1'] - metrics_v1['f1']) * 100
    print(f"\n🎉 VICTOIRE ! Divination V3 améliore le F1-Score de {improvement:.3f} points !")
    print(f"   Nouveau champion : F1 = {metrics_v3['f1']:.5f}")
    print(f"\n🚀 RECOMMANDATION : Soumettre Divination V3 sur Kaggle !")
elif metrics_v3['f1'] == metrics_v1['f1']:
    print(f"\n➡️ ÉGALITÉ : Les deux versions ont le même F1-Score ({metrics_v3['f1']:.5f})")
    print(f"   Analyser les différences de Précision/Rappel pour choisir")
else:
    decline = (metrics_v1['f1'] - metrics_v3['f1']) * 100
    print(f"\n⚠️ ATTENTION : V3 est légèrement inférieur de {decline:.3f} points")
    print(f"   V1 reste champion : F1 = {metrics_v1['f1']:.5f}")
    print(f"   Mais V3 peut avoir de meilleurs Précision/Rappel pour le LB")

print("\n" + "="*80)

# Entraînement final et prédictions
print("\n🚀 Entraînement du modèle final sur tout le dataset...\n")

pipeline_v3.fit(X, y)

test_proba = pipeline_v3.predict_proba(X_test)[:, 1]
test_predictions = (test_proba >= best_threshold).astype(int)

submission = pd.DataFrame({
    'converted': test_predictions
})

submission.to_csv('divination_v3_predictions.csv', index=False)

print("✅ Prédictions générées : divination_v3_predictions.csv")
print(f"   Taux de conversion prédit : {test_predictions.mean():.4%}")
print(f"   Nombre de conversions : {test_predictions.sum()} / {len(test_predictions)}")
print(f"\n🎯 Seuil utilisé : {best_threshold:.6f}")
print(f"\n{'='*70}")
print("🏆 DIVINATION V3 - PRÊT POUR SOUMISSION")
print(f"{'='*70}")
