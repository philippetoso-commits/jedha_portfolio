"""
DIVINATION V7 - OPTIMISATION AUTOMATIQUE DES POIDS
Utilise Bayesian Optimization pour trouver les poids optimaux scientifiquement
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score, accuracy_score, make_scorer
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import VotingClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from scipy.optimize import differential_evolution
import time

print("="*80)
print("🎯 DIVINATION V7 - OPTIMISATION AUTOMATIQUE DES POIDS")
print("="*80)
print("\n💡 MÉTHODE : Differential Evolution (Optimization Scientifique)")
print("   Recherche les poids optimaux pour maximiser le F1-Score\n")

# Chargement des données
print("⏳ Chargement des données...")
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

# Configuration du pipeline
print("🔧 Configuration du pipeline...")

numeric_features = ['age', 'total_pages_visited', 'interaction_age_pages', 'pages_per_age']
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols)
    ],
    remainder='passthrough'
)

# Les 5 modèles de base (composition V1)
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

# VotingClassifier
voting_model = VotingClassifier(
    estimators=[
        ('xgb1', clf_xgb1),
        ('xgb2', clf_xgb2),
        ('lgbm', clf_lgbm),
        ('gb', clf_gb),
        ('logreg', clf_logreg)
    ],
    voting='soft',
    weights=[0.7, 0.7, 1.2, 0.7, 0.5],  # Poids initiaux (V1)
    n_jobs=-1
)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', voting_model)
])

print("✅ Pipeline configuré\n")

# Fonction objectif pour l'optimisation
iteration_count = 0
best_f1_so_far = 0
best_weights_so_far = None

def objective_function(weights):
    """
    Fonction à minimiser (on minimise -F1 pour maximiser F1)
    """
    global iteration_count, best_f1_so_far, best_weights_so_far
    iteration_count += 1
    
    # Mettre à jour les poids
    pipeline.named_steps['classifier'].set_params(weights=list(weights))
    
    # Cross-validation 5-fold (plus rapide que 10-fold pour l'optimisation)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    f1_scorer = make_scorer(f1_score)
    
    scores = cross_val_score(
        pipeline, X, y,
        cv=skf,
        scoring=f1_scorer,
        n_jobs=-1
    )
    
    mean_f1 = scores.mean()
    
    # Tracking du meilleur
    if mean_f1 > best_f1_so_far:
        best_f1_so_far = mean_f1
        best_weights_so_far = weights.copy()
        print(f"   🎯 Itération {iteration_count}: Nouveau meilleur F1 = {mean_f1:.5f}")
        print(f"      Poids: [{weights[0]:.3f}, {weights[1]:.3f}, {weights[2]:.3f}, {weights[3]:.3f}, {weights[4]:.3f}]")
    else:
        if iteration_count % 5 == 0:
            print(f"   ⏳ Itération {iteration_count}: F1 = {mean_f1:.5f}")
    
    return -mean_f1  # Négatif car on minimise

# Optimisation avec Differential Evolution
print("🚀 DÉBUT DE L'OPTIMISATION")
print("="*80)
print("\n📊 Paramètres de recherche :")
print("   - XGBoost #1  : [0.3, 1.5]")
print("   - XGBoost #2  : [0.3, 1.5]")
print("   - LightGBM    : [0.5, 2.0]")
print("   - GradientBoosting : [0.3, 1.5]")
print("   - LogisticReg : [0.1, 1.0]")
print("\n   Méthode : Differential Evolution")
print("   Stratégie : best1bin (exploitation + exploration)")
print("   Générations max : 20")
print("   Population : 15 individus")
print("\n" + "="*80 + "\n")

# Définir les bornes pour chaque poids
bounds = [
    (0.3, 1.5),  # XGBoost #1
    (0.3, 1.5),  # XGBoost #2
    (0.5, 2.0),  # LightGBM (peut être plus fort)
    (0.3, 1.5),  # GradientBoosting
    (0.1, 1.0)   # LogisticRegression (généralement plus faible)
]

start_time = time.time()

# Lancer l'optimisation
result = differential_evolution(
    objective_function,
    bounds,
    strategy='best1bin',
    maxiter=20,
    popsize=15,
    tol=0.0001,
    mutation=(0.5, 1),
    recombination=0.7,
    seed=42,
    workers=1,
    updating='deferred',
    polish=True
)

elapsed_time = time.time() - start_time

print("\n" + "="*80)
print("✅ OPTIMISATION TERMINÉE")
print("="*80)
print(f"\n⏱️  Temps total : {elapsed_time/60:.1f} minutes")
print(f"🔢 Nombre d'évaluations : {iteration_count}")

# Résultats
optimal_weights = result.x
optimal_f1 = -result.fun

print(f"\n🏆 POIDS OPTIMAUX TROUVÉS :")
print(f"   XGBoost #1        : {optimal_weights[0]:.3f}")
print(f"   XGBoost #2        : {optimal_weights[1]:.3f}")
print(f"   LightGBM          : {optimal_weights[2]:.3f}")
print(f"   GradientBoosting  : {optimal_weights[3]:.3f}")
print(f"   LogisticReg       : {optimal_weights[4]:.3f}")
print(f"\n   F1-Score (5-Fold CV) : {optimal_f1:.5f}")

# Comparaison avec V1
v1_weights = [0.7, 0.7, 1.2, 0.7, 0.5]
print(f"\n📊 COMPARAISON AVEC V1 :")
print(f"   V1 (empirique)  : [{v1_weights[0]}, {v1_weights[1]}, {v1_weights[2]}, {v1_weights[3]}, {v1_weights[4]}]")
print(f"   V7 (optimisé)   : [{optimal_weights[0]:.3f}, {optimal_weights[1]:.3f}, {optimal_weights[2]:.3f}, {optimal_weights[3]:.3f}, {optimal_weights[4]:.3f}]")

# Validation finale avec 10-Fold CV
print(f"\n🔬 VALIDATION FINALE (10-Fold CV)...")
pipeline.named_steps['classifier'].set_params(weights=list(optimal_weights))

skf_final = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
oof_predictions = np.zeros(len(X))

for fold, (train_idx, val_idx) in enumerate(skf_final.split(X, y), 1):
    print(f"   Fold {fold}/10...", end=" ", flush=True)
    
    X_train_fold = X.iloc[train_idx]
    y_train_fold = y.iloc[train_idx]
    X_val_fold = X.iloc[val_idx]
    
    pipeline.fit(X_train_fold, y_train_fold)
    val_proba = pipeline.predict_proba(X_val_fold)[:, 1]
    oof_predictions[val_idx] = val_proba
    
    fold_f1 = f1_score(y.iloc[val_idx], (val_proba >= 0.5).astype(int))
    print(f"F1: {fold_f1:.5f}")

# Optimisation du seuil
print(f"\n🎯 Optimisation du seuil...")
best_f1 = 0
best_threshold = 0

for threshold in np.linspace(oof_predictions.min(), oof_predictions.max(), 1000):
    preds = (oof_predictions >= threshold).astype(int)
    score = f1_score(y, preds)
    if score > best_f1:
        best_f1 = score
        best_threshold = threshold

final_predictions = (oof_predictions >= best_threshold).astype(int)

metrics_v7 = {
    'f1': best_f1,
    'roc_auc': roc_auc_score(y, oof_predictions),
    'precision': precision_score(y, final_predictions),
    'recall': recall_score(y, final_predictions),
    'accuracy': accuracy_score(y, final_predictions),
    'threshold': best_threshold
}

print(f"   Seuil optimal : {best_threshold:.6f}")
print(f"\n{'='*70}")
print("🏆 RÉSULTATS FINAUX DIVINATION V7")
print(f"{'='*70}")
print(f"   F1-Score  : {metrics_v7['f1']:.5f}")
print(f"   ROC-AUC   : {metrics_v7['roc_auc']:.5f}")
print(f"   Precision : {metrics_v7['precision']:.5f}")
print(f"   Recall    : {metrics_v7['recall']:.5f}")
print(f"   Accuracy  : {metrics_v7['accuracy']:.5f}")
print(f"{'='*70}")

# Comparaison avec V1
v1_f1 = 0.77022
improvement = (metrics_v7['f1'] - v1_f1) * 100

print(f"\n📈 AMÉLIORATION vs V1 :")
print(f"   V1 (empirique) : F1 = {v1_f1:.5f}")
print(f"   V7 (optimisé)  : F1 = {metrics_v7['f1']:.5f}")
print(f"   Δ              : {improvement:+.3f} points")

if metrics_v7['f1'] > v1_f1:
    print(f"\n🎉 VICTOIRE ! L'optimisation a amélioré le modèle !")
elif abs(improvement) < 0.01:
    print(f"\n➡️ ÉGALITÉ : V1 était déjà quasi-optimal !")
else:
    print(f"\n➡️ V1 reste meilleur (peut-être overfitting sur 5-fold)")

# Entraînement final et prédictions
print(f"\n🚀 Entraînement du modèle final sur tout le dataset...\n")

pipeline.fit(X, y)
test_proba = pipeline.predict_proba(X_test)[:, 1]
test_predictions = (test_proba >= best_threshold).astype(int)

submission = pd.DataFrame({'converted': test_predictions})
submission.to_csv('divination_v7_predictions.csv', index=False)

print("✅ Prédictions générées : divination_v7_predictions.csv")
print(f"   Taux de conversion prédit : {test_predictions.mean():.4%}")
print(f"   Nombre de conversions : {test_predictions.sum()} / {len(test_predictions)}")

# Sauvegarder les poids optimaux
weights_df = pd.DataFrame({
    'Model': ['XGBoost #1', 'XGBoost #2', 'LightGBM', 'GradientBoosting', 'LogisticRegression'],
    'V1_Weight': v1_weights,
    'V7_Optimal_Weight': optimal_weights,
    'Difference': optimal_weights - np.array(v1_weights)
})

weights_df.to_csv('optimal_weights_v7.csv', index=False)
print(f"\n💾 Poids optimaux sauvegardés : optimal_weights_v7.csv")

print(f"\n{'='*70}")
print("🏆 DIVINATION V7 - OPTIMISATION AUTOMATIQUE TERMINÉE")
print(f"{'='*70}")
