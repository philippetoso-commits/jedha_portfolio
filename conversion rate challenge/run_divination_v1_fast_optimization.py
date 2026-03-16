"""
DIVINATION V1 - OPTIMISATION RAPIDE DES POIDS (OOF + F1-aware)
Approche rapide : 1 seul 5-fold CV + optimisation sur OOF predictions
Temps estimé : ~5-7 minutes au lieu de 15-20 minutes
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
from scipy.optimize import minimize
import time

print("="*80)
print("🎯 DIVINATION V1 - OPTIMISATION RAPIDE DES POIDS")
print("="*80)
print("\n💡 MÉTHODE : OOF Predictions + Nelder-Mead Optimization")
print("   Optimisation directe sur les prédictions OOF (beaucoup plus rapide !)\n")

# Chemins absolus
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
TRAIN_PATH = os.path.join(SCRIPT_DIR, 'conversion_data_train.csv')
TEST_PATH  = os.path.join(SCRIPT_DIR, 'conversion_data_test.csv')

print("⏳ Chargement des données...")
train_data = pd.read_csv(TRAIN_PATH)
test_data  = pd.read_csv(TEST_PATH)

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

# Configuration du preprocessor
print("🔧 Configuration du pipeline...")
numeric_features = ['age', 'total_pages_visited', 'interaction_age_pages', 'pages_per_age']
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols)
    ],
    remainder='passthrough'
)

# Modèles de base (composition V1)
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

estimators = [
    ('xgb1', clf_xgb1),
    ('xgb2', clf_xgb2),
    ('lgbm', clf_lgbm),
    ('gb', clf_gb),
    ('logreg', clf_logreg)
]

print("✅ Pipeline configuré\n")

# ---------------------------------------------------------------------
# 1️⃣ Génération des OOF predictions (1 seul 5-fold CV)
# ---------------------------------------------------------------------
print("🔬 ÉTAPE 1 — Génération des prédictions OOF (5-Fold CV)")
print("="*80)

start_time = time.time()
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = {name: np.zeros(len(X)) for name, _ in estimators}

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
    print(f"   Fold {fold}/5...", end=" ", flush=True)
    
    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
    
    X_tr_p = preprocessor.fit_transform(X_tr)
    X_va_p = preprocessor.transform(X_va)
    
    for name, model in estimators:
        model.fit(X_tr_p, y_tr)
        oof_preds[name][va_idx] = model.predict_proba(X_va_p)[:, 1]
    
    print("✓")

oof_time = time.time() - start_time
print(f"\n✅ OOF terminé en {oof_time/60:.1f} minutes")

# Matrice des probabilités (N x 5)
P = np.column_stack([oof_preds[name] for name, _ in estimators])

# ---------------------------------------------------------------------
# 2️⃣ Optimisation des poids (F1 direct sur OOF)
# ---------------------------------------------------------------------
print("\n🚀 ÉTAPE 2 — Optimisation des poids (Nelder-Mead)")
print("="*80)

iteration_count = 0
best_f1_so_far = 0
best_weights_so_far = None

def f1_loss(weights):
    """Fonction à minimiser : -F1 (on veut maximiser F1)"""
    global iteration_count, best_f1_so_far, best_weights_so_far
    iteration_count += 1
    
    # Normaliser les poids
    weights = np.clip(weights, 0, None)
    weights_sum = weights.sum()
    if weights_sum == 0:
        return 1.0  # Pénalité si tous les poids sont 0
    weights = weights / weights_sum
    
    # Prédiction blended
    blended = P @ weights
    f1 = f1_score(y, (blended >= 0.5).astype(int))
    
    # Tracking
    if f1 > best_f1_so_far:
        best_f1_so_far = f1
        best_weights_so_far = weights.copy()
        print(f"   🎯 Itération {iteration_count}: Nouveau meilleur F1 = {f1:.5f}")
        print(f"      Poids: [{weights[0]:.3f}, {weights[1]:.3f}, {weights[2]:.3f}, {weights[3]:.3f}, {weights[4]:.3f}]")
    
    return -f1

# Poids initiaux (V1 empiriques)
initial_weights = np.array([0.7, 0.7, 1.2, 0.7, 0.5])

print(f"   Poids initiaux : {initial_weights}")
print(f"   Méthode : Nelder-Mead (simplex)")
print(f"   Max itérations : 300\n")

opt_start = time.time()
res = minimize(
    f1_loss,
    x0=initial_weights,
    method="Nelder-Mead",
    options={"maxiter": 300, "disp": False}
)
opt_time = time.time() - opt_start

# Normaliser les poids optimaux
opt_weights = np.clip(res.x, 0, None)
opt_weights /= opt_weights.sum()

print(f"\n✅ Optimisation terminée en {opt_time:.1f} secondes")
print(f"   Nombre d'itérations : {iteration_count}")

print(f"\n🏆 POIDS OPTIMAUX TROUVÉS :")
print(f"   XGBoost #1        : {opt_weights[0]:.3f}")
print(f"   XGBoost #2        : {opt_weights[1]:.3f}")
print(f"   LightGBM          : {opt_weights[2]:.3f}")
print(f"   GradientBoosting  : {opt_weights[3]:.3f}")
print(f"   LogisticReg       : {opt_weights[4]:.3f}")

# Comparaison avec V1
v1_weights = np.array([0.7, 0.7, 1.2, 0.7, 0.5])
v1_weights_norm = v1_weights / v1_weights.sum()

print(f"\n📊 COMPARAISON AVEC V1 :")
print(f"   V1 (empirique)  : [{v1_weights_norm[0]:.3f}, {v1_weights_norm[1]:.3f}, {v1_weights_norm[2]:.3f}, {v1_weights_norm[3]:.3f}, {v1_weights_norm[4]:.3f}]")
print(f"   V1 Optimisé     : [{opt_weights[0]:.3f}, {opt_weights[1]:.3f}, {opt_weights[2]:.3f}, {opt_weights[3]:.3f}, {opt_weights[4]:.3f}]")

# ---------------------------------------------------------------------
# 3️⃣ Optimisation du seuil
# ---------------------------------------------------------------------
print(f"\n🎯 ÉTAPE 3 — Optimisation du seuil")
print("="*80)

blended_oof = P @ opt_weights
best_f1 = 0
best_thr = 0

for thr in np.linspace(blended_oof.min(), blended_oof.max(), 1000):
    sc = f1_score(y, (blended_oof >= thr).astype(int))
    if sc > best_f1:
        best_f1 = sc
        best_thr = thr

final_preds = (blended_oof >= best_thr).astype(int)

metrics = {
    'f1': best_f1,
    'roc_auc': roc_auc_score(y, blended_oof),
    'precision': precision_score(y, final_preds),
    'recall': recall_score(y, final_preds),
    'accuracy': accuracy_score(y, final_preds),
    'threshold': best_thr
}

print(f"   Seuil optimal : {best_thr:.6f}")
print(f"\n{'='*70}")
print("🏆 RÉSULTATS FINAUX DIVINATION V1 (OPTIMISÉ)")
print(f"{'='*70}")
for k, v in metrics.items():
    if k != 'threshold':
        print(f"   {k.upper():<10} : {v:.5f}")
print(f"{'='*70}")

# Comparaison avec V1 original
v1_f1 = 0.77022
improvement = (metrics['f1'] - v1_f1) * 100

print(f"\n📈 AMÉLIORATION vs V1 :")
print(f"   V1 (empirique) : F1 = {v1_f1:.5f}")
print(f"   V1 Optimisé    : F1 = {metrics['f1']:.5f}")
print(f"   Δ              : {improvement:+.3f} points")

if metrics['f1'] > v1_f1:
    print(f"\n🎉 VICTOIRE ! L'optimisation a amélioré le modèle !")
elif abs(improvement) < 0.01:
    print(f"\n➡️ ÉGALITÉ : V1 était déjà quasi-optimal !")
else:
    print(f"\n➡️ V1 reste meilleur")

# ---------------------------------------------------------------------
# 4️⃣ Entraînement final et prédictions test
# ---------------------------------------------------------------------
print(f"\n🚀 ÉTAPE 4 — Entraînement final sur tout le dataset")
print("="*80)

# Dénormaliser les poids pour VotingClassifier
weights_for_voting = opt_weights * 10  # Multiplier par 10 pour avoir des valeurs plus lisibles

voting_final = VotingClassifier(
    estimators=estimators,
    voting="soft",
    weights=list(weights_for_voting),
    n_jobs=-1
)

pipeline_final = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", voting_final)
])

pipeline_final.fit(X, y)

test_proba = pipeline_final.predict_proba(X_test)[:, 1]
test_pred = (test_proba >= best_thr).astype(int)

print(f"✅ Entraînement terminé")
print(f"   Taux de conversion prédit : {test_pred.mean():.4%}")
print(f"   Conversions : {test_pred.sum()} / {len(test_pred)}")

# ---------------------------------------------------------------------
# 5️⃣ Sauvegarde
# ---------------------------------------------------------------------
submission_path = os.path.join(SCRIPT_DIR, 'divination_v1_fast_predictions.csv')
pd.DataFrame({"converted": test_pred}).to_csv(submission_path, index=False)

weights_df = pd.DataFrame({
    "Model": [name for name, _ in estimators],
    "V1_Weight": v1_weights,
    "V1_Optimized_Weight": opt_weights,
    "Difference": opt_weights - v1_weights_norm
})
weights_path = os.path.join(SCRIPT_DIR, 'divination_v1_fast_weights.csv')
weights_df.to_csv(weights_path, index=False)

print(f"\n💾 Fichiers générés :")
print(f"   - {submission_path}")
print(f"   - {weights_path}")

total_time = time.time() - start_time
print(f"\n⏱️  Temps total : {total_time/60:.1f} minutes")

print(f"\n{'='*70}")
print("🏆 DIVINATION V1 - OPTIMISATION RAPIDE TERMINÉE")
print(f"{'='*70}")
