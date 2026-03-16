# =============================================================================
# DIVINATION V1.5 — OPTIMISATION DES POIDS PAR OOF (SINGLE CELL)
# =============================================================================

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline
from scipy.optimize import minimize

print("="*80)
print("🎯 DIVINATION V1.5 — OOF WEIGHT OPTIMIZATION (SINGLE CELL)")
print("="*80)

# ---------------------------------------------------------------------
# ⚠️ PRÉREQUIS ATTENDUS DANS LE KERNEL
# ---------------------------------------------------------------------
# - X, y
# - X_test
# - preprocessor
# - clf_xgb1, clf_xgb2, clf_lgbm, clf_gb, clf_logreg
# ---------------------------------------------------------------------

estimators = [
    ('xgb1', clf_xgb1),
    ('xgb2', clf_xgb2),
    ('lgbm', clf_lgbm),
    ('gb', clf_gb),
    ('logreg', clf_logreg)
]

# ---------------------------------------------------------------------
# 1️⃣ GÉNÉRATION DES PRÉDICTIONS OOF
# ---------------------------------------------------------------------
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_preds = {name: np.zeros(len(X)) for name, _ in estimators}

print("\n🔬 Génération des prédictions OOF...\n")

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
    print(f"   Fold {fold}/5")

    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

    X_tr_p = preprocessor.fit_transform(X_tr)
    X_va_p = preprocessor.transform(X_va)

    for name, model in estimators:
        m = model
        m.fit(X_tr_p, y_tr)
        oof_preds[name][va_idx] = m.predict_proba(X_va_p)[:, 1]

# ---------------------------------------------------------------------
# 2️⃣ MATRICE DES PROBABILITÉS
# ---------------------------------------------------------------------
P = np.column_stack([oof_preds[name] for name, _ in estimators])

# ---------------------------------------------------------------------
# 3️⃣ OPTIMISATION DES POIDS (F1 DIRECT)
# ---------------------------------------------------------------------
def f1_loss(weights):
    weights = np.clip(weights, 0, None)
    weights = weights / weights.sum()
    blended = P @ weights
    return -f1_score(y, blended >= 0.5)

print("\n🚀 Optimisation des poids...")

res = minimize(
    f1_loss,
    x0=np.ones(P.shape[1]),
    method='Nelder-Mead',
    options={'maxiter': 300, 'disp': True}
)

opt_weights = np.clip(res.x, 0, None)
opt_weights /= opt_weights.sum()

print("\n🏆 POIDS OPTIMAUX (NORMALISÉS) :")
for (name, _), w in zip(estimators, opt_weights):
    print(f"   {name:<10} : {w:.4f}")

# ---------------------------------------------------------------------
# 4️⃣ OPTIMISATION DU SEUIL
# ---------------------------------------------------------------------
print("\n🎯 Optimisation du seuil...")

blended_oof = P @ opt_weights
best_f1, best_thr = 0, 0

for thr in np.linspace(0.05, 0.95, 500):
    sc = f1_score(y, (blended_oof >= thr).astype(int))
    if sc > best_f1:
        best_f1, best_thr = sc, thr

print(f"   Seuil optimal : {best_thr:.4f}")
print(f"   F1 OOF final  : {best_f1:.5f}")

# ---------------------------------------------------------------------
# 5️⃣ MODÈLE FINAL
# ---------------------------------------------------------------------
voting_final = VotingClassifier(
    estimators=estimators,
    voting='soft',
    weights=list(opt_weights),
    n_jobs=-1
)

pipeline_final = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', voting_final)
])

# ---------------------------------------------------------------------
# 6️⃣ ENTRAÎNEMENT FINAL + PRÉDICTION TEST
# ---------------------------------------------------------------------
print("\n🚀 Entraînement final sur tout le dataset...")

pipeline_final.fit(X, y)
test_proba = pipeline_final.predict_proba(X_test)[:, 1]
test_pred = (test_proba >= best_thr).astype(int)

print(f"   Taux de conversion prédit : {test_pred.mean():.4%}")
print(f"   Conversions : {test_pred.sum()} / {len(test_pred)}")

# ---------------------------------------------------------------------
# 7️⃣ SAUVEGARDE
# ---------------------------------------------------------------------
pd.DataFrame({'converted': test_pred}).to_csv(
    'divination_v1_5_predictions.csv', index=False
)

pd.DataFrame({
    'Model': [name for name, _ in estimators],
    'Weight': opt_weights
}).to_csv('divination_v1_5_weights.csv', index=False)

print("\n💾 Fichiers générés :")
print("   - divination_v1_5_predictions.csv")
print("   - divination_v1_5_weights.csv")
print("="*80)
