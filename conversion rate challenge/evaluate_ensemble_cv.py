"""
EVALUATE ENSEMBLE CV - VERDICT FINAL
Calcule le F1-Score du "Grand Conseil" (V4) via Cross-Validation sur le Train Set.
"""

import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import VotingClassifier, GradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

print("="*80)
print("⚖️ AUDIT DE VÉRITÉ - GRAND CONSEIL (V4)")
print("="*80)

# Chemins
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
TRAIN_PATH = os.path.join(SCRIPT_DIR, 'conversion_data_train.csv')

print("⏳ Chargement des données...")
train_data = pd.read_csv(TRAIN_PATH)

def fe_v1_v2(df):
    """Feature Engineering Standard (V1 & V2)"""
    df_eng = df.copy()
    df_eng['is_active'] = (df_eng['total_pages_visited'] > 2).astype(int)
    df_eng['interaction_age_pages'] = df_eng['age'] * df_eng['total_pages_visited']
    df_eng['pages_per_age'] = df_eng['total_pages_visited'] / (df_eng['age'] + 0.1)
    return df_eng

def fe_v3(df, train_stats=None):
    """Feature Engineering Avancé (V3)"""
    df_eng = df.copy()
    df_eng['is_active'] = (df_eng['total_pages_visited'] > 2).astype(int)
    df_eng['interaction_age_pages'] = df_eng['age'] * df_eng['total_pages_visited']
    df_eng['pages_per_age'] = df_eng['total_pages_visited'] / (df_eng['age'] + 0.1)
    
    # New Features
    df_eng['new_user_pages'] = df_eng['new_user'] * df_eng['total_pages_visited']
    df_eng['is_hyper_active'] = (df_eng['total_pages_visited'] > 12).astype(int)
    
    if train_stats is None:
        country_means = df_eng.groupby('country')['total_pages_visited'].mean()
        train_stats = {'country_means': country_means}
    
    means = train_stats['country_means']
    df_eng['country_mean_pages'] = df_eng['country'].map(means).fillna(means.mean())
    df_eng['pages_relative_to_country'] = df_eng['total_pages_visited'] / df_eng['country_mean_pages']
    
    return df_eng

# Préparation Data
X_orig = train_data.drop('converted', axis=1)
y = train_data['converted']

# ---------------------------------------------------------------------
# DÉFINITION DES MODÈLES
# ---------------------------------------------------------------------

# Wrapper Poisson (pour V2)
class PoissonSupremacyClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, learning_rate=0.05, max_iter=500, max_depth=8, l2_regularization=0.1, random_state=42):
        self.learning_rate, self.max_iter, self.max_depth, self.l2_regularization, self.random_state = \
            learning_rate, max_iter, max_depth, l2_regularization, random_state
    def __sklearn_tags__(self): tags = super().__sklearn_tags__(); tags.estimator_type = "classifier"; return tags
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.model_ = HistGradientBoostingRegressor(loss='poisson', learning_rate=self.learning_rate, 
            max_iter=self.max_iter, max_depth=self.max_depth, l2_regularization=self.l2_regularization, random_state=self.random_state)
        self.model_.fit(X, y); return self
    def predict_proba(self, X):
        pred = np.clip(self.model_.predict(X), 0, 1); return np.column_stack([1-pred, pred])
    def predict(self, X): return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

# Modèles de base
xgb1 = XGBClassifier(n_estimators=350, max_depth=4, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9, eval_metric='logloss', random_state=42, n_jobs=1)
xgb2 = XGBClassifier(n_estimators=350, max_depth=4, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9, eval_metric='logloss', random_state=2025, n_jobs=1)
lgbm = LGBMClassifier(n_estimators=350, max_depth=4, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9, verbose=-1, random_state=42, n_jobs=1)
gb = GradientBoostingClassifier(n_estimators=350, max_depth=4, learning_rate=0.05, subsample=0.9, random_state=42)
lr = LogisticRegression(max_iter=1000, class_weight={0: 1, 1: 80}, solver='lbfgs', n_jobs=1)
poisson = PoissonSupremacyClassifier()

# Pipelines
# V1
pre_v1 = ColumnTransformer([
    ('num', StandardScaler(), ['age', 'total_pages_visited', 'interaction_age_pages', 'pages_per_age']),
    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), ['country', 'source'])
], remainder='passthrough')

pipe_v1 = Pipeline([('pre', pre_v1), ('clf', VotingClassifier([
    ('xgb1', xgb1), ('xgb2', xgb2), ('lgbm', lgbm), ('gb', gb), ('logreg', lr)], 
    voting='soft', weights=[0.7, 0.7, 1.2, 0.7, 0.5], n_jobs=-1))])

# V2 (Poisson)
pipe_v2 = Pipeline([('pre', pre_v1), ('clf', VotingClassifier([
    ('xgb1', xgb1), ('xgb2', xgb2), ('lgbm', lgbm), ('poisson', poisson), ('logreg', lr)], 
    voting='soft', weights=[0.35, 0.35, 0.40, 0.35, 0.12], n_jobs=-1))])

# V3 (Features)
pre_v3 = ColumnTransformer([
    ('num', StandardScaler(), ['age', 'total_pages_visited', 'interaction_age_pages', 'pages_per_age', 'new_user_pages', 'is_hyper_active', 'pages_relative_to_country']),
    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), ['country', 'source'])
], remainder='passthrough')

pipe_v3 = Pipeline([('pre', pre_v3), ('clf', VotingClassifier([
    ('xgb1', xgb1), ('xgb2', xgb2), ('lgbm', lgbm), ('gb', gb), ('logreg', lr)], 
    voting='soft', weights=[0.7, 0.7, 1.2, 0.7, 0.5], n_jobs=-1))])


# ---------------------------------------------------------------------
# CROSS-VALIDATION LOOP
# ---------------------------------------------------------------------
print("\n🚀 Démarrage de la Cross-Validation (5-Fold)...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_v1 = np.zeros(len(y))
oof_v2 = np.zeros(len(y))
oof_v3 = np.zeros(len(y))

for fold, (idx_tr, idx_va) in enumerate(skf.split(X_orig, y), 1):
    print(f"   Fold {fold}/5...", end=" ", flush=True)
    
    # Split
    X_tr, X_va = X_orig.iloc[idx_tr], X_orig.iloc[idx_va]
    y_tr = y.iloc[idx_tr]
    
    # --- V1 & V2 (Mêmes features) ---
    X_tr_std = fe_v1_v2(X_tr)
    X_va_std = fe_v1_v2(X_va)
    
    # Encode cols for V1/V2 before pipeline (si besoin par LabelEncoder, mais ici Pipeline gère OneHot)
    # On passe direct au pipeline avec OneHotEncoder inclus
    
    pipe_v1.fit(X_tr_std, y_tr)
    oof_v1[idx_va] = pipe_v1.predict_proba(X_va_std)[:, 1]
    
    pipe_v2.fit(X_tr_std, y_tr)
    oof_v2[idx_va] = pipe_v2.predict_proba(X_va_std)[:, 1]
    
    # --- V3 (Features Avancées) ---
    # Calc stats only on train !
    X_tr_adv = fe_v3(X_tr)
    train_stats = {'country_means': X_tr_adv.groupby('country')['total_pages_visited'].mean()}
    X_va_adv = fe_v3(X_va, train_stats) # Apply train stats
    
    pipe_v3.fit(X_tr_adv, y_tr)
    oof_v3[idx_va] = pipe_v3.predict_proba(X_va_adv)[:, 1]
    
    print("✓")

print("\n✅ Calcul des scores...")

def evaluate(preds, name):
    best_f1, best_thr = 0, 0
    for thr in np.linspace(0.3, 0.7, 100):
        sc = f1_score(y, (preds >= thr).astype(int))
        if sc > best_f1: best_f1, best_thr = sc, thr
    print(f"   {name:<20} : F1 = {best_f1:.5f} (Seuil {best_thr:.2f})")
    return best_f1, best_thr

# Scores Individuels
f1_v1, thr_v1 = evaluate(oof_v1, "V1 (Original)")
f1_v2, thr_v2 = evaluate(oof_v2, "V2 (Poisson)")
f1_v3, thr_v3 = evaluate(oof_v3, "V3 (Features)")

# --- V4 : LE GRAND CONSEIL ---
# Hard Voting sur les prédictions binaires (avec seuils optimaux respectifs)
pred_bin_v1 = (oof_v1 >= thr_v1).astype(int)
pred_bin_v2 = (oof_v2 >= thr_v2).astype(int)
pred_bin_v3 = (oof_v3 >= thr_v3).astype(int)

votes = pred_bin_v1 + pred_bin_v2 + pred_bin_v3
pred_v4 = (votes >= 2).astype(int)

f1_v4 = f1_score(y, pred_v4)

print(f"\n🧙‍♂️ RÉSULTAT OOF DU GRAND CONSEIL (V4) :")
print(f"   F1 = {f1_v4:.5f}")

if f1_v4 > f1_v1:
    print(f"\n🎉 VICTOIRE ! L'ensemble bat V1 de +{f1_v4 - f1_v1:.5f} points !")
else:
    print(f"\n😔 DÉCEPTION... V1 reste le maître (-{f1_v1 - f1_v4:.5f} points).")
    print("   Parfois, 'Less is More'.")

print(f"\n{'='*70}")
print("🔍 ANALYSE DE CORRÉLATION DES ERREURS")
print(f"{'='*70}")
# Matrice de corrélation des prédictions
res_df = pd.DataFrame({'V1': oof_v1, 'V2': oof_v2, 'V3': oof_v3})
print(res_df.corr())
