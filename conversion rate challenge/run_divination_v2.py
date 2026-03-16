"""
DIVINATION V2 - POISSON SUPREMACY REPLACES GRADIENTBOOSTING
Composition : XGB1 + XGB2 + LightGBM + Poisson Supremacy + LogisticRegression
Poids proposés : [0.35, 0.35, 0.40, 0.35, 0.12]
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
from sklearn.ensemble import VotingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

print("="*80)
print("🎯 DIVINATION V2 - POISSON SUPREMACY")
print("="*80)
print("\n💡 STRATÉGIE :")
print("   ❌ Retiré  : GradientBoosting (redondant avec LightGBM)")
print("   ✅ Ajouté  : Poisson Supremacy (macro expert)")
print("   ⚖️  Poids   : [0.35, 0.35, 0.40, 0.35, 0.12]\n")

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

# Wrapper pour Poisson Supremacy
class PoissonSupremacyClassifier(BaseEstimator, ClassifierMixin):
    """Poisson Supremacy - Macro expert basé sur régression Poisson"""
    
    def __init__(self, learning_rate=0.05, max_iter=500, max_depth=8, 
                 l2_regularization=0.1, random_state=42):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.max_depth = max_depth
        self.l2_regularization = l2_regularization
        self.random_state = random_state
    
    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.estimator_type = "classifier"
        return tags
        
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        
        cat_cols = ['country', 'source']
        if hasattr(X, 'columns'):
            cat_indices = [list(X.columns).index(col) for col in cat_cols if col in X.columns]
        else:
            cat_indices = []
        
        self.model_ = HistGradientBoostingRegressor(
            loss='poisson',
            learning_rate=self.learning_rate,
            max_iter=self.max_iter,
            max_depth=self.max_depth,
            l2_regularization=self.l2_regularization,
            categorical_features=cat_indices if cat_indices else 'from_dtype',
            random_state=self.random_state
        )
        self.model_.fit(X, y)
        
        return self
    
    def predict_proba(self, X):
        pred = self.model_.predict(X)
        proba_class_1 = np.clip(pred, 0, 1)
        proba_class_0 = 1 - proba_class_1
        return np.column_stack([proba_class_0, proba_class_1])
    
    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

print("🔧 Configuration du pipeline...")

numeric_features = ['age', 'total_pages_visited', 'interaction_age_pages', 'pages_per_age']
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols)
    ],
    remainder='passthrough'
)

# Les 5 modèles - DIVINATION V2
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

clf_poisson = PoissonSupremacyClassifier(
    learning_rate=0.05, max_iter=500, max_depth=8,
    l2_regularization=0.1, random_state=42
)

clf_logreg = LogisticRegression(
    max_iter=1000, class_weight={0: 1, 1: 80},
    solver='lbfgs', n_jobs=1
)

# VotingClassifier avec poids proposés
voting_model_v2 = VotingClassifier(
    estimators=[
        ('xgb1', clf_xgb1),
        ('xgb2', clf_xgb2),
        ('lgbm', clf_lgbm),
        ('poisson_supremacy', clf_poisson),  # 🆕 Remplace GB
        ('logreg', clf_logreg)
    ],
    voting='soft',
    weights=[0.35, 0.35, 0.40, 0.35, 0.12],  # Poids proposés
    n_jobs=-1
)

pipeline_v2 = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', voting_model_v2)
])

print("📊 Configuration V2 - POISSON SUPREMACY :")
print("   - XGBoost #1 (seed=42)     : poids 0.35 (22.3%)")
print("   - XGBoost #2 (seed=2025)   : poids 0.35 (22.3%)")
print("   - LightGBM                 : poids 0.40 (25.5%) ⬆")
print("   - Poisson Supremacy        : poids 0.35 (22.3%) 🆕")
print("   - LogisticRegression       : poids 0.12 (7.6%)")
print("\n🎯 Macro expert Poisson remplace GradientBoosting !")
print("   • Diversité : Linéaire + Arbres + Régression Poisson")
print("   • LightGBM légèrement renforcé (0.40 vs 0.35)")
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
    
    pipeline_v2.fit(X_train_fold, y_train_fold)
    val_proba = pipeline_v2.predict_proba(X_val_fold)[:, 1]
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

metrics_v2 = {
    'f1': best_f1,
    'roc_auc': roc_auc_score(y, oof_predictions),
    'precision': precision_score(y, final_predictions),
    'recall': recall_score(y, final_predictions),
    'accuracy': accuracy_score(y, final_predictions),
    'threshold': best_threshold
}

print(f"   Seuil optimal : {best_threshold:.6f}")
print(f"\n{'='*70}")
print("🏆 RÉSULTATS DIVINATION V2")
print(f"{'='*70}")
print(f"   F1-Score  : {metrics_v2['f1']:.5f}")
print(f"   ROC-AUC   : {metrics_v2['roc_auc']:.5f}")
print(f"   Precision : {metrics_v2['precision']:.5f}")
print(f"   Recall    : {metrics_v2['recall']:.5f}")
print(f"   Accuracy  : {metrics_v2['accuracy']:.5f}")
print(f"{'='*70}")

# Comparaison avec V1
v1_f1 = 0.77022
improvement = (metrics_v2['f1'] - v1_f1) * 100

print(f"\n📈 COMPARAISON vs V1 :")
print(f"   V1 (original)      : F1 = {v1_f1:.5f}")
print(f"   V2 (Poisson)       : F1 = {metrics_v2['f1']:.5f}")
print(f"   Δ                  : {improvement:+.3f} points")

if metrics_v2['f1'] > v1_f1:
    print(f"\n🎉 VICTOIRE ! Poisson Supremacy améliore le modèle !")
elif abs(improvement) < 0.01:
    print(f"\n➡️ ÉGALITÉ : Performance équivalente à V1")
else:
    print(f"\n➡️ V1 reste meilleur")

# Entraînement final et prédictions
print(f"\n🚀 Entraînement du modèle final sur tout le dataset...\n")

pipeline_v2.fit(X, y)
test_proba = pipeline_v2.predict_proba(X_test)[:, 1]
test_predictions = (test_proba >= best_threshold).astype(int)

submission = pd.DataFrame({'converted': test_predictions})
submission_path = os.path.join(SCRIPT_DIR, 'divination_v2_predictions.csv')
submission.to_csv(submission_path, index=False)

print(f"✅ Prédictions générées : {submission_path}")
print(f"   Taux de conversion prédit : {test_predictions.mean():.4%}")
print(f"   Nombre de conversions : {test_predictions.sum()} / {len(test_predictions)}")
print(f"   Seuil utilisé : {best_threshold:.6f}")

print(f"\n{'='*70}")
print("🏆 DIVINATION V2 - POISSON SUPREMACY - PRÊT POUR SOUMISSION")
print(f"{'='*70}")
