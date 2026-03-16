"""
DIVINATION V6 - SUPER ENSEMBLE ÉQUILIBRÉ
Stratégie : Intégrer les 3 champions (Ultimate Mix + Mariage Frère + LightGBM)
Poids égaux (1.0) pour tous les modèles - Approche démocratique
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
from sklearn.ensemble import VotingClassifier, HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

print("="*80)
print("🎯 DIVINATION V6 - SUPER ENSEMBLE ÉQUILIBRÉ")
print("="*80)
print("\n💡 STRATÉGIE :")
print("   ❌ Retiré  : XGBoost #1 (redondance)")
print("   ✅ Ajouté  : Ultimate Mix (Poisson+LogLoss) - F1=0.76877")
print("   ✅ Ajouté  : Mariage Frère (Tweedie) - F1=0.76666")
print("   ⚖️  Poids   : 1.0 pour TOUS (approche démocratique)\n")

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

# Wrapper pour Ultimate Mix
class UltimateMixClassifier(BaseEstimator, ClassifierMixin):
    """Ultimate Mix (Poisson + LogLoss)"""
    
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
        
        # Poisson
        self.model_poi_ = HistGradientBoostingRegressor(
            loss='poisson', learning_rate=self.learning_rate, max_iter=self.max_iter,
            max_depth=self.max_depth, l2_regularization=self.l2_regularization,
            categorical_features=cat_indices if cat_indices else 'from_dtype',
            random_state=self.random_state
        )
        self.model_poi_.fit(X, y)
        
        # LogLoss
        self.model_clf_ = HistGradientBoostingClassifier(
            loss='log_loss', learning_rate=self.learning_rate, max_iter=self.max_iter,
            max_depth=self.max_depth, l2_regularization=self.l2_regularization,
            categorical_features=cat_indices if cat_indices else 'from_dtype',
            random_state=self.random_state
        )
        self.model_clf_.fit(X, y)
        
        return self
    
    def predict_proba(self, X):
        pred_poi = self.model_poi_.predict(X)
        pred_clf = self.model_clf_.predict_proba(X)[:, 1]
        avg_pred = (pred_poi + pred_clf) / 2
        proba_class_1 = np.clip(avg_pred, 0, 1)
        proba_class_0 = 1 - proba_class_1
        return np.column_stack([proba_class_0, proba_class_1])
    
    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

# Wrapper pour Mariage Frère
class MariageFrèreClassifier(BaseEstimator, ClassifierMixin):
    """Mariage Frère (Tweedie)"""
    
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
            loss='poisson', learning_rate=self.learning_rate, max_iter=self.max_iter,
            max_depth=self.max_depth, l2_regularization=self.l2_regularization,
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

# Preprocessing
numeric_features = ['age', 'total_pages_visited', 'interaction_age_pages', 'pages_per_age']
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols)
    ],
    remainder='passthrough'
)

# Les 5 modèles - SUPER ENSEMBLE V6
clf_xgb = XGBClassifier(
    n_estimators=350, max_depth=4, learning_rate=0.05,
    subsample=0.9, colsample_bytree=0.9, eval_metric='logloss',
    random_state=2025, n_jobs=1
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

clf_mariage = MariageFrèreClassifier(
    learning_rate=0.05, max_iter=500, max_depth=8,
    l2_regularization=0.1, random_state=42
)

clf_logreg = LogisticRegression(
    max_iter=1000, class_weight={0: 1, 1: 80},
    solver='lbfgs', n_jobs=1
)

# VotingClassifier avec POIDS ÉGAUX
voting_model_v6 = VotingClassifier(
    estimators=[
        ('xgb', clf_xgb),
        ('ultimate_mix', clf_ultimate),      # 🆕 Champion F1=0.76877
        ('lgbm', clf_lgbm),
        ('mariage_frere', clf_mariage),      # 🆕 Champion F1=0.76666
        ('logreg', clf_logreg)
    ],
    voting='soft',
    weights=[1.0, 1.0, 1.0, 1.0, 1.0],      # ⚖️ TOUS ÉGAUX !
    n_jobs=-1
)

pipeline_v6 = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', voting_model_v6)
])

print("📊 Configuration V6 - SUPER ENSEMBLE :")
print("   - XGBoost (seed=2025)      : poids 1.0 (20%)")
print("   - Ultimate Mix (Poi+Log)   : poids 1.0 (20%) 🆕")
print("   - LightGBM                 : poids 1.0 (20%)")
print("   - Mariage Frère (Tweedie)  : poids 1.0 (20%) 🆕")
print("   - LogisticRegression       : poids 1.0 (20%)")
print("\n🎯 Approche démocratique : Tous les modèles ont la même voix !")
print("   • 3 Champions : Ultimate Mix + LightGBM + Mariage Frère")
print("   • Diversité : Linéaire + Arbres + Hybrides")
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
    
    pipeline_v6.fit(X_train_fold, y_train_fold)
    val_proba = pipeline_v6.predict_proba(X_val_fold)[:, 1]
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

metrics_v6 = {
    'f1': best_f1,
    'roc_auc': roc_auc_score(y, oof_predictions),
    'precision': precision_score(y, final_predictions),
    'recall': recall_score(y, final_predictions),
    'accuracy': accuracy_score(y, final_predictions),
    'threshold': best_threshold
}

print(f"   Seuil optimal : {best_threshold:.6f}")
print(f"\n{'='*70}")
print("🏆 RÉSULTATS DIVINATION V6")
print(f"{'='*70}")
print(f"   F1-Score  : {metrics_v6['f1']:.5f}")
print(f"   ROC-AUC   : {metrics_v6['roc_auc']:.5f}")
print(f"   Precision : {metrics_v6['precision']:.5f}")
print(f"   Recall    : {metrics_v6['recall']:.5f}")
print(f"   Accuracy  : {metrics_v6['accuracy']:.5f}")
print(f"{'='*70}")

# Comparaison complète
metrics_all = {
    'V1': {'f1': 0.77022, 'roc_auc': 0.98577, 'precision': 0.81007, 'recall': 0.73410, 'accuracy': 0.98587},
    'V3': {'f1': 0.76948, 'roc_auc': 0.98545, 'precision': 0.81970, 'recall': 0.72505, 'accuracy': 0.98599},
    'V4': {'f1': 0.76906, 'roc_auc': 0.98577, 'precision': 0.82071, 'recall': 0.72353, 'accuracy': 0.98598},
    'V5': {'f1': 0.77011, 'roc_auc': 0.98577, 'precision': 0.81947, 'recall': 0.72636, 'accuracy': 0.98601},
    'V6': metrics_v6
}

print("\n" + "="*110)
print("📊 COMPARAISON COMPLÈTE : V1 vs V3 vs V4 vs V5 vs V6")
print("="*110)
print(f"\n{'Métrique':<12} {'V1':<18} {'V3':<18} {'V4':<18} {'V5':<18} {'V6':<18}")
print("-"*110)

for metric in ['f1', 'roc_auc', 'precision', 'recall', 'accuracy']:
    vals = [metrics_all[v][metric] for v in ['V1', 'V3', 'V4', 'V5', 'V6']]
    best_val = max(vals)
    
    row = f"{metric.upper():<12}"
    for v in ['V1', 'V3', 'V4', 'V5', 'V6']:
        val = metrics_all[v][metric]
        val_str = f"{val:.5f}" + (" ⭐" if val == best_val else "")
        row += f" {val_str:<18}"
    print(row)

print("\n" + "="*110)

# Classement
all_f1 = [(f"V{i}", metrics_all[f"V{i}"]['f1']) for i in [1, 3, 4, 5, 6]]
all_f1_sorted = sorted(all_f1, key=lambda x: x[1], reverse=True)

print(f"\n🏆 CLASSEMENT F1-Score :")
for i, (name, score) in enumerate(all_f1_sorted, 1):
    emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
    print(f"   {emoji} {i}. {name:<10} F1 = {score:.5f}")

winner_name, winner_score = all_f1_sorted[0]

if winner_name == 'V6':
    print(f"\n🎉 VICTOIRE V6 - SUPER ENSEMBLE !")
    print(f"   Nouveau champion : F1 = {metrics_v6['f1']:.5f}")
    print(f"\n🚀 RECOMMANDATION : Soumettre Divination V6 sur Kaggle !")
else:
    print(f"\n➡️ {winner_name} reste champion avec F1 = {winner_score:.5f}")
    delta = (metrics_v6['f1'] - winner_score) * 100
    if delta > -0.02:
        print(f"   V6 est très proche ! Écart de seulement {abs(delta):.3f} points")

print("\n" + "="*110)

# Entraînement final
print("\n🚀 Entraînement du modèle final sur tout le dataset...\n")

pipeline_v6.fit(X, y)
test_proba = pipeline_v6.predict_proba(X_test)[:, 1]
test_predictions = (test_proba >= best_threshold).astype(int)

submission = pd.DataFrame({'converted': test_predictions})
submission.to_csv('divination_v6_predictions.csv', index=False)

print("✅ Prédictions générées : divination_v6_predictions.csv")
print(f"   Taux de conversion prédit : {test_predictions.mean():.4%}")
print(f"   Nombre de conversions : {test_predictions.sum()} / {len(test_predictions)}")
print(f"\n🎯 Seuil utilisé : {best_threshold:.6f}")
print(f"\n{'='*70}")
print("🏆 DIVINATION V6 - SUPER ENSEMBLE - PRÊT POUR SOUMISSION")
print(f"{'='*70}")
