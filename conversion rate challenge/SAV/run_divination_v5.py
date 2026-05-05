"""
DIVINATION V5 - SNIPER ÉLITE + MARIAGE FRÈRE
Stratégie : Remplacer GradientBoosting par Mariage Frère (Tweedie)
Intégrer un champion individuel (F1=0.76666) dans l'ensemble
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
from sklearn.ensemble import VotingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

print("="*80)
print("🎯 DIVINATION V5 - SNIPER ÉLITE + MARIAGE FRÈRE")
print("="*80)
print("\n💡 STRATÉGIE :")
print("   ❌ Retiré  : GradientBoosting (redondance avec LightGBM)")
print("   ✅ Ajouté  : Mariage Frère - Tweedie (F1=0.76666)")
print("   ✅ Poids   : 0.7 (même que GradientBoosting)\n")

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

# Wrapper pour Mariage Frère (Tweedie)
class MariageFrèreClassifier(BaseEstimator, ClassifierMixin):
    """Wrapper pour Mariage Frère (Tweedie) compatible avec sklearn"""
    
    def __init__(self, learning_rate=0.05, max_iter=500, max_depth=8, 
                 l2_regularization=0.1, random_state=42):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.max_depth = max_depth
        self.l2_regularization = l2_regularization
        self.random_state = random_state
    
    def __sklearn_tags__(self):
        """Tags sklearn pour validation"""
        tags = super().__sklearn_tags__()
        tags.estimator_type = "classifier"
        return tags
        
    def fit(self, X, y):
        """Entraîne le modèle Tweedie"""
        self.classes_ = np.unique(y)
        
        # Identifier les colonnes catégorielles
        cat_cols = ['country', 'source']
        if hasattr(X, 'columns'):
            cat_indices = [list(X.columns).index(col) for col in cat_cols if col in X.columns]
        else:
            cat_indices = []
        
        # Modèle Tweedie
        self.model_ = HistGradientBoostingRegressor(
            loss='poisson',  # Tweedie avec power=1 (Poisson)
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
        """Retourne les probabilités"""
        pred = self.model_.predict(X)
        
        # Convertir en probabilités
        proba_class_1 = np.clip(pred, 0, 1)
        proba_class_0 = 1 - proba_class_1
        
        return np.column_stack([proba_class_0, proba_class_1])
    
    def predict(self, X):
        """Retourne les prédictions binaires"""
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

# Les 5 modèles - COMPOSITION V5
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

clf_mariage = MariageFrèreClassifier(
    learning_rate=0.05, max_iter=500, max_depth=8,
    l2_regularization=0.1, random_state=42
)

clf_logreg = LogisticRegression(
    max_iter=1000, class_weight={0: 1, 1: 80},
    solver='lbfgs', n_jobs=1
)

# VotingClassifier avec Mariage Frère
voting_model_v5 = VotingClassifier(
    estimators=[
        ('xgb1', clf_xgb1),
        ('xgb2', clf_xgb2),
        ('lgbm', clf_lgbm),
        ('mariage_frere', clf_mariage),  # 🆕 Remplace GradientBoosting
        ('logreg', clf_logreg)
    ],
    voting='soft',
    weights=[0.7, 0.7, 1.2, 0.7, 0.5],  # Mariage Frère: poids 0.7
    n_jobs=-1
)

pipeline_v5 = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', voting_model_v5)
])

print("📊 Configuration V5 :")
print("   - XGBoost #1 (seed=42)     : poids 0.7")
print("   - XGBoost #2 (seed=2025)   : poids 0.7")
print("   - LightGBM                 : poids 1.2 ⭐")
print("   - Mariage Frère (Tweedie)  : poids 0.7 🆕")
print("   - LogisticRegression       : poids 0.5")
print("\n🎯 Diversité maximale :")
print("   • Linéaire    : LogisticRegression")
print("   • Arbres      : XGBoost×2, LightGBM")
print("   • Régression  : Mariage Frère (Tweedie)")
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
    pipeline_v5.fit(X_train_fold, y_train_fold)
    
    # Prédictions
    val_proba = pipeline_v5.predict_proba(X_val_fold)[:, 1]
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
metrics_v5 = {
    'f1': best_f1,
    'roc_auc': roc_auc_score(y, oof_predictions),
    'precision': precision_score(y, final_predictions),
    'recall': recall_score(y, final_predictions),
    'accuracy': accuracy_score(y, final_predictions),
    'threshold': best_threshold
}

print(f"   Seuil optimal : {best_threshold:.6f}")
print(f"\n{'='*70}")
print("🏆 RÉSULTATS DIVINATION V5")
print(f"{'='*70}")
print(f"   F1-Score  : {metrics_v5['f1']:.5f}")
print(f"   ROC-AUC   : {metrics_v5['roc_auc']:.5f}")
print(f"   Precision : {metrics_v5['precision']:.5f}")
print(f"   Recall    : {metrics_v5['recall']:.5f}")
print(f"   Accuracy  : {metrics_v5['accuracy']:.5f}")
print(f"{'='*70}")

# Comparaison avec toutes les versions
metrics_v1 = {'f1': 0.77022, 'roc_auc': 0.98577, 'precision': 0.81007, 'recall': 0.73410, 'accuracy': 0.98587}
metrics_v3 = {'f1': 0.76948, 'roc_auc': 0.98545, 'precision': 0.81970, 'recall': 0.72505, 'accuracy': 0.98599}
metrics_v4 = {'f1': 0.76906, 'roc_auc': 0.98577, 'precision': 0.82071, 'recall': 0.72353, 'accuracy': 0.98598}

print("\n" + "="*100)
print("📊 COMPARAISON COMPLÈTE : V1 vs V3 vs V4 vs V5")
print("="*100)
print(f"\n{'Métrique':<15} {'V1 (Original)':<20} {'V3 (Ult→LogR)':<20} {'V4 (Ult→XGB2)':<20} {'V5 (MF→GB)':<20}")
print("-"*100)

for metric in ['f1', 'roc_auc', 'precision', 'recall', 'accuracy']:
    v1_val = metrics_v1[metric]
    v3_val = metrics_v3[metric]
    v4_val = metrics_v4[metric]
    v5_val = metrics_v5[metric]
    
    best_val = max(v1_val, v3_val, v4_val, v5_val)
    
    v1_str = f"{v1_val:.5f}" + (" ⭐" if v1_val == best_val else "")
    v3_str = f"{v3_val:.5f}" + (" ⭐" if v3_val == best_val else "")
    v4_str = f"{v4_val:.5f}" + (" ⭐" if v4_val == best_val else "")
    v5_str = f"{v5_val:.5f}" + (" ⭐" if v5_val == best_val else "")
    
    print(f"{metric.upper():<15} {v1_str:<20} {v3_str:<20} {v4_str:<20} {v5_str:<20}")

print("\n" + "="*100)

# Verdict
all_f1 = [
    ('V1 (Original)', metrics_v1['f1']),
    ('V3 (Ultimate Mix → LogReg)', metrics_v3['f1']),
    ('V4 (Ultimate Mix → XGB2)', metrics_v4['f1']),
    ('V5 (Mariage Frère → GB)', metrics_v5['f1'])
]
all_f1_sorted = sorted(all_f1, key=lambda x: x[1], reverse=True)

print(f"\n🏆 CLASSEMENT F1-Score :")
for i, (name, score) in enumerate(all_f1_sorted, 1):
    emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
    print(f"   {emoji} {i}. {name:<45} F1 = {score:.5f}")

winner_name, winner_score = all_f1_sorted[0]

if winner_name.startswith('V5'):
    improvement_v1 = (metrics_v5['f1'] - metrics_v1['f1']) * 100
    print(f"\n🎉 VICTOIRE V5 !")
    print(f"   Amélioration vs V1 : +{improvement_v1:.3f} points")
    print(f"\n🚀 RECOMMANDATION : Soumettre Divination V5 sur Kaggle !")
else:
    print(f"\n➡️ {winner_name} reste champion avec F1 = {winner_score:.5f}")
    if metrics_v5['f1'] > 0.77:
        print(f"   Mais V5 dépasse 0.77 ! Excellente performance !")

print("\n" + "="*100)

# Entraînement final et prédictions
print("\n🚀 Entraînement du modèle final sur tout le dataset...\n")

pipeline_v5.fit(X, y)

test_proba = pipeline_v5.predict_proba(X_test)[:, 1]
test_predictions = (test_proba >= best_threshold).astype(int)

submission = pd.DataFrame({
    'converted': test_predictions
})

submission.to_csv('divination_v5_predictions.csv', index=False)

print("✅ Prédictions générées : divination_v5_predictions.csv")
print(f"   Taux de conversion prédit : {test_predictions.mean():.4%}")
print(f"   Nombre de conversions : {test_predictions.sum()} / {len(test_predictions)}")
print(f"\n🎯 Seuil utilisé : {best_threshold:.6f}")
print(f"\n{'='*70}")
print("🏆 DIVINATION V5 - PRÊT POUR SOUMISSION")
print(f"{'='*70}")
