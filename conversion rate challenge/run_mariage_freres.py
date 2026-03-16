
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import warnings

warnings.filterwarnings('ignore')

print("="*60)
print("🍵 PROTOCOLE DE VALIDATION : MARIAGE FRÈRES")
print("="*60)

# 1. Chargement & Préparation
print("⏳ Chargement des données Train...")
df = pd.read_csv('conversion_data_train.csv')

# Feature Engineering
df['interaction'] = df['age'] * df['total_pages_visited']
df['pages_per_age'] = df['total_pages_visited'] / (df['age'] + 0.1)
#'is_active' n'est pas strictement nécessaire pour les tree-models mais inclus pour cohérence
df['is_active'] = (df['total_pages_visited'] > 2).astype(int)

X = df.drop('converted', axis=1)
y = df['converted']

# 2. Simulation du Sénat (Modèle Robuste V1-Like)
# On utilise un VotingClassifier simplifié mais représentatif pour la validation
print("🤖 Construction du Sénat Simulé (XGB + LGBM + LR)...")

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), ['age', 'total_pages_visited', 'interaction', 'pages_per_age']),
    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), ['country', 'source'])
])

senat_simule = VotingClassifier(
    estimators=[
        ('xgb', XGBClassifier(n_estimators=200, max_depth=4, random_state=42, n_jobs=1)),
        ('lgbm', LGBMClassifier(n_estimators=200, max_depth=4, verbose=-1, random_state=42, n_jobs=1)),
        ('lr', LogisticRegression(class_weight={0:1, 1:10}, max_iter=1000, random_state=42, n_jobs=1))
    ],
    voting='soft'
)

pipeline = Pipeline([('pre', preprocessor), ('clf', senat_simule)])

# 3. Génération OOF (Out-Of-Fold)
oof_proba = np.zeros(len(df))
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("🔄 Démarrage de la Cross-Validation (5-Folds)...")
for i, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
    pipeline.fit(X.iloc[tr_idx], y.iloc[tr_idx])
    oof_proba[va_idx] = pipeline.predict_proba(X.iloc[va_idx])[:, 1]
    print(f"   Fold {i}/5 terminé.")

# 4. Évaluation & Rattrapage
base_preds = (oof_proba >= 0.5).astype(int)
f1_base = f1_score(y, base_preds)

print("\n📊 RÉSULTATS PRÉLIMINAIRES")
print(f"🔹 F1 Score de Base (Sénat) : {f1_base:.6f}")

# Définition de la Zone de Rattrapage (Hypothèse Mariage Frères)
# Cible : Old User, 9-16 pages (légèrement élargi), Proba incertaine (0.30 - 0.50)
condition_rattrapage = (
    (df['new_user'] == 0) &
    (df['total_pages_visited'] >= 9) & 
    (df['total_pages_visited'] <= 16) &
    (oof_proba >= 0.25) &  # On descend un peu le seuil pour attraper plus de "Fantômes"
    (oof_proba < 0.50)
)

n_sauves = condition_rattrapage.sum()
print(f"🎯 Candidats au repêchage identifiés : {n_sauves}")

# Application
new_preds = base_preds.copy()
new_preds[condition_rattrapage] = 1

f1_new = f1_score(y, new_preds)
delta = f1_new - f1_base

print("\n⚖️ VERDICT FINAL")
print(f"🔸 F1 Score Mariage Frères  : {f1_new:.6f}")
print(f"📈 Variation : {delta:+.6f}")

if delta > 0:
    print("\n✅ SUCCÈS : La stratégie est validée ! Elle apporte du gain.")
else:
    print("\n❌ ÉCHEC : La stratégie a dégradé le score. Trop de faux positifs.")

print("="*60)
