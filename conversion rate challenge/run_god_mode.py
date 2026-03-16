
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import f1_score
import warnings

warnings.filterwarnings('ignore')

print("⚡ GOD MODE: INITIALISATION (KNN k=1 - MEMORIZATION) ⚡")
print("📥 Chargement des données...")
train_df = pd.read_csv('conversion rate challenge/conversion_data_train.csv')
test_df = pd.read_csv('conversion rate challenge/conversion_data_test.csv')

target = train_df['converted']
train_df = train_df.drop('converted', axis=1)

# 1. Feature Engineering (Strict Minimum pour KNN)
# KNN a besoin de distances, donc il faut normaliser.
# On garde les features numériques et on encode pays/source.

full_df = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)

# Encoding Simple
le = LabelEncoder()
full_df['country'] = le.fit_transform(full_df['country'])
full_df['source'] = le.fit_transform(full_df['source'])

# Scaling (CRUCIAL pour KNN)
scaler = StandardScaler()
X_full = scaler.fit_transform(full_df)

X = X_full[:len(train_df)]
X_test = X_full[len(train_df):]
y = target

# 2. Evaluation Locale (10-Fold CV)
print("\n⚔️ Lancement de l'évaluation Locale (10-Fold CV)...")
model = KNeighborsClassifier(n_neighbors=1, n_jobs=-1)
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

scoring = ['f1', 'recall', 'precision', 'roc_auc']
scores = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)

print("\n📊 RÉSULTATS GOD MODE (Local) :")
print(f"   🎯 F1 Score  : {scores['test_f1'].mean():.5f} (+/- {scores['test_f1'].std():.4f})")
print(f"   🔄 Recall    : {scores['test_recall'].mean():.5f}")
print(f"   🎯 Precision : {scores['test_precision'].mean():.5f}")
print(f"   📈 ROC AUC   : {scores['test_roc_auc'].mean():.5f}")

# 3. Entraînement Final & Soumission
print("\n🚀 Entraînement Final sur tout le Dataset...")
model.fit(X, y)
y_pred = model.predict(X_test)

filename = 'conversion rate challenge/submission_GOD_MODE.csv'
pd.DataFrame({'converted': y_pred}).to_csv(filename, index=False)

print(f"✅ Fichier '{filename}' généré.")
print(f"   Nombre de conversions prédites : {y_pred.sum()}")
