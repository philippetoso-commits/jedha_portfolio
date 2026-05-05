
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score

print("🧠 NEURAL NETWORK PROTOTYPE: Can Deep Learning beat Trees? 🧠")
print("📥 Loading Data...")
df = pd.read_csv('conversion rate challenge/conversion_data_train.csv')

# Feature Engineering
def feature_engineering(df):
    df_eng = df.copy()
    df_eng['pages_per_age'] = df_eng['total_pages_visited'] / (df_eng['age'] + 0.1)
    df_eng['active_user'] = (df_eng['total_pages_visited'] > 10).astype(int) 
    return df_eng

X = feature_engineering(df.drop('converted', axis=1))
y = df['converted']

# Preprocessing (CRITIQUE pour les Réseaux de Neurones)
# Les NN détestent les échelles différentes. Scaling OBLIGATOIRE.
numeric_features = ['age', 'new_user', 'total_pages_visited', 'pages_per_age']
categorical_features = ['country', 'source']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
    ]
)

# Architecture du Réseau
# 2 Couches cachées (64 neurones, 32 neurones)
# Activation ReLU (Standard)
# Solver Adam (Standard)
mlp = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation='relu',
    solver='adam',
    alpha=0.0001, # Petite régularisation L2
    batch_size=256,
    learning_rate_init=0.001,
    max_iter=200,
    early_stopping=True, # Pour éviter l'overfitting
    validation_fraction=0.1,
    random_state=42
)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('mlp', mlp)
])

print("\n⚙️ Training & Evaluating Neural Network (10-Fold CV)...")
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
scores = cross_val_score(pipeline, X, y, cv=cv, scoring='f1', n_jobs=-1)

print(f"\n📊 RESULTATS NEURAL NETWORK :")
print(f"   🎯 F1 Score Moyen : {scores.mean():.5f} (+/- {scores.std():.4f})")

# Comparatif
print(f"\n🆚 Comparaison :")
print(f"   Gradient Boosting : ~0.764")
print(f"   Random Forest     : ~0.579")
print(f"   Neural Network    : {scores.mean():.5f}")

if scores.mean() > 0.76:
    print("✅ SUCCÈS : Le Deep Learning est compétitif ! Il peut renforcer le Syndicat.")
elif scores.mean() > 0.74:
    print("⚠️ MOYEN : Pas ridicule, mais pas suffisant pour détrôner le Roi XGBoost.")
else:
    print("❌ ÉCHEC : Les données tabulaires préfèrent les arbres.")
