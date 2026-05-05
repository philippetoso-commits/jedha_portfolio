
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score

print("🌲 RANDOM FOREST: THE OLD KING RETURNS? 🌲")
print("📥 Loading Data...")
df = pd.read_csv('conversion rate challenge/conversion_data_train.csv')

# Feature Engineering
X = df.drop('converted', axis=1)
y = df['converted']
X['pages_per_age'] = X['total_pages_visited'] / (X['age'] + 0.1)

# Encoding (Random Forest handles label encoding fine, widely accepted)
for col in ['country', 'source']:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

print("\n⚙️ Training Tuned Random Forest (n=200, depth=15)...")
# On limite la profondeur pour éviter le pur cœur par cœur, mais assez profond pour voir les patterns
rf = RandomForestClassifier(
    n_estimators=200, 
    max_depth=15, 
    min_samples_leaf=5, 
    random_state=42, 
    n_jobs=-1,
    class_weight='balanced' # Important car dataset déséquilibré
)

print("⚔️ Launching 10-Fold CV...")
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
scores = cross_val_score(rf, X, y, cv=cv, scoring='f1', n_jobs=-1)

print(f"\n📊 RESULTATS RANDOM FOREST :")
print(f"   🎯 F1 Score Moyen : {scores.mean():.5f} (+/- {scores.std():.4f})")

# Comparatif
print(f"\n🆚 Comparaison :")
print(f"   XGBoost / HistGB : ~0.764")
print(f"   Random Forest    : {scores.mean():.5f}")

if scores.mean() > 0.764:
    print("✅ WOW ! Le Random Forest bat le Boosting ! Il faut creuser.")
else:
    print("❌ Verdict : Le Boosting reste roi (comme souvent sur Kaggle).")
