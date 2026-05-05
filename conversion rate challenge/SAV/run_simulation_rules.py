
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, classification_report
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import cross_val_score, StratifiedKFold

print("📜 RULE-BASED 'HUMAN' MODEL EVALUATION 📜")

class HumanRulesClassifier(BaseEstimator, ClassifierMixin):
    def fit(self, X, y=None):
        return self
        
    def predict(self, X):
        # Initialisation : Tout le monde à 0 par défaut
        # (Car le taux de conversion de base est 3%, donc 0 est le meilleur guess par défaut)
        preds = np.zeros(len(X), dtype=int)
        
        # 1. RÈGLE D'OR : Pages > 20 (Probabilité = 100%)
        # C'est la règle absolue.
        mask_sure_win = (X['total_pages_visited'] > 18)
        preds[mask_sure_win] = 1
        
        # 2. RÈGLE D'ACIER : Pages < 8 (Probabilité ~0%)
        # Même si le modèle ML hésite, nous on sait.
        # (On laisse à 0)
        
        # 3. RÈGLE CHINE : (Probabilité ~0.1%)
        # On force 0 pour la Chine sauf cas extrêmes (>20 pages)
        mask_china = (X['country'] == 'China')
        preds[mask_china] = 0 # Reset même si >18? Non, >18 prime. 
        # On remet >18 à 1 car c'est physique.
        preds[mask_sure_win] = 1
        
        # 4. LE BOOST ERASMUS (Jeunes Européens, >12 pages)
        mask_erasmus = (
            (X['country'].isin(['Germany', 'UK'])) & 
            (X['age'] < 25) & 
            (X['total_pages_visited'] >= 11)   # On est agressif (11)
        )
        preds[mask_erasmus] = 1
        
        # 5. LE BOOST AMERICAN HUSTLE (US, 20-30, >11 pages)
        mask_usa = (
            (X['country'] == 'US') & 
            (X['age'].between(20, 30)) & 
            (X['total_pages_visited'] >= 11)
        )
        preds[mask_usa] = 1
        
        # 6. LE "VENTRE MOU" (Les cas ambigus : Pages 10-18 hors cibles)
        # Ici, sans ML, c'est dur. On peut dire >14 pages convertit souvent.
        mask_standard = (X['total_pages_visited'] >= 14)
        preds[mask_standard] = 1
        
        return preds

# EVALUATION
print("📥 Loading Data...")
df = pd.read_csv('conversion rate challenge/conversion_data_train.csv')
X = df.drop('converted', axis=1)
y = df['converted']

print("⚙️ Evaluating Hand-Crafted Rules (10-Fold CV)...")
model = HumanRulesClassifier()
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv, scoring='f1', n_jobs=-1)

print(f"\n📊 RESULTATS DU MODÈLE 'HUMAIN' :")
print(f"   🎯 F1 Score Moyen : {scores.mean():.5f} (+/- {scores.std():.4f})")

# Comparaison
print(f"\n🆚 Comparaison :")
print(f"   RegLog Basique : ~0.72")
print(f"   XGBoost Tuné   : ~0.764")
print(f"   Syndicat USA   : ~0.77+ (Espéré)")

if scores.mean() > 0.73:
    print("\n✅ WOW : Vos règles battent une Régression Logistique !")
else:
    print("\n⚠️ Pas mal, mais le ML reste utile pour les cas fins.")
