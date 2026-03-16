import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.metrics import f1_score, roc_auc_score, recall_score, precision_recall_curve
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings('ignore')

# Configuration
ALPHA_RANGE = [1, 5, 10, 20, 50, 100, 200]
N_FOLDS = 10
SEED = 42

print("="*80)
print("📚 MÉTHODE PROF : Consolidation & Validation Rigoureuse")
print("="*80)

# 1. Chargement et Création des Profils
# -----------------------------------------------------------------------------
print("\n[1/5] Chargement et Préparation des données...")
df_train = pd.read_csv('conversion_data_train.csv')
df_test = pd.read_csv('conversion_data_test.csv')

def create_profiles(df):
    df_c = df.copy()
    
    # Discrétisation (Binning)
    # Age: Capturer les comportements par tranche
    df_c['age_bin'] = pd.cut(df_c['age'], bins=[0, 18, 25, 30, 35, 40, 45, 50, 60, 100], labels=False)
    
    # Pages: Le signal le plus fort, granularité fine au début
    df_c['pages_bin'] = pd.cut(df_c['total_pages_visited'], 
                               bins=[-1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 25, 100], 
                               labels=False)
    
    # Création de la clé de Profil
    # Profile = Country + Source + NewUser + AgeBin + PagesBin
    df_c['profile_id'] = (
        df_c['country'].astype(str) + "_" + 
        df_c['source'].astype(str) + "_" + 
        df_c['new_user'].astype(str) + "_" + 
        df_c['age_bin'].astype(str) + "_" + 
        df_c['pages_bin'].astype(str)
    )
    return df_c

df_train = create_profiles(df_train)
df_test = create_profiles(df_test)
print(f"   Nombre de profils uniques (Train): {df_train['profile_id'].nunique()}")

# 2. Classe d'Encodage Target Encoding Out-of-Fold (Strict)
# -----------------------------------------------------------------------------
class BayesianTargetEncoderCV(BaseEstimator, TransformerMixin):
    def __init__(self, key_col, target_col, n_folds=5, alpha=20, seed=42):
        self.key_col = key_col
        self.target_col = target_col
        self.n_folds = n_folds
        self.alpha = alpha
        self.seed = seed
        self.global_mean = None
        self.global_stats = None
        
    def fit(self, X, y=None):
        # Calculer les stats globales pour l'inférence sur le Test set (et non pour le Train !)
        self.global_mean = X[self.target_col].mean()
        self.global_stats = X.groupby(self.key_col)[self.target_col].agg(['count', 'mean'])
        self.global_stats.columns = ['n', 'p']
        
        # Smoothed global value
        self.global_stats['smoothed_val'] = (
            self.global_stats['n'] * self.global_stats['p'] + self.alpha * self.global_mean
        ) / (self.global_stats['n'] + self.alpha)
        
        return self
        
    def transform(self, X):
        # Note: Cette méthode transform est pour le TEST set ou la PROD
        # Pour le TRAIN set, il faut utiliser fit_transform qui fait le OOF
        
        X_out = X.copy()
        
        # Mapping avec les stats globales (apprises sur tout le train)
        # Si le profil est inconnu --> Global Mean (avec prior)
        mapped = X_out[self.key_col].map(self.global_stats['smoothed_val'])
        params_n = X_out[self.key_col].map(self.global_stats['n'])
        
        X_out['smoothed_prob'] = mapped.fillna(self.global_mean) # Fallback safe
        X_out['profile_support'] = params_n.fillna(0)
        
        return X_out

    def fit_transform_oof(self, X, y=None):
        # C'est ici que la magie OOF opère pour le Train set
        self.fit(X, y) # On fit les stats globales pour plus tard (Test)
        
        X_out = X.copy()
        X_out['smoothed_prob'] = np.nan
        X_out['profile_support'] = np.nan
        
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)
        
        for train_idx, val_idx in skf.split(X, X[self.target_col]):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            
            # Calcul des stats sur les K-1 folds
            fold_mean = X_tr[self.target_col].mean()
            stats = X_tr.groupby(self.key_col)[self.target_col].agg(['count', 'mean'])
            stats.columns = ['n', 'p']
            
            # Smoothing Bayésien
            stats['smoothed_val'] = (stats['n'] * stats['p'] + self.alpha * fold_mean) / (stats['n'] + self.alpha)
            
            # Application sur le fold courant (Validation)
            mapped_val = X_val[self.key_col].map(stats['smoothed_val'])
            mapped_cnt = X_val[self.key_col].map(stats['n'])
            
            X_out.loc[val_idx, 'smoothed_prob'] = mapped_val.fillna(fold_mean)
            X_out.loc[val_idx, 'profile_support'] = mapped_cnt.fillna(0)
            
        return X_out

# 3. Analyse de Sensibilité du paramètre Alpha
# -----------------------------------------------------------------------------
print("\n[2/5] Analyse de sensibilité (Alpha Tuning)...")

results_alpha = []

# Préparation features de base pour le modèle de stacking
# On encode Country et Source simplement
le_country = LabelEncoder().fit(pd.concat([df_train['country'], df_test['country']]))
le_source = LabelEncoder().fit(pd.concat([df_train['source'], df_test['source']]))

def prepare_features_for_model(df):
    df_mod = df.copy()
    df_mod['country_enc'] = le_country.transform(df_mod['country'])
    df_mod['source_enc'] = le_source.transform(df_mod['source'])
    # Features finales: Originales + Supervisées
    features = ['country_enc', 'source_enc', 'total_pages_visited', 'new_user', 'age', 'smoothed_prob', 'profile_support']
    return df_mod[features]

best_alpha = 20
best_score = 0

for alpha_val in ALPHA_RANGE:
    # 1. Encodage OOF
    encoder = BayesianTargetEncoderCV(key_col='profile_id', target_col='converted', n_folds=5, alpha=alpha_val)
    df_train_encoded = encoder.fit_transform_oof(df_train)
    
    # 2. Préparation X, y
    X_enc = prepare_features_for_model(df_train_encoded)
    y_enc = df_train['converted']
    
    # 3. Évaluation rapide (Logistic Regression pour la rapidité ou XGB)
    # On utilise XGB car c'est le modèle cible
    model = XGBClassifier(eval_metric='logloss', use_label_encoder=False, random_state=SEED, n_jobs=1)
    
    # CV du Modèle (On a déjà fait OOF pour les features, donc c'est un "Stacking" valide)
    scores = cross_val_score(model, X_enc, y_enc, cv=5, scoring='f1')
    mean_score = scores.mean()
    
    print(f"   Alpha = {alpha_val:3d} | F1-Score = {mean_score:.5f}")
    results_alpha.append((alpha_val, mean_score))
    
    if mean_score > best_score:
        best_score = mean_score
        best_alpha = alpha_val

print(f"👉 Meilleur Alpha retenu : {best_alpha}")

# 4. Entraînement Final & Optimisation du Seuil
# -----------------------------------------------------------------------------
print(f"\n[3/5] Entraînement Final (Alpha={best_alpha})...")

# Encodage Final Train
final_encoder = BayesianTargetEncoderCV(key_col='profile_id', target_col='converted', n_folds=10, alpha=best_alpha)
df_train_final = final_encoder.fit_transform_oof(df_train)

X_train_final = prepare_features_for_model(df_train_final)
y_train = df_train['converted']

# Modèle Final
final_model = XGBClassifier(
    n_estimators=200, 
    learning_rate=0.03, 
    max_depth=6, 
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='logloss', 
    use_label_encoder=False, 
    random_state=SEED,
    n_jobs=-1
)

# Cross-Val Predict pour optimiser le seuil SANS fuite
print("   Génération des prédictions OOF pour optimisation du seuil...")
oof_preds_proba = cross_val_predict(final_model, X_train_final, y_train, cv=10, method='predict_proba')[:, 1]

# Optimisation du seuil
precisions, recalls, thresholds = precision_recall_curve(y_train, oof_preds_proba)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
# Note: len(thresholds) == len(f1_scores) - 1 usually, need care
thresholds = np.append(thresholds, 1) 
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]
optimal_f1 = f1_scores[optimal_idx]

print(f"   Seuil Optimal trouvé (CV) : {optimal_threshold:.4f}")
print(f"   F1-Score attendu (CV)     : {optimal_f1:.4f}")

# 5. Application sur le Test Set & Génération Fichier
# -----------------------------------------------------------------------------
print("\n[4/5] Prédiction sur le Test Set et génération submission...")

# Retrain sur tout le train
final_model.fit(X_train_final, y_train)

# Encoder le Test Set (transform standard utilisant les stats globales)
df_test_encoded = final_encoder.transform(df_test)
X_test_final = prepare_features_for_model(df_test_encoded)

# Predire
test_probs = final_model.predict_proba(X_test_final)[:, 1]
test_preds = (test_probs >= optimal_threshold).astype(int)

# 6. Baseline Comparison (Pédagogique)
# -----------------------------------------------------------------------------
print("\n[5/5] Comparaison Baseline Pédagogique...")
# Prepare raw features manually
df_base = df_train[['country', 'source', 'total_pages_visited', 'new_user', 'age']].copy()
df_base['country_enc'] = le_country.transform(df_base['country'])
df_base['source_enc'] = le_source.transform(df_base['source'])
features_base = ['country_enc', 'source_enc', 'total_pages_visited', 'new_user', 'age']
X_base = df_base[features_base]

base_model = LogisticRegression(max_iter=1000)
base_results = cross_val_score(base_model, X_base, y_train, cv=10, scoring='f1')
print(f"   Baseline (LogReg simple) : {base_results.mean():.4f}")
print(f"   Gain Méthode Prof        : +{(optimal_f1 - base_results.mean())*100:.2f} points")

# Save
submission = pd.DataFrame({'converted': test_preds})
submission.to_csv('submission_METHODE_PROF.csv', index=False)
print("\n✅ Fichier 'submission_METHODE_PROF.csv' généré.")
