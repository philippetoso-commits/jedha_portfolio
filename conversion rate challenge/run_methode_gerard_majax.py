import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.base import BaseEstimator, TransformerMixin
import warnings

warnings.filterwarnings('ignore')
SEED = 42
N_FOLDS = 10
ALPHA = 5 # Reduced from 200 to trust the blocks more!

print("="*80)
print("🎩 MÉTHODE GÉRARD MAJAX (Block Decision)")
print("="*80)

# 1. Chargement
print("Chargement des données...")
df_train = pd.read_csv('conversion_data_train.csv')
df_test = pd.read_csv('conversion_data_test.csv')

# 2. Profilage
def create_profiles(df):
    df_c = df.copy()
    df_c['age_bin'] = pd.cut(df_c['age'], bins=[0, 18, 25, 30, 35, 40, 45, 50, 60, 100], labels=False).fillna(-1).astype(int)
    df_c['pages_bin'] = pd.cut(df_c['total_pages_visited'], 
                               bins=[-1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 25, 100], 
                               labels=False).fillna(-1).astype(int)
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

print(f"Nombre de Profils Uniques (Blocs) : {df_train['profile_id'].nunique()}")
print("Exemples de Profils TRAIN :")
print(df_train['profile_id'].head().tolist())
print("Exemples de Profils TEST :")
print(df_test['profile_id'].head().tolist())

# 3. Estimation OOF (Bayesian Target Encoding)
class BayesianTargetEncoderCV(BaseEstimator, TransformerMixin):
    def __init__(self, key_col, target_col, n_folds=5, alpha=20, seed=42):
        self.key_col = key_col
        self.target_col = target_col
        self.n_folds = n_folds
        self.alpha = alpha
        self.seed = seed
        self.global_stats = None
        self.global_mean = None
        
    def fit(self, X, y=None):
        self.global_mean = X[self.target_col].mean()
        self.global_stats = X.groupby(self.key_col)[self.target_col].agg(['count', 'mean'])
        self.global_stats.columns = ['n', 'p']
        self.global_stats['smoothed_val'] = (
            self.global_stats['n'] * self.global_stats['p'] + self.alpha * self.global_mean
        ) / (self.global_stats['n'] + self.alpha)
        return self
        
    def transform(self, X):
        X_out = X.copy()
        mapped = X_out[self.key_col].map(self.global_stats['smoothed_val'])
        X_out['majax_prob'] = mapped.fillna(self.global_mean)
        return X_out

    def fit_transform_oof(self, X, y=None):
        self.fit(X, y)
        X_out = X.copy()
        X_out['majax_prob'] = np.nan
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)
        for train_idx, val_idx in skf.split(X, X[self.target_col]):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            fold_mean = X_tr[self.target_col].mean()
            stats = X_tr.groupby(self.key_col)[self.target_col].agg(['count', 'mean'])
            stats.columns = ['n', 'p']
            stats['smoothed_val'] = (stats['n'] * stats['p'] + self.alpha * fold_mean) / (stats['n'] + self.alpha)
            
            X_out.iloc[val_idx, X_out.columns.get_loc('majax_prob')] = X_val[self.key_col].map(stats['smoothed_val']).fillna(fold_mean)
        return X_out

print("Calcul des probabilités par bloc (OOF)...")
encoder = BayesianTargetEncoderCV(key_col='profile_id', target_col='converted', n_folds=10, alpha=ALPHA)
df_train = encoder.fit_transform_oof(df_train)

print(f"Stats Probabilités Majax (Alpha={ALPHA}):")
print(f"  Min : {df_train['majax_prob'].min():.4f}")
print(f"  Max : {df_train['majax_prob'].max():.4f}")
print(f"  Mean: {df_train['majax_prob'].mean():.4f}")
print(f"  Std : {df_train['majax_prob'].std():.4f}")

# 4. Optimisation du Seuil Majax
print("\nRecherche du Seuil Optimal (Block Optimization)...")
thresholds = np.linspace(0.2, 0.6, 100)
scores = []
y_true = df_train['converted']
probs = df_train['majax_prob']

best_th = 0.5
best_f1 = 0

for th in thresholds:
    # La décision est prise sur la probabilité du PROFIL.
    # Si P(Profil) >= th, ALORS Pred=1 pour tout le monde dans ce profil
    preds = (probs >= th).astype(int)
    f = f1_score(y_true, preds)
    if f > best_f1:
        best_f1 = f
        best_th = th

print(f"🎯 Seuil Optimal : {best_th:.4f}")
print(f"🚀 F1 Score (Validation Train) : {best_f1:.5f}")

# 5. Prediction Test
print("\nApplication au Test Set...")
df_test = encoder.transform(df_test)

print(f"Stats Probabilités TEST (Alpha={ALPHA}):")
print(f"  Min : {df_test['majax_prob'].min():.4f}")
print(f"  Max : {df_test['majax_prob'].max():.4f}")
print(f"  Mean: {df_test['majax_prob'].mean():.4f}")

test_preds = (df_test['majax_prob'] >= best_th).astype(int)

sub_name = 'submission_GERARD_MAJAX.csv'
sub = pd.DataFrame({'converted': test_preds})
sub.to_csv(sub_name, index=False)

print(f"✅ Soumission générée : {sub_name}")
print(f"   Nombre de conversions prédites : {test_preds.sum()}")
print("="*80)
