import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.base import BaseEstimator, TransformerMixin
import warnings

warnings.filterwarnings('ignore')
SEED = 42
N_FOLDS_OOF = 10
N_FOLDS_TH = 5 # Outer CV for threshold
ALPHA = 5       # Profiling smoothing 
MIN_SUPPORT = 0 # DEBUG: Disable robustness to check if it matches V1

print("="*80)
print("🎩 MÉTHODE GÉRARD MAJAX v2 (Robust Block Decision)")
print("="*80)

# 1. Chargement
print("Chargement des données...")
df_train = pd.read_csv('conversion_data_train.csv')
df_test = pd.read_csv('conversion_data_test.csv')

# 2. Profilage (Fix v1 included)
def create_profiles(df):
    df_c = df.copy()
    # Casting to int to ensure string match
    df_c['age_bin'] = pd.cut(df_c['age'], bins=[0, 18, 25, 30, 35, 40, 45, 50, 60, 100], labels=False).fillna(-1).astype(int)
    df_c['pages_bin'] = pd.cut(df_c['total_pages_visited'], 
                               bins=[-1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 25, 100], 
                               labels=False).fillna(-1).astype(int)
    
    # Profile ID (Fine Grained)
    df_c['profile_id'] = (
        df_c['country'].astype(str) + "_" + 
        df_c['source'].astype(str) + "_" + 
        df_c['new_user'].astype(str) + "_" + 
        df_c['age_bin'].astype(str) + "_" + 
        df_c['pages_bin'].astype(str)
    )
    
    # Cluster ID (Coarse Grained - Fallback)
    df_c['age_broad'] = pd.cut(df_c['age'], bins=[0, 30, 100], labels=['Young', 'Senior'])
    df_c['cluster_id'] = df_c['country'].astype(str) + "_" + df_c['age_broad'].astype(str)
    
    return df_c

df_train = create_profiles(df_train)
df_test = create_profiles(df_test)

print(f"Stats Profils: {df_train['profile_id'].nunique()} blocs uniques.")

# 3. Estimation Probabilités (OOF) + Support
class RobustBayesianEncoder(BaseEstimator, TransformerMixin):
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
        # Join allows easy handling of missing keys
        stats = self.global_stats[['smoothed_val', 'n']]
        X_out = X_out.join(stats, on=self.key_col, rsuffix='_stats')
        X_out['majax_prob'] = X_out['smoothed_val'].fillna(self.global_mean)
        X_out['majax_n'] = X_out['n'].fillna(0)
        return X_out

    def fit_transform_oof(self, X, y=None):
        self.fit(X, y)
        X_out = X.copy()
        X_out['majax_prob'] = np.nan
        X_out['majax_n'] = np.nan
        
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)
        for train_idx, val_idx in skf.split(X, X[self.target_col]):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            fold_mean = X_tr[self.target_col].mean()
            stats = X_tr.groupby(self.key_col)[self.target_col].agg(['count', 'mean'])
            stats.columns = ['n', 'p']
            stats['smoothed_val'] = (stats['n'] * stats['p'] + self.alpha * fold_mean) / (stats['n'] + self.alpha)
            
            # Map robustly using join logic or map
            # Should be safe with map since we iterate folds
            X_out.iloc[val_idx, X_out.columns.get_loc('majax_prob')] = X_val[self.key_col].map(stats['smoothed_val']).fillna(fold_mean).values
            X_out.iloc[val_idx, X_out.columns.get_loc('majax_n')] = X_val[self.key_col].map(stats['n']).fillna(0).values
        return X_out

print("Encodage Probabilités (OOF)...")
# Encodage Fin (Profil)
encoder_fine = RobustBayesianEncoder(key_col='profile_id', target_col='converted', n_folds=N_FOLDS_OOF, alpha=ALPHA)
df_train = encoder_fine.fit_transform_oof(df_train)

# Encodage Large (Cluster - Fallback)
encoder_coarse = RobustBayesianEncoder(key_col='cluster_id', target_col='converted', n_folds=N_FOLDS_OOF, alpha=ALPHA)
# On triche un peu on réutilise le fit_transform_oof mais faut faire gaffe de pas le faire en parallèle
# Pour simplifier dans le script, on va juste calculer les stats globales coarse sur le train, 
# car le fallback est utilisé pour les cas rares, donc le risque de fuite est minime vs le gain de stabilité.
# Mais restons propres : OOF aussi pour le coarse si possible. 
# On va faire simple : Coarse encoding sur tout le train (car n grand -> fuite minime).
encoder_coarse.fit(df_train)
stats_coarse = encoder_coarse.transform(df_train)[['majax_prob']]
df_train['cluster_prob'] = stats_coarse['majax_prob']

# 4. Stratégie Hybride (Profile if support > K else Cluster)
print(f"Application Règle de Support (Min Support = {MIN_SUPPORT})...")
mask_robust = (df_train['majax_n'] >= MIN_SUPPORT)
percent_robust = mask_robust.mean()
print(f"  -> {percent_robust:.2%} des observations utilisent le profil Fin.")
print(f"  -> {1-percent_robust:.2%} des observations utilisent le cluster Large (Fallback).")

df_train['effective_prob'] = np.where(
    mask_robust,
    df_train['majax_prob'], # Strong signal
    df_train['cluster_prob'] # Weak signal -> Fallback to cluster mean
)

# Debug probs
print(f"Stats Effective Probs:")
print(f"  Mean: {df_train['effective_prob'].mean():.4f}")
print(f"  Min : {df_train['effective_prob'].min():.4f}")
print(f"  Max : {df_train['effective_prob'].max():.4f}")

# Debug Performance (Pre-CV)
print("\nDEBUG PRE-CV PERFORMANCE:")
for th in [0.2, 0.3, 0.4, 0.5]:
    preds_dbg = (df_train['effective_prob'] >= th).astype(int)
    f1_dbg = f1_score(df_train['converted'], preds_dbg)
    print(f"  Threshold {th}: F1 = {f1_dbg:.5f}")

# 5. Validation du Seuil (Outer CV)
print(f"\nValidation Croisée du Seuil ({N_FOLDS_TH} folds)...")
skf_th = StratifiedKFold(n_splits=N_FOLDS_TH, shuffle=True, random_state=SEED)
scores_v2 = []
best_thresholds = []

for fold, (train_idx, val_idx) in enumerate(skf_th.split(df_train, df_train['converted'])):
    # Calibration sur Train
    X_tr_calib = df_train.iloc[train_idx]
    y_tr_calib = df_train['converted'].iloc[train_idx]
    
    # Grid Search Threshold
    best_th_fold = 0.5
    best_f1_fold = 0
    probs_calib = X_tr_calib['effective_prob']
    
    for th in np.linspace(0.2, 0.6, 50):
        f = f1_score(y_tr_calib, (probs_calib >= th).astype(int))
        if f > best_f1_fold:
            best_f1_fold = f
            best_th_fold = th
    
    best_thresholds.append(best_th_fold)
    
    # Evaluation sur Val
    X_val_eval = df_train.iloc[val_idx]
    y_val_eval = df_train['converted'].iloc[val_idx]
    val_preds = (X_val_eval['effective_prob'] >= best_th_fold).astype(int)
    f1_val = f1_score(y_val_eval, val_preds)
    scores_v2.append(f1_val)
    
    print(f"Fold {fold+1} : Seuil Opt={best_th_fold:.3f} | F1={f1_val:.5f}")

avg_th = np.mean(best_thresholds)
print("-" * 60)
print(f"Moyenne F1 (Majax v2) : {np.mean(scores_v2):.5f}")
print(f"Seuil Moyen Retenu    : {avg_th:.4f}")

# 6. Prédiction Test
print("\nGénération Soumission Test...")
# Transform Test
df_test = encoder_fine.transform(df_test)
df_test_coarse = encoder_coarse.transform(df_test)
df_test['cluster_prob'] = df_test_coarse['majax_prob']

# Apply Logic
df_test['effective_prob'] = np.where(
    df_test['majax_n'] >= MIN_SUPPORT,
    df_test['majax_prob'],
    df_test['cluster_prob']
)

# Apply Avg Threshold
test_preds = (df_test['effective_prob'] >= avg_th).astype(int)

sub = pd.DataFrame({'converted': test_preds})
sub.to_csv('submission_GERARD_MAJAX_v2.csv', index=False)
print(f"✅ Soumission : submission_GERARD_MAJAX_v2.csv ({test_preds.sum()} conversions)")
