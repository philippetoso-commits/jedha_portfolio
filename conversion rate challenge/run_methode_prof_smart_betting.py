import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import f1_score
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings('ignore')

# Configuration
ALPHA = 200
SEED = 42
N_FOLDS = 10

print("="*80)
print("🎰 MÉTHODE PROF + SMART BETTING (Casino Edition)")
print("="*80)

# 1. Chargement et Feature Engineering (Méthode Prof Standard)
# -----------------------------------------------------------------------------
df_train = pd.read_csv('conversion_data_train.csv')
df_test = pd.read_csv('conversion_data_test.csv')

def create_profiles(df):
    df_c = df.copy()
    df_c['age_bin'] = pd.cut(df_c['age'], bins=[0, 18, 25, 30, 35, 40, 45, 50, 60, 100], labels=False)
    # Cluster definition for Betting (Simpler than Profile)
    # We use Country + Broad Age Group (Young < 30, Senior > 30)
    df_c['age_broad'] = pd.cut(df_c['age'], bins=[0, 30, 100], labels=['Young', 'Senior'])
    df_c['cluster_betting'] = df_c['country'].astype(str) + "_" + df_c['age_broad'].astype(str)
    
    df_c['pages_bin'] = pd.cut(df_c['total_pages_visited'], 
                               bins=[-1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 25, 100], 
                               labels=False)
    
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

# Encodage LabelEncoder
le_country = LabelEncoder().fit(pd.concat([df_train['country'], df_test['country']]))
le_source = LabelEncoder().fit(pd.concat([df_train['source'], df_test['source']]))

def prepare_features(df):
    df_m = df.copy()
    df_m['country_enc'] = le_country.transform(df_m['country'])
    df_m['source_enc'] = le_source.transform(df_m['source'])
    return df_m[['country_enc', 'source_enc', 'total_pages_visited', 'new_user', 'age', 'smoothed_prob', 'profile_support']]

# 2. Validation Croisée avec Smart Betting
# -----------------------------------------------------------------------------
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

f1_global_standard = []
f1_global_smart = []

print(f"Lancement de la validation ({N_FOLDS} folds)...")

# Pre-calcul OOF Features (Pour gagner du temps, on fait un seul passage propre)
# On simule le OOF pour l'entraînement du modèle
# Pour simplifier dans ce script de démo, on fait le OOF "in-place" dans la boucle principale CV
# C'est moins efficace mais plus lisible.

for fold, (train_idx, val_idx) in enumerate(skf.split(df_train, df_train['converted'])):
    X_tr = df_train.iloc[train_idx].copy()
    X_val = df_train.iloc[val_idx].copy()
    y_tr, y_val = X_tr['converted'], X_val['converted']
    
    # A. Encodage Supervisé (Training du fold)
    stats = X_tr.groupby('profile_id')['converted'].agg(['count', 'mean'])
    stats.columns = ['n', 'p']
    g_mean = y_tr.mean()
    stats['smoothed_val'] = (stats['n'] * stats['p'] + ALPHA * g_mean) / (stats['n'] + ALPHA)
    
    # Mapping
    X_val['smoothed_prob'] = X_val['profile_id'].map(stats['smoothed_val']).fillna(g_mean)
    X_val['profile_support'] = X_val['profile_id'].map(stats['n']).fillna(0)
    
    # Pour X_tr, on doit faire un OOF interne ou (pour simplifier le run) utiliser le Leave-One-Out approximation
    # ou simplement spliter X_tr en 2 pour apprendre les features.
    # Ici, pour maximiser la performance, on va faire un KFold interne rapide pour X_tr.
    kf_in = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    X_tr['smoothed_prob'] = np.nan
    X_tr['profile_support'] = np.nan
    for t_i, v_i in kf_in.split(X_tr, y_tr):
        x_ti, x_vi = X_tr.iloc[t_i], X_tr.iloc[v_i]
        s_i = x_ti.groupby('profile_id')['converted'].agg(['count', 'mean'])
        s_i.columns = ['n', 'p']
        gm_i = x_ti['converted'].mean()
        s_i['sv'] = (s_i['n'] * s_i['p'] + ALPHA * gm_i) / (s_i['n'] + ALPHA)
        X_tr.iloc[v_i, X_tr.columns.get_loc('smoothed_prob')] = x_vi['profile_id'].map(s_i['sv']).fillna(gm_i)
        X_tr.iloc[v_i, X_tr.columns.get_loc('profile_support')] = x_vi['profile_id'].map(s_i['n']).fillna(0)
    
    X_tr['smoothed_prob'] = X_tr['smoothed_prob'].fillna(g_mean)
    X_tr['profile_support'] = X_tr['profile_support'].fillna(0)
    
    # B. Entraînement Modèle (XGBoost)
    model = XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=6, use_label_encoder=False, eval_metric='logloss', random_state=SEED, n_jobs=1)
    
    X_tr_feat = prepare_features(X_tr)
    X_val_feat = prepare_features(X_val)
    
    model.fit(X_tr_feat, y_tr)
    
    # Prédictions Probabilistes
    probs_tr = model.predict_proba(X_tr_feat)[:, 1]
    probs_val = model.predict_proba(X_val_feat)[:, 1]
    
    # C.1 Optimisation Standard (Seuil Global)
    thresholds = np.linspace(0.1, 0.9, 100)
    best_th_global = 0.5
    best_f1_global = 0
    for th in thresholds:
        s = f1_score(y_tr, (probs_tr >= th).astype(int))
        if s > best_f1_global:
            best_f1_global = s
            best_th_global = th
            
    preds_val_standard = (probs_val >= best_th_global).astype(int)
    score_standard = f1_score(y_val, preds_val_standard)
    f1_global_standard.append(score_standard)
    
    # C.2 Optimisation Smart Betting (Seuil par Cluster)
    # Clusters: Country_AgeBroad
    clusters = X_tr['cluster_betting'].unique()
    cluster_thresholds = {}
    
    for cl in clusters:
        # Masque Cluster Training
        m_tr = (X_tr['cluster_betting'] == cl)
        if m_tr.sum() < 50: # Pas assez de data pour optimiser
            cluster_thresholds[cl] = best_th_global
            continue
            
        y_c = y_tr[m_tr]
        p_c = probs_tr[m_tr]
        
        if y_c.sum() == 0:
            cluster_thresholds[cl] = 0.99
            continue
            
        best_th_cl = 0.5
        best_f1_cl = 0
        
        # Grid Search Local
        for th in np.linspace(0.1, 0.9, 50):
            s = f1_score(y_c, (p_c >= th).astype(int))
            if s > best_f1_cl:
                best_f1_cl = s
                best_th_cl = th
        cluster_thresholds[cl] = best_th_cl
        
    # Application sur Validation
    preds_val_smart = np.zeros(len(X_val))
    # On itère sur les clusters présents dans val
    for cl in X_val['cluster_betting'].unique():
        m_val = (X_val['cluster_betting'] == cl)
        th = cluster_thresholds.get(cl, best_th_global) # Fallback sur global si cluster inconnu
        preds_val_smart[m_val] = (probs_val[m_val] >= th).astype(int)
        
    score_smart = f1_score(y_val, preds_val_smart)
    f1_global_smart.append(score_smart)
    
    print(f"   Fold {fold+1} | Standard: {score_standard:.5f} | Smart: {score_smart:.5f} | Gain: {score_smart - score_standard:.5f}")

print("\n" + "="*80)
print("🏆 RÉSULTATS FINAUX")
print("-" * 60)
print(f"Standard Mean F1 : {np.mean(f1_global_standard):.5f}")
print(f"Smart Betting F1 : {np.mean(f1_global_smart):.5f}")
print(f"Gain Moyen       : {np.mean(f1_global_smart) - np.mean(f1_global_standard):.5f} pts")
print("-" * 60)

from sklearn.base import BaseEstimator, TransformerMixin

# ==========================================
# GENERATION SOUMISSION FINALE (RETRAIN FULL)
# ==========================================
print("\n📝 Génération de la soumission finale (Retrain SUR TOUT LE TRAIN)...")

# Definition de la classe AVANT usage
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
        self.global_stats['smoothed_val'] = (self.global_stats['n'] * self.global_stats['p'] + self.alpha * self.global_mean) / (self.global_stats['n'] + self.alpha)
        return self

    def transform(self, X):
        X_out = X.copy()
        X_out['smoothed_prob'] = X_out[self.key_col].map(self.global_stats['smoothed_val']).fillna(self.global_mean)
        X_out['profile_support'] = X_out[self.key_col].map(self.global_stats['n']).fillna(0)
        return X_out

    def fit_transform_oof(self, X, y=None):
        self.fit(X, y) # Fit global stats first
        X_out = X.copy()
        X_out['smoothed_prob'] = np.nan
        X_out['profile_support'] = np.nan
        skf_enc = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)
        
        for t_idx, v_idx in skf_enc.split(X, X[self.target_col]):
            xt, xv = X.iloc[t_idx], X.iloc[v_idx]
            fold_mean = xt[self.target_col].mean()
            stats = xt.groupby(self.key_col)[self.target_col].agg(['count', 'mean'])
            stats.columns = ['n', 'p']
            stats['sv'] = (stats['n']*stats['p'] + self.alpha*fold_mean)/(stats['n']+self.alpha)
            
            X_out.iloc[v_idx, X_out.columns.get_loc('smoothed_prob')] = xv[self.key_col].map(stats['sv']).fillna(fold_mean)
            X_out.iloc[v_idx, X_out.columns.get_loc('profile_support')] = xv[self.key_col].map(stats['n']).fillna(0)
        return X_out

# 1. Encodage OOF sur tout le Train (pour avoir les stats pour le modèle)
encoder = BayesianTargetEncoderCV(key_col='profile_id', target_col='converted', n_folds=10, alpha=ALPHA)

df_train_enc = encoder.fit_transform_oof(df_train)
X_train_final = prepare_features(df_train_enc)
y_train_final = df_train['converted']

# Train Model
model_final = XGBClassifier(n_estimators=200, learning_rate=0.03, max_depth=6, subsample=0.8, colsample_bytree=0.8, eval_metric='logloss', use_label_encoder=False, random_state=SEED, n_jobs=1)
model_final.fit(X_train_final, y_train_final)

# Thresholds per Cluster (Smart Betting) on Train Probs
# We need OOF probs for threshold tuning (as seen in CV)
probs_train_oof = cross_val_predict(model_final, X_train_final, y_train_final, cv=10, method='predict_proba', n_jobs=-1)[:, 1]

cluster_thresholds_final = {}
for cl in df_train_enc['cluster_betting'].unique():
    m = (df_train_enc['cluster_betting'] == cl)
    if m.sum() < 50:
        cluster_thresholds_final[cl] = 0.5
        continue
    y_c = y_train_final[m]
    p_c = probs_train_oof[m]
    if y_c.sum() == 0:
        best_th = 0.99
    else:
        best_th = 0.5
        best_f = 0
        for th in np.linspace(0.1, 0.9, 100):
            if f1_score(y_c, (p_c >= th).astype(int)) > best_f:
                best_f = f1_score(y_c, (p_c >= th).astype(int))
                best_th = th
        cluster_thresholds_final[cl] = best_th

# Predict Test
df_test_enc = encoder.transform(df_test)
X_test_final = prepare_features(df_test_enc)
probs_test = model_final.predict_proba(X_test_final)[:, 1]

preds_test = np.zeros(len(df_test))
for cl in df_test_enc['cluster_betting'].unique():
    m = (df_test_enc['cluster_betting'] == cl)
    th = cluster_thresholds_final.get(cl, 0.5)
    preds_test[m] = (probs_test[m] >= th).astype(int)

print(f"Nombre total de conversions prédites : {int(preds_test.sum())}")
sub = pd.DataFrame({'converted': preds_test.astype(int)})
sub.to_csv('submission_METHODE_PROF_v2.csv', index=False)
