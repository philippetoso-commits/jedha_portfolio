import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings('ignore')

# Configuration
ALPHA = 200  # Best alpha from previous step
SEED = 42
N_FOLDS = 10
S_MIN = 15       # Support min pour considérer le profil fiable
EPSILON = 0.05   # Zone d'ambiguïté [0.05, 0.95]

print("="*80)
print("🔬 ÉVALUATION CIBLÉE : PROFILS AMBIGUS")
print("="*80)
print(f"Paramètres : Alpha={ALPHA}, S_min={S_MIN}, Epsilon={EPSILON}")

# 1. Chargement et Préparation
# -----------------------------------------------------------------------------
df_train = pd.read_csv('conversion_data_train.csv')

# Profiling (Méthode Prof)
def create_profiles(df):
    df_c = df.copy()
    df_c['age_bin'] = pd.cut(df_c['age'], bins=[0, 18, 25, 30, 35, 40, 45, 50, 60, 100], labels=False)
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

# Encodage LabelEncoder pour les modèles
le_country = LabelEncoder().fit(df_train['country'])
le_source = LabelEncoder().fit(df_train['source'])

def prepare_features(df, use_supervised=False):
    df_m = df.copy()
    df_m['country_enc'] = le_country.transform(df_m['country'])
    df_m['source_enc'] = le_source.transform(df_m['source'])
    
    base_feats = ['country_enc', 'source_enc', 'total_pages_visited', 'new_user', 'age']
    if use_supervised:
        return df_m[base_feats + ['smoothed_prob', 'profile_support']]
    return df_m[base_feats]

# 2. Boucle d'Évaluation (10-Fold OOF Strict)
# -----------------------------------------------------------------------------
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

metrics_global_base = []
metrics_global_prof = []
metrics_ambig_base = []
metrics_ambig_prof = []

print(f"\nLancement de la validation croisée ({N_FOLDS} folds)...")

for fold, (train_idx, val_idx) in enumerate(skf.split(df_train, df_train['converted'])):
    X_tr, X_val = df_train.iloc[train_idx].copy(), df_train.iloc[val_idx].copy()
    y_tr, y_val = X_tr['converted'], X_val['converted']
    
    # A. Apprentissage Statistique du Profil (Training Fold)
    stats = X_tr.groupby('profile_id')['converted'].agg(['count', 'mean'])
    stats.columns = ['n', 'p']
    global_mean = y_tr.mean()
    stats['smoothed_val'] = (stats['n'] * stats['p'] + ALPHA * global_mean) / (stats['n'] + ALPHA)
    
    # B. Mapping sur Validation Fold (OOF)
    X_val['smoothed_prob'] = X_val['profile_id'].map(stats['smoothed_val']).fillna(global_mean)
    X_val['profile_support'] = X_val['profile_id'].map(stats['n']).fillna(0)
    
    # Note: Pour le training du modèle 'Prof', on doit aussi avoir ces features sur le Train set
    # Pour faire ça proprement sans fuite interne au train, il faudrait refaire un OOF interne.
    # MAIS, pour simplifier et suivre la logique standard : on peut utiliser un Leave-One-Out ou K-Fold interne.
    # Ici, pour l'évaluation COMPARATIVE, on va utiliser la validation.
    # Pour entraîner le modèle 'Prof' sur X_tr, on va faire un OOF interne rapide sur X_tr.
    
    # --- Interne OOF pour X_tr ---
    kf_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    X_tr['smoothed_prob'] = np.nan
    X_tr['profile_support'] = np.nan
    
    for t_in, v_in in kf_inner.split(X_tr, y_tr):
        x_ti = X_tr.iloc[t_in]
        x_vi = X_tr.iloc[v_in]
        s_i = x_ti.groupby('profile_id')['converted'].agg(['count', 'mean'])
        s_i.columns = ['n', 'p']
        gm_i = x_ti['converted'].mean()
        s_i['sv'] = (s_i['n'] * s_i['p'] + ALPHA * gm_i) / (s_i['n'] + ALPHA)
        
        X_tr.iloc[v_in, X_tr.columns.get_loc('smoothed_prob')] = x_vi['profile_id'].map(s_i['sv']).fillna(gm_i)
        X_tr.iloc[v_in, X_tr.columns.get_loc('profile_support')] = x_vi['profile_id'].map(s_i['n']).fillna(0)
        
    X_tr['smoothed_prob'] = X_tr['smoothed_prob'].fillna(global_mean) # safety
    X_tr['profile_support'] = X_tr['profile_support'].fillna(0)
    
    # C. Définition des Modèles (Même complexité : XGBoost Optimisé)
    params = {
        'n_estimators': 200,
        'learning_rate': 0.03,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'use_label_encoder': False,
        'eval_metric': 'logloss',
        'random_state': SEED,
        'n_jobs': 1
    }
    model_base = XGBClassifier(**params)
    model_prof = XGBClassifier(**params)
    
    # Preparation Features
    X_tr_base = prepare_features(X_tr, use_supervised=False)
    X_tr_prof = prepare_features(X_tr, use_supervised=True)
    
    X_val_base = prepare_features(X_val, use_supervised=False)
    X_val_prof = prepare_features(X_val, use_supervised=True)
    
    # Entraînement
    model_base.fit(X_tr_base, y_tr)
    model_prof.fit(X_tr_prof, y_tr)
    
    # Prédictions Probabilistes
    proba_tr_base = model_base.predict_proba(X_tr_base)[:, 1]
    proba_tr_prof = model_prof.predict_proba(X_tr_prof)[:, 1]
    
    proba_val_base = model_base.predict_proba(X_val_base)[:, 1]
    proba_val_prof = model_prof.predict_proba(X_val_prof)[:, 1]
    
    # Optimisation du Seuil sur le Train (F1-Check simple)
    # On teste 50 seuils entre 0.1 et 0.9 pour aller vite
    thresholds = np.linspace(0.1, 0.9, 50)
    best_th_base, best_f1_base = 0.5, 0
    best_th_prof, best_f1_prof = 0.5, 0
    
    for th in thresholds:
        f1_b = f1_score(y_tr, (proba_tr_base >= th).astype(int))
        if f1_b > best_f1_base:
            best_f1_base = f1_b
            best_th_base = th
            
        f1_p = f1_score(y_tr, (proba_tr_prof >= th).astype(int))
        if f1_p > best_f1_prof:
            best_f1_prof = f1_p
            best_th_prof = th
            
    # Application du Seuil Optimal sur Validation
    p_base = (proba_val_base >= best_th_base).astype(int)
    p_prof = (proba_val_prof >= best_th_prof).astype(int)
    
    metrics_global_base.append(f1_score(y_val, p_base))
    metrics_global_prof.append(f1_score(y_val, p_prof))
    
    # D. Identification des Profils Ambigus
    # Critère : Support suffisant ET Probabilité incertaine (pas 0 ni 1)
    mask_ambig = (
        (X_val['profile_support'] >= S_MIN) & 
        (X_val['smoothed_prob'] > EPSILON) & 
        (X_val['smoothed_prob'] < (1 - EPSILON))
    )
    
    if mask_ambig.sum() > 0:
        y_ambig = y_val[mask_ambig]
        p_base_ambig = p_base[mask_ambig]
        p_prof_ambig = p_prof[mask_ambig]
        
        metrics_ambig_base.append(f1_score(y_ambig, p_base_ambig))
        metrics_ambig_prof.append(f1_score(y_ambig, p_prof_ambig))
    
    print(f"   Fold {fold+1}: {mask_ambig.sum()} profils ambigus trouvés.")

# 3. Résultats et Analyse
# -----------------------------------------------------------------------------
f1_global_base = np.mean(metrics_global_base)
f1_global_prof = np.mean(metrics_global_prof)
f1_ambig_base = np.mean(metrics_ambig_base)
f1_ambig_prof = np.mean(metrics_ambig_prof)

print("\n" + "="*80)
print("📊 RÉSULTATS DE L'ÉVALUATION")
print("="*80)
print(f"Partie Ambiguë du dataset (~{np.mean([m for m in metrics_ambig_base]) if metrics_ambig_base else 0:.0f} échantillons/fold)")
print("-" * 60)
print(f"{'Scope':<20} | {'Modele':<15} | {'F1-Score':<10} | {'Gain':<10}")
print("-" * 60)
print(f"{'GLOBAL':<20} | {'Baseline':<15} | {f1_global_base:.5f}     | -")
print(f"{'GLOBAL':<20} | {'Méthode Prof':<15} | {f1_global_prof:.5f}     | +{(f1_global_prof - f1_global_base)*100:.2f} pts")
print("-" * 60)
print(f"{'ZONE AMBIGUË':<20} | {'Baseline':<15} | {f1_ambig_base:.5f}     | -")
print(f"{'ZONE AMBIGUË':<20} | {'Méthode Prof':<15} | {f1_ambig_prof:.5f}     | +{(f1_ambig_prof - f1_ambig_base)*100:.2f} pts")
print("-" * 60)

print("\n🧪 INTERPRÉTATION")
if (f1_ambig_prof - f1_ambig_base) > (f1_global_prof - f1_global_base):
    print("✅ HYPOTHÈSE VALIDÉE : L'amélioration est nettement plus forte sur les profils ambigus.")
    print("   Cela confirme que la méthode agit comme un 'réducteur de bruit' là où la décision est difficile.")
else:
    print("❌ HYPOTHÈSE NON VALIDÉE : Les gains sont uniformes ou absents.")
