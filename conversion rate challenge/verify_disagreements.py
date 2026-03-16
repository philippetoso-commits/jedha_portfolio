import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.ensemble import VotingClassifier, HistGradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.base import BaseEstimator, TransformerMixin
import warnings

warnings.filterwarnings('ignore')

SEED = 42
N_FOLDS = 5 # Speed up checks

print("="*80)
print("🧐 VÉRIFICATION DES DÉSACCORDS (Train Set Cross-Validation)")
print("="*80)

# 1. LOAD DATA
df_train = pd.read_csv('conversion rate challenge/conversion_data_train.csv')
y = df_train['converted']

# 2. DEFINITION MODELE A : "LE SENAT" (Proxy: Ensemble + Rules)
# -------------------------------------------------------------
# On reconstruit un pipeline similaire à "The Senate"
def features_senate(df):
    d = df.copy()
    d['pages_per_age'] = d['total_pages_visited'] / (d['age'] + 0.1)
    d['interaction'] = d['total_pages_visited'] * d['age']
    return d

numeric_feats = ['age', 'total_pages_visited', 'pages_per_age', 'interaction']
cat_feats = ['country', 'source', 'new_user']

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_feats),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_feats)
])

clf1 = XGBClassifier(n_estimators=100, max_depth=6, random_state=SEED, n_jobs=1, eval_metric='logloss', use_label_encoder=False)
clf2 = LGBMClassifier(n_estimators=100, random_state=SEED, n_jobs=1, verbose=-1)
clf3 = HistGradientBoostingClassifier(max_iter=100, random_state=SEED)

ensemble = VotingClassifier([('xgb', clf1), ('lgb', clf2), ('hgb', clf3)], voting='soft')
pipe_senate = Pipeline([
    ('prep', preprocessor),
    ('model', ensemble)
])

# 3. DEFINITION MODELE B : "METHODE PROF v2" (OOF + Smart Betting)
# ----------------------------------------------------------------
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
        # Smoothing global
        self.global_stats['sv'] = (self.global_stats['n']*self.global_stats['p'] + self.alpha*self.global_mean)/(self.global_stats['n']+self.alpha)
        return self
    
    def transform(self, X):
        # Pour le script de verif, on a besoin que du OOF loop manuel
        return X

def create_profiles(df):
    df_c = df.copy()
    df_c['age_bin'] = pd.cut(df_c['age'], bins=[0, 18, 25, 30, 35, 40, 45, 50, 60, 100], labels=False)
    df_c['pages_bin'] = pd.cut(df_c['total_pages_visited'], bins=[-1,1,2,3,4,5,6,7,8,9,10,12,14,16,18,20,25,100], labels=False)
    df_c['profile_id'] = df_c['country'].astype(str)+"_"+df_c['source'].astype(str)+"_"+df_c['new_user'].astype(str)+"_"+df_c['age_bin'].astype(str)+"_"+df_c['pages_bin'].astype(str)
    
    df_c['age_broad'] = pd.cut(df_c['age'], bins=[0, 30, 100], labels=['Young', 'Senior'])
    df_c['cluster_betting'] = df_c['country'].astype(str) + "_" + df_c['age_broad'].astype(str)
    return df_c

# Encoders simples pour Model Prof
le_country = LabelEncoder().fit(df_train['country'])
le_source = LabelEncoder().fit(df_train['source'])

def feats_prof_prep(df):
    d = df.copy()
    d['country_enc'] = le_country.transform(d['country'])
    d['source_enc'] = le_source.transform(d['source'])
    return d[['country_enc', 'source_enc', 'total_pages_visited', 'new_user', 'age', 'smoothed_prob', 'profile_support']]

model_prof_base = XGBClassifier(n_estimators=100, learning_rate=0.03, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=SEED, n_jobs=1, eval_metric='logloss', use_label_encoder=False)

# 4. BOUCLE DE VERIFICATION (CV)
# ------------------------------
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

results = {
    'Sénat_Correct': 0,
    'Prof_Correct': 0,
    'Both_Correct': 0,
    'Both_Wrong': 0,
    'S_Right_P_Wrong': 0, # Sénat a raison, Prof a tort
    'P_Right_S_Wrong': 0  # Prof a raison, Sénat a tort
}

print(f"Lancement Cross-Validation ({N_FOLDS} folds)...")
df_prof = create_profiles(df_train)

# Stockage pour analyse
disagreements_log = []

for fold, (train_idx, val_idx) in enumerate(skf.split(df_train, y)):
    X_tr, X_val = df_train.iloc[train_idx].copy(), df_train.iloc[val_idx].copy()
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # --- MODEL A (SYNDICATE FINAL CUT - PROXY) ---
    X_tr_sen = features_senate(X_tr)
    X_val_sen = features_senate(X_val)
    
    # 1. Base Vote
    pipe_senate.fit(X_tr_sen, y_tr)
    probs_sen = pipe_senate.predict_proba(X_val_sen)[:, 1]
    preds_sen = (probs_sen >= 0.45).astype(int) # Seuil Syndicate agressif
    
    # 2. Amendments (Rules)
    # American Hustle
    mask_us = ((X_val['country'] == 'US') & (X_val['age'] >= 20) & (X_val['age'] <= 30) & (X_val['total_pages_visited'] >= 12))
    preds_sen[mask_us] = 1
    
    # Mariage Freres (Approx proba > 0.33 instead of vote count)
    mask_mf = ((X_val['new_user']==0) & (X_val['total_pages_visited']>=12) & (X_val['total_pages_visited']<=16) & (probs_sen > 0.33))
    preds_sen[mask_mf] = 1
    
    # Erasmus
    mask_er = ((X_val['new_user']==1) & (X_val['total_pages_visited']>=8) & (X_val['total_pages_visited']<=16) & (X_val['country'].isin(['Germany','UK'])) & (X_val['age']<25))
    preds_sen[mask_er] = 1
    
    # 3. Forensic Audit (Cleaning)
    # Train Logistic Formula on same fold
    from sklearn.linear_model import LogisticRegression
    clf_formula = LogisticRegression(solver='lbfgs', max_iter=1000, C=1e9, random_state=SEED, n_jobs=1)
    pipe_formula = Pipeline([
        ('prep', preprocessor),
        ('model', clf_formula)
    ])
    pipe_formula.fit(X_tr_sen, y_tr)
    probs_formula = pipe_formula.predict_proba(X_val_sen)[:, 1]
    
    # Remove False Positives < 10%
    hallucinations = (preds_sen == 1) & (probs_formula < 0.10)
    preds_sen[hallucinations] = 0
    
    # --- MODEL B (PROF) ---
    # 1. OOF encoding for X_tr features
    # (Simplified: using fold statistics directly is fine for model training input here as model is trained on X_tr)
    X_tr_prof = df_prof.iloc[train_idx].copy()
    X_val_prof = df_prof.iloc[val_idx].copy()
    
    # Calc stats on X_tr
    stats = X_tr_prof.groupby('profile_id')['converted'].agg(['count', 'mean'])
    stats.columns = ['n', 'p']
    gm = y_tr.mean()
    stats['sv'] = (stats['n']*stats['p'] + 200*gm)/(stats['n']+200)
    
    # Map to X_tr for training (Internal OOF approximation or just fit on self for speed in this check)
    # Correct way: use internal OOF or just accept slight leak for training the model B.
    # To be fair to logic, lets map using stats (slight overfit but OK for checking logic power)
    X_tr_prof['smoothed_prob'] = X_tr_prof['profile_id'].map(stats['sv']).fillna(gm)
    X_tr_prof['profile_support'] = X_tr_prof['profile_id'].map(stats['n']).fillna(0)
    
    # Map to X_val
    X_val_prof['smoothed_prob'] = X_val_prof['profile_id'].map(stats['sv']).fillna(gm)
    X_val_prof['profile_support'] = X_val_prof['profile_id'].map(stats['n']).fillna(0)
    
    # Train B
    X_tr_feat = feats_prof_prep(X_tr_prof)
    X_val_feat = feats_prof_prep(X_val_prof)
    model_prof_base.fit(X_tr_feat, y_tr)
    probs_prof = model_prof_base.predict_proba(X_val_feat)[:, 1]
    
    # Smart Betting Thresholds (Simplified: Train on X_tr predictions)
    # Pour le script de verif, on prend les clusters principaux et on optimise vite fait
    preds_prof = np.zeros(len(X_val))
    # Threshold global approx 0.38
    # On va faire simple: Optimiser seuil global sur train
    probs_tr_prof = model_prof_base.predict_proba(X_tr_feat)[:, 1]
    best_th = 0.5
    best_f1 = 0
    for th in np.linspace(0.2, 0.6, 20):
        if f1_score(y_tr, (probs_tr_prof >= th).astype(int)) > best_f1:
            best_f1 = f1_score(y_tr, (probs_tr_prof >= th).astype(int))
            best_th = th
            
    # Apply global tuned threshold (approximation of smart betting for speed)
    preds_prof = (probs_prof >= best_th).astype(int)
    
    # --- COMPARE ---
    # Identify disagreements
    disagreements = (preds_sen != preds_prof)
    
    y_val_arr = y_val.values
    
    # Stats
    for i in np.where(disagreements)[0]:
        true_val = y_val_arr[i]
        p_s = preds_sen[i]
        p_p = preds_prof[i]
        
        if p_s == true_val:
            results['S_Right_P_Wrong'] += 1
        else:
            results['P_Right_S_Wrong'] += 1
            
    print(f"Fold {fold+1} : {disagreements.sum()} désaccords.")

print("\n" + "="*80)
print("⚖️ VERDICT DU JUGE (GROUND TRUTH)")
print("-" * 60)
total_disagreements = results['S_Right_P_Wrong'] + results['P_Right_S_Wrong']
print(f"Total Désaccords analysés : {total_disagreements}")
print(f"Le Sénat a raison : {results['S_Right_P_Wrong']} fois ({results['S_Right_P_Wrong']/total_disagreements:.1%})")
print(f"Prof a raison     : {results['P_Right_S_Wrong']} fois ({results['P_Right_S_Wrong']/total_disagreements:.1%})")

print("\nConclusion :")
if results['S_Right_P_Wrong'] > results['P_Right_S_Wrong']:
    print("👉 Le Sénat est plus fiable. Les ajouts de la méthode Prof sont souvent des Hallucinations.")
else:
    print("👉 La Méthode Prof est plus fiable. Elle corrige les erreurs du Sénat.")
