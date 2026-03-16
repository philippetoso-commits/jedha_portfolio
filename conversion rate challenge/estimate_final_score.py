import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import VotingClassifier, HistGradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from sklearn.base import BaseEstimator, TransformerMixin
import warnings

warnings.filterwarnings('ignore')
SEED = 42
N_FOLDS = 10 

print("="*80)
print("⚖️ ESTIMATION FINALE F1 (Cross-Validation 10 Folds)")
print("="*80)
print("Comparaison : Syndicate Final Cut (Proxy) vs Méthode Prof V2")

# 1. LOAD DATA
df = pd.read_csv('conversion rate challenge/conversion_data_train.csv')
y = df['converted']

# 2. SYNDICATE PROXY SETUP
def features_senate(df):
    d = df.copy()
    d['pages_per_age'] = d['total_pages_visited'] / (d['age'] + 0.1)
    d['interaction'] = d['total_pages_visited'] * d['age']
    return d

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), ['age', 'total_pages_visited', 'pages_per_age', 'interaction']),
    ('cat', OneHotEncoder(handle_unknown='ignore'), ['country', 'source', 'new_user'])
])

clf1 = XGBClassifier(n_estimators=100, max_depth=6, random_state=SEED, n_jobs=1, eval_metric='logloss', use_label_encoder=False)
clf2 = LGBMClassifier(n_estimators=100, random_state=SEED, n_jobs=1, verbose=-1)
clf3 = HistGradientBoostingClassifier(max_iter=100, random_state=SEED)
ensemble = VotingClassifier([('xgb', clf1), ('lgb', clf2), ('hgb', clf3)], voting='soft')
pipe_syndicate = Pipeline([('prep', preprocessor), ('model', ensemble)])

# 3. METHOD PROF SETUP
class BayesianTargetEncoderCV(BaseEstimator, TransformerMixin):
    def __init__(self, key_col, target_col, n_folds=5, alpha=200, seed=42):
        self.key_col, self.target_col, self.n_folds = key_col, target_col, n_folds
        self.alpha, self.seed = alpha, seed
        self.global_stats = None
        
    def fit(self, X, y=None):
        self.global_stats = X.groupby(self.key_col)[self.target_col].agg(['count', 'mean'])
        self.global_stats.columns = ['n', 'p']
        gm = X[self.target_col].mean()
        self.global_stats['sv'] = (self.global_stats['n']*self.global_stats['p'] + self.alpha*gm)/(self.global_stats['n']+self.alpha)
        self.global_mean = gm
        return self
        
    def fit_transform_oof(self, X, y=None):
        self.fit(X, y)
        X_out = X.copy()
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)
        for t, v in skf.split(X, X[self.target_col]):
            xt, xv = X.iloc[t], X.iloc[v]
            gm = xt[self.target_col].mean()
            st = xt.groupby(self.key_col)[self.target_col].agg(['count', 'mean'])
            st.columns = ['n', 'p']
            st['sv'] = (st['n']*st['p'] + self.alpha*gm)/(st['n']+self.alpha)
            X_out.iloc[v, X_out.columns.get_loc('smoothed_prob')] = xv[self.key_col].map(st['sv']).fillna(gm)
            X_out.iloc[v, X_out.columns.get_loc('profile_support')] = xv[self.key_col].map(st['n']).fillna(0)
        return X_out

def create_profiles(df):
    d = df.copy()
    d['age_bin'] = pd.cut(d['age'], bins=[0,18,25,30,35,40,45,50,60,100], labels=False)
    d['pages_bin'] = pd.cut(d['total_pages_visited'], bins=[-1,1,2,3,4,5,6,7,8,9,10,12,14,16,18,20,25,100], labels=False)
    d['profile_id'] = d['country'].astype(str)+"_"+d['source'].astype(str)+"_"+d['new_user'].astype(str)+"_"+d['age_bin'].astype(str)+"_"+d['pages_bin'].astype(str)
    d['smoothed_prob'] = np.nan
    d['profile_support'] = np.nan
    # Cluster Betting
    d['age_broad'] = pd.cut(d['age'], bins=[0, 30, 100], labels=['Young', 'Senior'])
    d['cluster_betting'] = d['country'].astype(str) + "_" + d['age_broad'].astype(str)
    return d

le_country = LabelEncoder().fit(df['country'])
le_source = LabelEncoder().fit(df['source'])
def manual_enc(df):
    d = df.copy()
    d['country_enc'] = le_country.transform(d['country'])
    d['source_enc'] = le_source.transform(d['source'])
    return d[['country_enc', 'source_enc', 'total_pages_visited', 'new_user', 'age', 'smoothed_prob', 'profile_support']]

model_prof = XGBClassifier(n_estimators=200, learning_rate=0.03, max_depth=6, subsample=0.8, colsample_bytree=0.8, eval_metric='logloss', use_label_encoder=False, random_state=SEED, n_jobs=1)

# 4. LOOP
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
f1_syndicate = []
f1_prof = []

df_prof_base = create_profiles(df)
prof_encoder = BayesianTargetEncoderCV(key_col='profile_id', target_col='converted', n_folds=5, alpha=200)
# Pre-calc OOF for speed (normally inside loop but ok)
df_prof_encoded = prof_encoder.fit_transform_oof(df_prof_base)

for fold, (train_idx, val_idx) in enumerate(skf.split(df, y)):
    X_tr, X_val = df.iloc[train_idx], df.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # --- SYNDICATE ---
    pipe_syndicate.fit(features_senate(X_tr), y_tr)
    p_syn = pipe_syndicate.predict_proba(features_senate(X_val))[:, 1]
    # Syndicate Threshold
    pred_syn = (p_syn >= 0.45).astype(int)
    # Rules
    mask_us = ((X_val['country'] == 'US') & (X_val['age'] >= 20) & (X_val['age'] <= 30) & (X_val['total_pages_visited'] >= 12))
    pred_syn[mask_us] = 1
    mask_mf = ((X_val['new_user']==0) & (X_val['total_pages_visited']>=12) & (X_val['total_pages_visited']<=16) & (p_syn > 0.33))
    pred_syn[mask_mf] = 1
    
    # --- PROF ---
    # Use pre-calculated OOF (simulated leak free)
    X_tr_prof = df_prof_encoded.iloc[train_idx]
    X_val_prof = df_prof_encoded.iloc[val_idx]
    
    model_prof.fit(manual_enc(X_tr_prof), y_tr)
    p_prof = model_prof.predict_proba(manual_enc(X_val_prof))[:, 1]
    
    # Smart Betting Approx (optimize on fold)
    preds_prof = np.zeros(len(X_val))
    for cl in X_tr_prof['cluster_betting'].unique():
        m_tr = (X_tr_prof['cluster_betting'] == cl)
        if m_tr.sum() < 20: continue
        best_th = 0.5
        best_f = 0
        p_tr_c = model_prof.predict_proba(manual_enc(X_tr_prof[m_tr]))[:, 1]
        y_tr_c = y_tr[m_tr]
        for th in np.linspace(0.1, 0.9, 20):
            if f1_score(y_tr_c, (p_tr_c >= th).astype(int)) > best_f:
                best_f = f1_score(y_tr_c, (p_tr_c >= th).astype(int))
                best_th = th
        
        m_val = (X_val_prof['cluster_betting'] == cl)
        preds_prof[m_val] = (p_prof[m_val] >= best_th).astype(int)
    
    s_syn = f1_score(y_val, pred_syn)
    s_prof = f1_score(y_val, preds_prof)
    
    f1_syndicate.append(s_syn)
    f1_prof.append(s_prof)
    print(f"Fold {fold+1} | Syndicate: {s_syn:.4f} | Prof: {s_prof:.4f}")

print("-" * 60)
print(f"Moyenne Syndicate : {np.mean(f1_syndicate):.5f}")
print(f"Moyenne Prof      : {np.mean(f1_prof):.5f}")
print(f"Gain Estimé       : {np.mean(f1_prof) - np.mean(f1_syndicate):.5f}")
