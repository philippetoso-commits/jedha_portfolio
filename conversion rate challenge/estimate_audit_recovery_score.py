import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import f1_score
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings('ignore')
SEED = 42
N_FOLDS = 10

print("="*80)
print("🔎 ESTIMATION F1 : AUDIT PIPELINE + FORENSIC RECOVERY")
print("="*80)

# 1. Data Prep
df_train = pd.read_csv('conversion_data_train.csv')

# Encode
for col in ['country', 'source']:
    le = LabelEncoder()
    df_train[col] = le.fit_transform(df_train[col])

# Bins for Audit
df_train['age_bin'] = pd.cut(df_train['age'], bins=[0, 18, 25, 30, 35, 40, 45, 50, 60, 100], labels=False).fillna(-1).astype(int)
df_train['pages_bin'] = pd.cut(df_train['total_pages_visited'], bins=[0, 5, 10, 15, 20, 30], labels=False).fillna(-1).astype(int)

features_model = ['country', 'age', 'new_user', 'source', 'total_pages_visited']
features_audit = ['country', 'source', 'new_user', 'age_bin', 'pages_bin']
X = df_train[features_model + ['age_bin', 'pages_bin']]
y = df_train['converted']

# 2. Define Audit Pipeline (Copy of Logic)
class AuditDecisionPipeline(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator=None, audit_segments=None, correction_dampening=0.5, audit_cv=5):
        self.base_estimator = base_estimator if base_estimator else xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
        self.audit_segments = audit_segments if audit_segments else ['country', 'source', 'new_user', 'age_bin']
        self.correction_dampening = correction_dampening
        self.audit_cv = audit_cv
        self.corrections_ = {}
        self.classes_ = None
        
    def fit(self, X, y):
        self.base_estimator.fit(X, y)
        if hasattr(self.base_estimator, 'classes_'):
             self.classes_ = self.base_estimator.classes_
        else:
             self.classes_ = np.unique(y)
        
        # Internal OOF for Audit
        if self.audit_cv > 1:
            try:
                oof_preds = cross_val_predict(clone(self.base_estimator), X, y, cv=self.audit_cv, method='predict_proba')[:, 1]
            except:
                 oof_preds = cross_val_predict(clone(self.base_estimator), X, y, cv=self.audit_cv)
        else:
            oof_preds = self.base_estimator.predict_proba(X)[:, 1]
            
        residuals = y - oof_preds
        self.corrections_ = {}
        for seg_col in self.audit_segments:
            if seg_col not in X.columns: continue
            stats = pd.DataFrame({'feature': X[seg_col], 'residual': residuals}).groupby('feature')['residual'].agg(['mean', 'count', 'std'])
            stats['se'] = stats['std'] / np.sqrt(stats['count'])
            significant = stats[np.abs(stats['mean']) > 1.96 * stats['se']]
            if not significant.empty:
                self.corrections_[seg_col] = (significant['mean'] * self.correction_dampening).to_dict()
        return self
    
    def predict_proba(self, X):
        base_probs = self.base_estimator.predict_proba(X)[:, 1]
        final_probs = base_probs.copy()
        for seg_col, corr_map in self.corrections_.items():
            if seg_col in X.columns:
                adjustments = X[seg_col].map(corr_map).fillna(0).values
                final_probs += adjustments
        final_probs = np.clip(final_probs, 0, 1)
        return np.vstack([1-final_probs, final_probs]).T

# 3. CV Loop
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

scores_audit = []
scores_recovery = []

print(f"Running {N_FOLDS}-Fold CV...")

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # A. Train Audit Pipeline (Conservative)
    pipeline = AuditDecisionPipeline(
        base_estimator=xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42, eval_metric='logloss'),
        audit_segments=features_audit,
        correction_dampening=0.5,
        audit_cv=5
    )
    pipeline.fit(X_tr, y_tr)
    
    # Predict Audit
    # We optimize threshold on Train to be fair? 
    # Or just use the global known good threshold ~0.37?
    # Let's use a fixed threshold to isolate the impact of recovery.
    TH_AUDIT = 0.37
    prob_audit = pipeline.predict_proba(X_val)[:, 1]
    pred_audit = (prob_audit >= TH_AUDIT).astype(int)
    
    # B. Train Aggressive Proxy (Simulating Mariage Frères)
    # Standard XGB with lower threshold
    proxy = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42, eval_metric='logloss')
    proxy.fit(X_tr, y_tr)
    prob_proxy = proxy.predict_proba(X_val)[:, 1]
    # Mariage is aggressive but high quality. Let's align with Audit.
    # Audit is at 0.37. Let's make Proxy slightly more aggressive but not loose.
    TH_PROXY = 0.35 
    pred_proxy = (prob_proxy >= TH_PROXY).astype(int)
    
    # C. Apply Recovery Logic
    # Rule: If Proxy=1 AND Audit=0 AND (Page>=6 OR Age>=25) -> Recover
    
    # Identify Candidates
    candidates_mask = (pred_proxy == 1) & (pred_audit == 0)
    
    # Check Safety Features
    # Note: feature indices might be messed up if we don't use dataframe. X_val is dataframe.
    safe_mask = (X_val['total_pages_visited'] >= 6) | (X_val['age'] >= 25)
    
    # Recover
    pred_recovered = pred_audit.copy()
    # Indices where both candidate and safe are True
    recovery_indices = candidates_mask & safe_mask
    pred_recovered[recovery_indices] = 1
    
    # Score
    f1_audit = f1_score(y_val, pred_audit)
    f1_recovery = f1_score(y_val, pred_recovered)
    
    scores_audit.append(f1_audit)
    scores_recovery.append(f1_recovery)
    
    print(f"  Fold {fold+1}: Audit={f1_audit:.5f} | Recovery={f1_recovery:.5f} | Recovered={recovery_indices.sum()}")

avg_audit = np.mean(scores_audit)
avg_recovery = np.mean(scores_recovery)

print("-" * 60)
print(f"🏆 Résultats Moyens ({N_FOLDS} folds)")
print(f"  Audit Pipeline F1 : {avg_audit:.5f}")
print(f"  + Forensic Recov  : {avg_recovery:.5f}")
print(f"  GAIN              : {avg_recovery - avg_audit:+.5f}")
