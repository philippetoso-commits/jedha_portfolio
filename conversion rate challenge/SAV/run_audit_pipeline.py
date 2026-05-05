import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.metrics import f1_score, log_loss, brier_score_loss
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings('ignore')

print("="*80)
print("🧐 AUDIT-DRIVEN ERROR-AWARE PIPELINE")
print("="*80)

# ==================================================================================
# 1. UTILS & DATA PREP
# ==================================================================================
def load_and_prep():
    df_train = pd.read_csv('conversion_data_train.csv')
    df_test = pd.read_csv('conversion_data_test.csv')
    
    # Basic Prep
    for df in [df_train, df_test]:
        df['age_bin'] = pd.cut(df['age'], bins=[0, 18, 25, 30, 35, 40, 45, 50, 60, 100], labels=False).fillna(-1).astype(int)
        # Simplify Pages for segment analysis
        df['pages_bin'] = pd.cut(df['total_pages_visited'], bins=[0, 5, 10, 15, 20, 30], labels=False).fillna(-1).astype(int)
    
    return df_train, df_test

# ==================================================================================
# 2. THE ERROR AUDITOR & CORRECTOR KERNEL
# ==================================================================================
class AuditDecisionPipeline(BaseEstimator, ClassifierMixin):
    """
    A Pipeline that:
    1. Trains a Base Model.
    2. Audits Residuals on the Training Set (via internal CV to avoid overfitting).
    3. Learns Additive Corrections for biased segments.
    4. Applies Base + Correction at Prediction time.
    """
    def __init__(self, base_estimator=None, audit_segments=None, correction_dampening=0.5, audit_cv=5):
        self.base_estimator = base_estimator if base_estimator else xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
        self.audit_segments = audit_segments if audit_segments else ['country', 'source', 'new_user', 'age_bin']
        self.correction_dampening = correction_dampening
        self.audit_cv = audit_cv
        self.corrections_ = {} # Store {segment_col: {value: bias_adjustment}}
        self.global_threshold_ = 0.5
        
    def fit(self, X, y):
        # 1. Train Base Model (for final predictions)
        self.base_estimator.fit(X, y)
        if hasattr(self.base_estimator, 'classes_'):
             self.classes_ = self.base_estimator.classes_
        else:
             self.classes_ = np.unique(y)
        
        # 2. Generate OOF Predictions on X to audit "honest" errors
        # We need OOF preds on X to know where the model fails. 
        # Making corrections on Training Error (non-OOF) would just overfit noise.
        if self.audit_cv > 1:
            try:
                oof_preds = cross_val_predict(clone(self.base_estimator), X, y, cv=self.audit_cv, method='predict_proba')[:, 1]
            except:
                 # Fallback for models without predict_proba in cross_val_predict (rare)
                 oof_preds = cross_val_predict(clone(self.base_estimator), X, y, cv=self.audit_cv) 
        else:
            # Dangerous: Training error
            oof_preds = self.base_estimator.predict_proba(X)[:, 1]
            
        # 3. Calculate Residuals
        # Residual = Truth - Prob
        # Positive Residual = Underestimation (Model < Truth) -> Need to Add Prob
        # Negative Residual = Overestimation (Model > Truth) -> Need to Sub Prob
        residuals = y - oof_preds
        
        # 4. Audit Segments
        self.corrections_ = {}
        
        for seg_col in self.audit_segments:
            # Group by segment value
            segment_analysis = pd.DataFrame({
                'feature': X[seg_col],
                'residual': residuals
            })
            
            # Calculate stats
            stats = segment_analysis.groupby('feature')['residual'].agg(['mean', 'count', 'std'])
            stats['se'] = stats['std'] / np.sqrt(stats['count'])
            
            # Identify Significant Biases (e.g., |Mean| > 2 * SE)
            # Correction = Mean Residual * Dampening
            # We filter for only statistically significant biases to be "Robust"
            significant = stats[np.abs(stats['mean']) > 1.96 * stats['se']]
            
            if not significant.empty:
                # Store corrections map
                # correction = bias * dampening
                self.corrections_[seg_col] = (significant['mean'] * self.correction_dampening).to_dict()
                
        return self
    
    def predict_proba(self, X):
        # 1. Base Prediction
        base_probs = self.base_estimator.predict_proba(X)[:, 1]
        
        # 2. Add Corrections
        final_probs = base_probs.copy()
        
        for seg_col, corr_map in self.corrections_.items():
            if seg_col in X.columns:
                # Map values to corrections, fill missing with 0
                adjustments = X[seg_col].map(corr_map).fillna(0).values
                final_probs += adjustments
        
        # 3. Clip to [0, 1]
        final_probs = np.clip(final_probs, 0, 1)
        
        return np.vstack([1-final_probs, final_probs]).T
        
    def predict(self, X):
        probs = self.predict_proba(X)[:, 1]
        return (probs >= 0.5).astype(int)

# ==================================================================================
# 3. EXECUTION
# ==================================================================================

df_train, df_test = load_and_prep()

# Encode Categoricals (Basic)
cat_cols = ['country', 'source']
for col in cat_cols:
    le = LabelEncoder()
    # Fit on combined to ensure coverage
    full_vals = pd.concat([df_train[col], df_test[col]], axis=0)
    le.fit(full_vals)
    df_train[col] = le.transform(df_train[col])
    df_test[col] = le.transform(df_test[col])

X = df_train.drop(columns=['converted', 'age_bin', 'pages_bin']) # Keep age/pages raw for model? No, let's keep bins for audit
# Actually, let's pass specific cols to model.
features_model = ['country', 'age', 'new_user', 'source', 'total_pages_visited']
features_audit = ['country', 'source', 'new_user', 'age_bin', 'pages_bin']

# Add bins to X for auditing
X = df_train[features_model + ['age_bin', 'pages_bin']] # Include audit features in X
y = df_train['converted']
X_test = df_test[features_model + ['age_bin', 'pages_bin']]

# Define Base Model
xgb_base = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)

# Standard XGB Baseline Check
print("Training Standard Baseline...")
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
baseline_scores = cross_val_score(xgb_base, X[features_model], y, cv=kf, scoring='f1')
print(f"Base XGB F1: {baseline_scores.mean():.5f} (+/- {baseline_scores.std()*2:.5f})")

# Audit Pipeline Check
print("\nTraining Audit-Driven Pipeline...")
# Note: The Audited Pipeline can use the extra bin features for auditing, but the base model should arguably see them or not? 
# The wrapper fits base_estimator on X. If we want base estimator to ignore audit cols, we need a refined class.
# For simplicity here: Base XGB will see all cols passed to fit. XGB handles redundancy well.
pipeline = AuditDecisionPipeline(
    base_estimator=xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42, eval_metric='logloss'),
    audit_segments=features_audit,
    correction_dampening=0.5,
    audit_cv=5
)

pipeline_scores = cross_val_score(pipeline, X, y, cv=kf, scoring='f1')
print(f"Audit Pipeline F1: {pipeline_scores.mean():.5f} (+/- {pipeline_scores.std()*2:.5f})")

# Full Training & Submission
print("\nGenerating Final Model...")
pipeline.fit(X, y)

print("Applied Corrections Summary:")
for seg, corrections in pipeline.corrections_.items():
    if corrections:
        print(f"  Segment '{seg}': {len(corrections)} corrections applied.")
        # Print top 3 corrections
        sorted_corr = sorted(corrections.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        for val, adj in sorted_corr:
            print(f"    - Val {val}: {adj:+.4f}")

# Threshold Optimization for Pipeline
# We need probas
print("\nOptimizing Threshold...")
train_probs = cross_val_predict(pipeline, X, y, cv=5, method='predict_proba')[:, 1]

best_th = 0.5
best_f1 = 0
for th in np.linspace(0.3, 0.7, 100):
    f = f1_score(y, (train_probs >= th).astype(int))
    if f > best_f1:
        best_f1 = f
        best_th = th

print(f"Best Threshold: {best_th:.4f} | F1: {best_f1:.5f}")

# Predict Test
test_probs = pipeline.predict_proba(X_test)[:, 1]
test_preds = (test_probs >= best_th).astype(int)

sub = pd.DataFrame({'converted': test_preds})
sub.to_csv('submission_AUDIT_PIPELINE.csv', index=False)
print(f"Submitted {test_preds.sum()} conversions to submission_AUDIT_PIPELINE.csv")
