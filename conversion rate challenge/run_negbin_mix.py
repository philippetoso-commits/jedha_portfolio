
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
import warnings

warnings.filterwarnings('ignore')

print("⏳ Chargement des données...")
train_data = pd.read_csv('conversion rate challenge/conversion_data_train.csv')
test_data = pd.read_csv('conversion rate challenge/conversion_data_test.csv')

# Smart Feature Engineering (Version ULTIMATE)
def smart_feature_engineering(df):
    df_eng = df.copy()
    df_eng['is_active'] = (df_eng['total_pages_visited'] > 2).astype(int)
    df_eng['interaction_age_pages'] = df_eng['age'] * df_eng['total_pages_visited']
    df_eng['pages_per_age'] = df_eng['total_pages_visited'] / (df_eng['age'] + 0.1)
    return df_eng

X = smart_feature_engineering(train_data.drop('converted', axis=1))
y = train_data['converted']
X_test = smart_feature_engineering(test_data)

# Encodage
categorical_cols = ['country', 'source']
for col in categorical_cols:
    le = LabelEncoder()
    combined = pd.concat([X[col], X_test[col]])
    le.fit(combined)
    X[col] = le.transform(X[col])
    X_test[col] = le.transform(X_test[col])

print(f"✅ Dataset prêt : {len(X)} lignes")

# ====================================================================
# MODÈLE 2 : NEGATIVE BINOMIAL + LOGLOSS MIX (Hybride)
# ====================================================================
print("🧪 Entraînement NEGBIN + LOGLOSS MIX...")

oof_preds_mix = np.zeros(len(X))
test_preds_mix = np.zeros(len(X_test))

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train_fold, y_train_fold = X.iloc[train_idx], y.iloc[train_idx]
    X_val_fold, y_val_fold = X.iloc[val_idx], y.iloc[val_idx]
    
    # MODÈLE A : Negative Binomial
    params_nb = {
        'objective': 'count:poisson',
        'max_delta_step': 0.7,
        'learning_rate': 0.05,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'seed': 42 + fold
    }
    
    dtrain = xgb.DMatrix(X_train_fold, label=y_train_fold)
    dval = xgb.DMatrix(X_val_fold)
    dtest = xgb.DMatrix(X_test)
    
    model_nb = xgb.train(params_nb, dtrain, num_boost_round=500, verbose_eval=False)
    val_pred_nb = model_nb.predict(dval)
    test_pred_nb = model_nb.predict(dtest)
    
    # MODÈLE B : LogLoss Classifier
    cat_indices = [X.columns.get_loc(col) for col in categorical_cols]
    model_clf = HistGradientBoostingClassifier(
        loss='log_loss', learning_rate=0.05, max_iter=500, max_depth=8,
        l2_regularization=0.1, categorical_features=cat_indices, random_state=42+fold
    )
    model_clf.fit(X_train_fold, y_train_fold)
    val_pred_clf = model_clf.predict_proba(X_val_fold)[:, 1]
    test_pred_clf = model_clf.predict_proba(X_test)[:, 1]
    
    # MIX
    val_mix = (val_pred_nb + val_pred_clf) / 2
    test_mix = (test_pred_nb + test_pred_clf) / 2
    
    oof_preds_mix[val_idx] = val_mix
    test_preds_mix += test_mix
    
    print(f"  > Fold {fold+1}/10 done.")

test_preds_mix /= 10

# Optimisation seuil
best_f1_mix = 0
best_thresh_mix = 0
for t in np.linspace(oof_preds_mix.min(), oof_preds_mix.max(), 1000):
    preds = (oof_preds_mix >= t).astype(int)
    score = f1_score(y, preds)
    if score > best_f1_mix:
        best_f1_mix = score
        best_thresh_mix = t

print(f"✅ Best CV F1: {best_f1_mix:.5f} at Threshold: {best_thresh_mix:.6f}")

# Export
final_test = (test_preds_mix >= best_thresh_mix).astype(int)
filename = 'conversion rate challenge/submission_NEGBIN_MIX.csv'
submission = pd.DataFrame({'converted': final_test})
submission.to_csv(filename, index=False)
print(f"✅ Fichier '{filename}' généré avec {final_test.sum()} conversions.")
