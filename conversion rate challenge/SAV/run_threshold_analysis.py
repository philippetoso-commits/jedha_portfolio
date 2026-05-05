
import pandas as pd
import numpy as np
import warnings
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')

print("📥 Loading Data...")
train_df = pd.read_csv('conversion rate challenge/conversion_data_train.csv')
test_df = pd.read_csv('conversion rate challenge/conversion_data_test.csv')
v9_labels = pd.read_csv('conversion rate challenge/submission_LE_SENAT_V9.csv')
test_df['converted'] = v9_labels['converted']
full_df = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)

# Feature Engineering
def feature_engineering(df):
    df = df.copy()
    df['pages_per_age'] = df['total_pages_visited'] / (df['age'] + 0.1)
    df['interaction_age_pages'] = df['age'] * df['total_pages_visited']
    df['is_active'] = (df['total_pages_visited'] > 2).astype(int)
    return df

X_full = feature_engineering(full_df.drop('converted', axis=1))
y_full = full_df['converted']
X_test = feature_engineering(test_df.drop('converted', axis=1))

# Encoding
cat_cols = ['country', 'source']
cat_indices = []
for col in cat_cols:
    le = LabelEncoder()
    le.fit(X_full[col])
    X_full[col] = le.transform(X_full[col])
    X_test[col] = le.transform(X_test[col])
    cat_indices.append(X_full.columns.get_loc(col))

print("🔥 Retraining Monster to get Probabilities...")
model = HistGradientBoostingClassifier(
    loss="log_loss", learning_rate=0.03, max_iter=500, max_depth=8,
    l2_regularization=0.1, categorical_features=cat_indices, random_state=42
)
model.fit(X_full, y_full)
final_proba = model.predict_proba(X_test)[:, 1]

print("✅ Probabilities Computed.")
sniper_pred = pd.read_csv('conversion rate challenge/submission_FN_SNIPER.csv')['converted']
senate_yes = v9_labels['converted']
monster_yes = (final_proba >= 0.5).astype(int)

print("\n=== SENSITIVITY ANALYSIS (Target: Senate V9) ===")
print(f"{'Threshold':<10} | {'Count':<6} | {'F1 (vs Senate)':<15} | {'Precision':<10} | {'Recall':<10}")
print("-" * 65)

thresholds = np.linspace(0.05, 0.50, 10)
for t in thresholds:
    safety_check = (final_proba >= t).astype(int)
    syndicate_pred = (monster_yes | sniper_pred | (senate_yes & safety_check)).astype(int)
    
    count = syndicate_pred.sum()
    f1 = f1_score(senate_yes, syndicate_pred)
    prec = precision_score(senate_yes, syndicate_pred)
    rec = recall_score(senate_yes, syndicate_pred)
    
    print(f"{t:.2f}       | {count:<6} | {f1:.4f}          | {prec:.4f}     | {rec:.4f}")

print("-" * 65)
