import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder

# 1. Load & Preprocess (Same as FN_SNIPER)
print("📥 Loading Data...")
df = pd.read_csv('conversion rate challenge/conversion_data_train.csv')

def feature_engineering(df):
    df = df.copy()
    df['pages_per_age'] = df['total_pages_visited'] / (df['age'] + 0.1)
    df['interaction_age_pages'] = df['age'] * df['total_pages_visited']
    df['is_active'] = (df['total_pages_visited'] > 2).astype(int)
    return df

X = feature_engineering(df.drop('converted', axis=1))
y = df['converted']

# Encoding
cat_cols = ['country', 'source']
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

cat_idx = [X.columns.get_loc(c) for c in cat_cols]

# 2. Setup Model
model = HistGradientBoostingClassifier(
    loss="log_loss",
    learning_rate=0.05,
    max_iter=400,
    max_depth=8,
    l2_regularization=0.1,
    categorical_features=cat_idx,
    random_state=42
)

# 3. Compute Learning Curve
print("🚀 Computing Learning Curve (F1 Score)...")
train_sizes, train_scores, test_scores = learning_curve(
    model, X, y,
    cv=3,  # 3-Fold for speed
    scoring='f1',
    n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 5), # 10%, 32%, 55%, 77%, 100%
    random_state=42
)

# 4. Aggregate
train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)

print("\n📊 RESULTS (F1 Score):")
print(f"{'Size':<10} {'Train F1':<10} {'CV F1':<10} {'Gap':<10}")
print("-" * 40)
for sz, tr, te in zip(train_sizes, train_mean, test_mean):
    print(f"{sz:<10d} {tr:.4f}     {te:.4f}     {tr-te:.4f}")

# 5. Interpretation
gap = train_mean[-1] - test_mean[-1]
slope = test_mean[-1] - test_mean[-2]

print("\n💡 DIAGNOSIS:")
if gap > 0.05:
    print("⚠️  OVERFITTING: Large gap between Train and CV.")
    print("   -> Solution: More regularization (max_depth, l2) or more data.")
elif slope > 0.001:
    print("📈  LEARNING: CV score is still rising significanly.")
    print("   -> Solution: We need MORE DATA. The model is not full yet.")
else:
    print("✅  SATURATED: CV score has plateaued.")
    print("   -> Solution: Adding data won't help. We need better FEATURES.")
