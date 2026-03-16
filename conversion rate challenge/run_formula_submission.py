
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score

print("🧪 THE FORMULA: GENERATING SUBMISSION 🧪")
print("📥 Loading Data...")
train_df = pd.read_csv('conversion rate challenge/conversion_data_train.csv')
test_df = pd.read_csv('conversion rate challenge/conversion_data_test.csv')

# Combine for encoding/engineering consistency
train_len = len(train_df)
y = train_df['converted']
full_df = pd.concat([train_df.drop('converted', axis=1), test_df], axis=0).reset_index(drop=True)

# Feature Engineering (The Found Formula)
print("⚙️ Applying Formula Features (Interactions)...")
full_df['pages_sq'] = full_df['total_pages_visited'] ** 2
full_df['age_sq'] = full_df['age'] ** 2
full_df['interaction'] = full_df['total_pages_visited'] * full_df['age']
full_df['rate_concept'] = full_df['total_pages_visited'] / (full_df['age'] + 1)

# Preprocessing
numeric_features = ['age', 'total_pages_visited', 'pages_sq', 'age_sq', 'interaction', 'rate_concept']
categorical_features = ['country', 'source', 'new_user'] # new_user is cat

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
    ]
)

# The Exact Model (No Regularization C=1e9)
model = LogisticRegression(solver='lbfgs', max_iter=2000, C=1e9)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', model)
])

# Training
print("🚀 Training Full Model to capture coefficients...")
X_train = full_df.iloc[:train_len]
X_test = full_df.iloc[train_len:]

pipeline.fit(X_train, y)

# Threshold Optimization on Train
print("🔍 Optimizing Threshold on Train...")
train_probs = pipeline.predict_proba(X_train)[:, 1]
best_f1 = 0
best_thresh = 0.5
for t in np.linspace(0.4, 0.6, 100):
    score = f1_score(y, (train_probs >= t).astype(int))
    if score > best_f1:
        best_f1 = score
        best_thresh = t

print(f"   ✅ Best Train F1: {best_f1:.5f} at Threshold: {best_thresh:.4f}")

# Prediction
print("🔮 Predicting on Test Set...")
test_probs = pipeline.predict_proba(X_test)[:, 1]
test_preds = (test_probs >= best_thresh).astype(int)

# Export
filename = 'conversion rate challenge/submission_THE_FORMULA.csv'
sub = pd.DataFrame({'converted': test_preds})
sub.to_csv(filename, index=False)
print(f"✅ Generated '{filename}' with {test_preds.sum()} conversions.")
