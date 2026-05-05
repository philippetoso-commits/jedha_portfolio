
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

print("🔫 FORENSIC CORRECTION: APPLYING MICRO-WEIGHTS 🔫")

# 1. Load Data & Train Formula Base
# We use the Formula Model as the "Unbiased Observer" to apply corrections on top of.
print("📥 Loading Data...")
train_df = pd.read_csv('conversion rate challenge/conversion_data_train.csv')
test_df = pd.read_csv('conversion rate challenge/conversion_data_test.csv')

train_len = len(train_df)
y = train_df['converted']
full_df = pd.concat([train_df.drop('converted', axis=1), test_df], axis=0).reset_index(drop=True)

# Feature Engineering
full_df['pages_sq'] = full_df['total_pages_visited'] ** 2
full_df['age_sq'] = full_df['age'] ** 2
full_df['interaction'] = full_df['total_pages_visited'] * full_df['age']
full_df['rate_concept'] = full_df['total_pages_visited'] / (full_df['age'] + 1)

numeric_features = ['age', 'total_pages_visited', 'pages_sq', 'age_sq', 'interaction', 'rate_concept']
categorical_features = ['country', 'source', 'new_user']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
    ]
)

model = LogisticRegression(solver='lbfgs', max_iter=2000, C=1e9)
pipeline = Pipeline([('preprocessor', preprocessor), ('model', model)])

print("⚙️ Training Base Formula Model...")
X_train = full_df.iloc[:train_len]
X_test = full_df.iloc[train_len:]
pipeline.fit(X_train, y)

# 2. Get Raw Probabilities on Test
print("🔮 Predicting Raw Probabilities...")
raw_probs = pipeline.predict_proba(X_test)[:, 1]

# 3. Apply Forensic Corrections
# We modify the PROBABILITY directly based on the Residual Audit.
# Germany (20-30): +0.093
# UK (40-60): +0.068
# US (20-30): +0.043
# US (<20): -0.085
# UK (<20): -0.093

print("🔧 Applying Micro-Weights...")
adj_probs = raw_probs.copy()
test_indices = X_test.index

# Helper to find indices in X_test (pandas indexing)
# X_test is a DataFrame slice
def get_mask(condition):
    return condition[test_indices]

# BONUSES
mask_de_2030 = (X_test['country'] == 'Germany') & (X_test['age'] > 20) & (X_test['age'] <= 30)
adj_probs[mask_de_2030] += 0.093

mask_uk_old = (X_test['country'] == 'UK') & (X_test['age'] > 40) & (X_test['age'] <= 60)
adj_probs[mask_uk_old] += 0.068

mask_us_2030 = (X_test['country'] == 'US') & (X_test['age'] > 20) & (X_test['age'] <= 30)
adj_probs[mask_us_2030] += 0.043

# PENALTIES
mask_us_young = (X_test['country'] == 'US') & (X_test['age'] <= 20)
adj_probs[mask_us_young] -= 0.085

mask_uk_young = (X_test['country'] == 'UK') & (X_test['age'] <= 20)
adj_probs[mask_uk_young] -= 0.093

# Clip to [0, 1]
adj_probs = np.clip(adj_probs, 0, 1)

# 4. Thresholding
# With calibrated probabilities, 0.5 should be the theoretical optimal for Accuracy.
# For F1, we usually tune. Let's stick to the "Formula Optimal" found earlier (0.41) or safer 0.45?
# Given we added bonuses, 0.5 is safer to avoid over-prediction.
threshold = 0.50
final_preds = (adj_probs >= threshold).astype(int)

print(f"\n📊 CORRECTION RESULTS:")
print(f"   Unadjusted Conversions: {(raw_probs >= threshold).sum()}")
print(f"   Forensic Conversions:   {final_preds.sum()}")
print(f"   Difference: {final_preds.sum() - (raw_probs >= threshold).sum()}")

# Save
sub = pd.DataFrame({'converted': final_preds})
sub.to_csv('conversion rate challenge/submission_FORENSIC.csv', index=False)
print("✅ Saved 'submission_FORENSIC.csv'.")
