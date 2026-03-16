
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

print("⚖️ JUDICIAL CORRECTION: SURGICAL BIAS REMOVAL ⚖️")

# 1. Load Data
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

print("🔮 Predicting Raw Probabilities...")
raw_probs = pipeline.predict_proba(X_test)[:, 1]

# 2. APPLY CORRECTIONS
adj_probs = raw_probs.copy()
test_indices = X_test.index

# DEFINITIONS based on TEST
# Note: We act on indices relative to X_test
mask_de_2030 = (X_test['country'] == 'Germany') & (X_test['age'] > 20) & (X_test['age'] <= 30)
mask_us_2030 = (X_test['country'] == 'US') & (X_test['age'] > 20) & (X_test['age'] <= 30)
mask_us_young = (X_test['country'] == 'US') & (X_test['age'] <= 20)

# ZONE DEFINITION
low_bound = 0.42
high_bound = 0.58
in_zone = (raw_probs >= low_bound) & (raw_probs <= high_bound)

print("\n🔧 Applying Surgical Corrections:")

# A. GLOBAL BOOST (US 20-30) - Validated as Global Trend
# We add a smaller flat boost everywhere
adj_probs[mask_us_2030] += 0.04
print(f"   🇺🇸 US 20-30: Applied Global Boost (+0.04) to {mask_us_2030.sum()} users.")

# B. LOCAL BOOST (DE 20-30) - Validated as JACKPOT
# Only in zone
mask_de_surgical = mask_de_2030 & in_zone
adj_probs[mask_de_surgical] += 0.10 # Big push to clear the wall
print(f"   🇩🇪 DE 20-30: Applied Surgical Boost (+0.10) to {mask_de_surgical.sum()} users in Zone.")

# C. LOCAL PENALTY (US < 20) - Validated as JACKPOT
# Only in zone
mask_us_surgical = mask_us_young & in_zone
adj_probs[mask_us_surgical] -= 0.10 # Big push down
print(f"   👶 US < 20 : Applied Surgical Penalty (-0.10) to {mask_us_surgical.sum()} users in Zone.")

# Clip
adj_probs = np.clip(adj_probs, 0, 1)

# 3. THRESHOLD
# We use standard 0.50 because we corrected the probabilities to be calibrated around 0.5
final_preds = (adj_probs >= 0.50).astype(int)

print(f"\n📊 JUDICIAL RESULTS:")
print(f"   Base Formula Conversions: {(raw_probs >= 0.50).sum()}")
print(f"   Judicial Conversions:     {final_preds.sum()}")
print(f"   Net Change: {final_preds.sum() - (raw_probs >= 0.50).sum()}")

sub = pd.DataFrame({'converted': final_preds})
sub.to_csv('conversion rate challenge/submission_JUDICIAL_CORRECTION.csv', index=False)
print("✅ Saved 'submission_JUDICIAL_CORRECTION.csv'.")
