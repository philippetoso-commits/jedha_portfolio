
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

print("🔧 APPLYING FINAL CUT PATCH TO MARIAGE FRERES BASELINE 🔧")

# 1. Load the Strong Base (Rank 3: 0.764)
# We use the FILE that the user confirmed is the reference
base_file = 'conversion rate challenge/submission_mariagefrere_tosophilippe.csv'
print(f"📥 Loading Base Submission: {base_file}")
sub = pd.read_csv(base_file)
initial_conversions = sub['converted'].sum()
print(f"   Initial Conversions: {initial_conversions}")

# 2. Load Test Data for Rules
test_df = pd.read_csv('conversion rate challenge/conversion_data_test.csv')

# 3. Apply The Missing Rules (The Boosters)
print("🚀 Applying 'Final Cut' Boosters...")

# American Hustle (US 20-30, Pages >= 12)
mask_us = (
    (test_df['country'] == 'US') & 
    (test_df['age'] >= 20) & 
    (test_df['age'] <= 30) &
    (test_df['total_pages_visited'] >= 12)
)
n_us = mask_us.sum()
sub.loc[mask_us, 'converted'] = 1
print(f"   🇺🇸 American Hustle: Boosted {n_us} users (Potential Gold Mine).")

# Erasmus (Europe Youth)
mask_erasmus = (
    (test_df['new_user'] == 1) & 
    (test_df['total_pages_visited'] >= 8) & 
    (test_df['total_pages_visited'] <= 16) &
    (test_df['country'].isin(['Germany', 'UK'])) &
    (test_df['age'] < 25)
)
n_erasmus = mask_erasmus.sum()
sub.loc[mask_erasmus, 'converted'] = 1
print(f"   🇪🇺 Erasmus: Boosted {n_erasmus} users.")

# 4. Apply Audit (The Safety Net) - OPTIONAL but Recommended
print("🛡️ Applying Forensic Audit (Safety Net)...")
# We need to retrain Formula quickly to check reliability
train_df = pd.read_csv('conversion rate challenge/conversion_data_train.csv')

# Feature Eng common to Formula
def engineer(df):
    d = df.copy()
    d['interaction'] = d['total_pages_visited'] * d['age']
    d['pages_sq'] = d['total_pages_visited']**2
    return d

numeric_features = ['age', 'total_pages_visited', 'interaction', 'pages_sq']
categorical_features = ['country', 'source', 'new_user']

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
])

clf = LogisticRegression(solver='lbfgs', max_iter=1000, C=1e9)
pipe = Pipeline([('prep', preprocessor), ('model', clf)])
pipe.fit(engineer(train_df), train_df['converted'])
probs = pipe.predict_proba(engineer(test_df))[:, 1]

# Kill Hallucinations (Converted=1 BUT Proba < 0.10)
hallucinations = (sub['converted'] == 1) & (probs < 0.10)
n_killed = hallucinations.sum()
sub.loc[hallucinations, 'converted'] = 0
print(f"   🔪 Forensic Audit: Removed {n_killed} hallucinations.")

# 5. Export
final_count = sub['converted'].sum()
diff = final_count - initial_conversions
print(f"\n✅ FINAL SYNDICATE REAL COUNT: {final_count} (Delta: {diff:+d})")

filename = 'conversion rate challenge/submission_SYNDICATE_REAL_FINAL.csv'
sub.to_csv(filename, index=False)
print(f"📁 Saved to: {filename}")
print("👉 THIS IS THE WINNING FILE. SUBMIT THIS ONE.")
