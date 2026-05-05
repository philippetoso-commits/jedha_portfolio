
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

print("🧧 THE CHINESE HEIST: RESCUING THE DRAGONS 🧧")

# 1. SETUP
print("📥 Loading Data & Syndicate Final Cut...")
train_df = pd.read_csv('conversion rate challenge/conversion_data_train.csv')
test_df = pd.read_csv('conversion rate challenge/conversion_data_test.csv')
submission_df = pd.read_csv('conversion rate challenge/submission_SYNDICATE_FINAL_CUT.csv')

# Re-Train Formula Model to get the "Monster Proba" (Base Prob)
# We need this to ensure we don't pick up trash.
full_df = pd.concat([train_df.drop('converted', axis=1), test_df], axis=0).reset_index(drop=True)

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

print("⚙️ Training Base Probability Model...")
X_train = full_df.iloc[:len(train_df)]
X_test = full_df.iloc[len(train_df):]
pipeline.fit(X_train, train_df['converted'])

probs = pipeline.predict_proba(X_test)[:, 1]

# 2. DEFINE THE HEIST RULES
# "Monster" Stats from Test Set
p95_pages = X_test['total_pages_visited'].quantile(0.95)
p95_interaction = X_test['interaction'].quantile(0.95)

print(f"📊 Outlier Thresholds (Test Set P95):")
print(f"   - Pages:       > {p95_pages:.2f}")
print(f"   - Interaction: > {p95_interaction:.2f}")

# Identify The Targets
mask_china = (X_test['country'] == 'China')
mask_monster_pages = (X_test['total_pages_visited'] >= p95_pages)
mask_monster_inter = (X_test['interaction'] >= p95_interaction)
mask_safety_prob = (probs >= 0.30)

# The Intersection
heist_targets = mask_china & mask_monster_pages & mask_monster_inter & mask_safety_prob

# 3. EXECUTE THE HEIST
current_preds = submission_df['converted'].values.copy()
heist_indices = np.where(heist_targets)[0]

print(f"\n🕵️ HEIST REPORT:")
print(f"   Targets Identified: {len(heist_indices)}")

if len(heist_indices) > 0:
    for idx in heist_indices:
        user_row = X_test.iloc[idx]
        was_converted = current_preds[idx]
        print(f"   💎 Target #{idx}: Age {user_row['age']} | Pages {user_row['total_pages_visited']} | Prob {probs[idx]:.3f} | Prev Status: {was_converted}")
        
        # FLIP TO 1
        current_preds[idx] = 1

# 4. SAVE ULTIMATE SUBMISSION
print(f"\n✨ FINAL TALLY:")
print(f"   Old Count: {submission_df['converted'].sum()}")
print(f"   New Count: {current_preds.sum()}")

sub = pd.DataFrame({'converted': current_preds})
sub.to_csv('conversion rate challenge/submission_SYNDICATE_ULTIMATE.csv', index=False)
print("✅ Saved 'submission_SYNDICATE_ULTIMATE.csv'. The Braquage is complete.")
