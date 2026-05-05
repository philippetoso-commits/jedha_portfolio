
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

print("🧼 FINAL POLISH: WASHING SYNDICATE WITH THE FORMULA 🧼")

# 1. Load Data
print("📥 Loading Data & Syndicate Submission...")
train_df = pd.read_csv('conversion rate challenge/conversion_data_train.csv')
test_df = pd.read_csv('conversion rate challenge/conversion_data_test.csv')
sub_syndicate = pd.read_csv('conversion rate challenge/submission_SYNDICATE_USA.csv')

# 2. Train The Formula (Logistic Regression)
# exact same setup as before
train_len = len(train_df)
y = train_df['converted']
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

print("⚙️ Training Formula Model on Full Train Set...")
X_train = full_df.iloc[:train_len]
X_test = full_df.iloc[train_len:]
pipeline.fit(X_train, y)

# 3. Get Formula Probabilities for Test
print("🔮 Calculating Formula Probabilities for Test Set...")
formulas_probs = pipeline.predict_proba(X_test)[:, 1]

# 4. Audit Syndicate
syndicate_preds = sub_syndicate['converted'].values
df_audit = test_df.copy()
df_audit['syndicate_pred'] = syndicate_preds
df_audit['formula_prob'] = formulas_probs

print("\n🕵️ AUDIT REPORT:")

# Case A: Syndicate says YES, Formula says NO WAY (< 0.10)
# These are likely "Hallucinations" or "Rule Oversights".
weird_positives = df_audit[(df_audit['syndicate_pred'] == 1) & (df_audit['formula_prob'] < 0.10)]
print(f"👉 Syndicate YES but Formula < 10% Risk: {len(weird_positives)}")
if len(weird_positives) > 0:
    print(weird_positives[['age', 'country', 'source', 'total_pages_visited', 'formula_prob']].head())

# Case B: Syndicate says NO, Formula says CERTAINLY (> 0.90)
# These are likely "Misses".
weird_negatives = df_audit[(df_audit['syndicate_pred'] == 0) & (df_audit['formula_prob'] > 0.90)]
print(f"👉 Syndicate NO but Formula > 90% Chance: {len(weird_negatives)}")
if len(weird_negatives) > 0:
    print(weird_negatives[['age', 'country', 'source', 'total_pages_visited', 'formula_prob']].head())

# 5. Refining
print("\n🔧 REFINING SUBMISSION...")
refined_preds = syndicate_preds.copy()

# Kill the weird positives (Safety Filter)
if len(weird_positives) > 0:
    print(f"   🔪 Killing {len(weird_positives)} suspicious positives.")
    refined_preds[weird_positives.index] = 0

# Rescue the weird negatives (Safety Net)
if len(weird_negatives) > 0:
    print(f"   🚑 Rescuing {len(weird_negatives)} obvious misses.")
    refined_preds[weird_negatives.index] = 1

# Stats
changes = np.sum(refined_preds != syndicate_preds)
print(f"   ✨ Total Changes: {changes}")
print(f"   📊 Final Conversion Count: {refined_preds.sum()} (New) vs {syndicate_preds.sum()} (Old)")

# Save
sub_refined = pd.DataFrame({'converted': refined_preds})
sub_refined.to_csv('conversion rate challenge/submission_SYNDICATE_FINAL_CUT.csv', index=False)
print("✅ Saved 'submission_SYNDICATE_FINAL_CUT.csv'.")
