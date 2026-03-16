import pandas as pd
import numpy as np

# 1. Load Data
# We need: 
# - Remaining Errors (to see the Gain/Recall)
# - Test Predictions (to see the Impact/Change)
# - Test Features (to apply the rule)

print("📥 Loading datasets...")
remaining_errors = pd.read_csv('conversion rate challenge/remaining_errors.csv')
test_data = pd.read_csv('conversion rate challenge/conversion_data_test.csv')

# Load current V8 predictions (simulated by summing individual csvs)
v1 = pd.read_csv('conversion rate challenge/divination_v1_fast_predictions.csv').iloc[:, 0]
v2 = pd.read_csv('conversion rate challenge/divination_v2_predictions.csv').iloc[:, 0]
v3 = pd.read_csv('conversion rate challenge/divination_v3_predictions.csv').iloc[:, 0] # Using REAL V3 for accurate simulation

current_votes = pd.DataFrame({'V1': v1, 'V2': v2, 'V3': v3})
current_votes['Sum'] = current_votes.sum(axis=1)

# 2. Define Rule V9
# Rule: New User + Pages[8-16] + Country in [Germany, UK] + Age < 25
def apply_rule_v9(df):
    mask = (
        (df['new_user'] == 1) & 
        (df['total_pages_visited'] >= 8) & 
        (df['total_pages_visited'] <= 16) &
        (df['country'].isin(['Germany', 'UK'])) &
        (df['age'] < 25)
    )
    return mask

# 3. Measure Gain (On Known Errors)
errors_caught = apply_rule_v9(remaining_errors)
n_fixed = errors_caught.sum()
print(f"\n✅ POTENTIAL GAIN (Recall):")
print(f"   The rule would have fixed {n_fixed} KNOWN errors from the past.")
print(f"   (out of {len(remaining_errors)} remaining False Negatives)")

# 4. Measure Impact (On Test Set)
# We apply the rule to Flip 'NO' votes to 'YES'
# We focus on cases where the Senate currently votes NO (<2)
test_candidates = apply_rule_v9(test_data)
# Filter for those who are currently rejected (Sum < 2)
flipped_cases = test_candidates & (current_votes['Sum'] < 2)
n_flipped = flipped_cases.sum()

print(f"\n🚀 PROJECTED IMPACT (Test Set):")
print(f"   The rule would FLIP {n_flipped} predictions from NO to YES.")

# 5. Risk Assessment
ratio = n_fixed / len(remaining_errors) * 100
print(f"\n⚖️  RISK RATIO:")
print(f"   Gain: {n_fixed} errors fixed.")
print(f"   Cost: {n_flipped} changes in submission.")
if n_flipped > n_fixed * 3:
    print("   ⚠️ WARNING: High Impact/Gain ratio. Risk of False Positives.")
else:
    print("   ✅ VERDICT: Safe Ratio. Seems like a surgical strike.")

# Print some examples of flipped test users
if n_flipped > 0:
    print("\n🔍 Examples of Rescued Test Users:")
    print(test_data[flipped_cases].head())
