
import pandas as pd
import numpy as np

print("🛡️ FINAL SAFETY AUDIT: SYNDICATE USA 🛡️")

# Load Submission & Test Data
sub_path = 'conversion rate challenge/submission_SYNDICATE_USA.csv'
test_path = 'conversion rate challenge/conversion_data_test.csv'

print(f"📥 Loading '{sub_path}'...")
sub = pd.read_csv(sub_path)
df_test = pd.read_csv(test_path)

# Merge to check features
df_check = pd.concat([df_test, sub], axis=1)

# CHECK 1: The Iron Law (> 20 Pages)
print("\n🔍 CHECK 1: Iron Law (> 18 Pages actually, just to be safe)")
# Note: In Train, >18 pages had 100% conversion except outliers.
# Let's say > 20 is the safe zone.
missed_iron = df_check[(df_check['total_pages_visited'] > 18) & (df_check['converted'] == 0)]

print(f"   Users with > 18 pages: {len(df_check[df_check['total_pages_visited'] > 18])}")
print(f"   Predicted as 0 (MISSES): {len(missed_iron)}")

if len(missed_iron) > 0:
    print("   ❌ ALERT! We missed obvious conversions!")
    print(missed_iron[['total_pages_visited', 'age', 'country', 'source']].head())
else:
    print("   ✅ CLEAN. No high-page users left behind.")

# CHECK 2: The Youth Law (< 18yo & > 15 Pages)
print("\n🔍 CHECK 2: Youth Law (< 18yo & > 15 Pages)")
missed_youth = df_check[(df_check['age'] < 18) & (df_check['total_pages_visited'] > 15) & (df_check['converted'] == 0)]

print(f"   Young High-Activity Users: {len(df_check[(df_check['age'] < 18) & (df_check['total_pages_visited'] > 15)])}")
print(f"   Predicted as 0 (MISSES): {len(missed_youth)}")

if len(missed_youth) > 0:
    print("   ❌ ALERT! We missed energetic youth!")
    print(missed_youth[['total_pages_visited', 'age', 'country', 'source']].head())
else:
    print("   ✅ CLEAN. All young active users captured.")

# OPTIONAL PATCHING
if len(missed_iron) > 0 or len(missed_youth) > 0:
    print("\n🔧 PATCHING SUBMISSION...")
    # Force 1
    ids_to_fix = pd.concat([missed_iron, missed_youth]).index
    sub.loc[ids_to_fix, 'converted'] = 1
    sub.to_csv('conversion rate challenge/submission_SYNDICATE_USA_PATCHED.csv', index=False)
    print(f"   ✅ Fixed {len(ids_to_fix)} errors. Saved as 'submission_SYNDICATE_USA_PATCHED.csv'.")
    print(f"   New Total Conversions: {sub['converted'].sum()}")
else:
    print("\n✨ SUBMISSION IS PERFECT. NO PATCH NEEDED.")
