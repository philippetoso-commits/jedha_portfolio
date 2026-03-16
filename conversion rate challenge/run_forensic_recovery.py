import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import f1_score

# Load Predictions
file_mariage = 'submission_mariagefrere_tosophilippe.csv'
file_audit = 'submission_AUDIT_PIPELINE.csv'
file_test = 'conversion_data_test.csv'
file_train = 'conversion_data_train.csv'

print("🕵️‍♂️ FORENSIC RECOVERY: Analyzing the 31 Lost Cases")
print("="*60)

df_mariage = pd.read_csv(file_mariage)
df_audit = pd.read_csv(file_audit)
df_test = pd.read_csv(file_test)

# Identify Disagreements (Lost: Mariage=1, Audit=0)
mask_lost = (df_mariage['converted'] == 1) & (df_audit['converted'] == 0)
df_lost = df_test[mask_lost].copy()

print(f"Number of Lost Cases: {len(df_lost)}")

if len(df_lost) > 0:
    print("\nCharacteristics of Lost Cases:")
    print("-" * 30)
    print(df_lost[['country', 'age', 'total_pages_visited', 'source']].describe(include='all'))
    print("-" * 30)
    print("\nSample of Lost Cases:")
    print(df_lost[['country', 'age', 'total_pages_visited', 'source']].head())

    # Strategy: Reintegrate High-Confidence Lost Cases
    # Rule: If Mariage was confident enough (it uses specific rules), maybe we trust it 
    # IF the profile is not "sketchy".
    # Check "Sketchy" criteria: China, Young, Low Pages?
    
    print("\nApplying Recovery Logic...")
    # Recovery Rule: Trust Mariage IF (Pages > 6 OR Age > 25)
    # Rationale: Very young low page users are noisy. Mature or engaged users are signals.
    
    recover_mask = (df_lost['total_pages_visited'] >= 6) | (df_lost['age'] >= 25)
    n_recovered = recover_mask.sum()
    
    print(f"Candidates for Recovery (Pages>=6 or Age>=25): {n_recovered} / {len(df_lost)}")
    
    # Create New Submission
    final_preds = df_audit['converted'].copy()
    indices_to_recover = df_lost[recover_mask].index
    
    final_preds.iloc[indices_to_recover] = 1
    
    sub_name = 'submission_AUDIT_RECOVERY.csv'
    pd.DataFrame({'converted': final_preds}).to_csv(sub_name, index=False)
    
    print(f"\n✅ Created {sub_name}")
    print(f"   Original Audit Conversions : {df_audit['converted'].sum()}")
    print(f"   Recovered Conversions      : + {n_recovered}")
    print(f"   Final Total Conversions    : {final_preds.sum()}")
    
else:
    print("No lost cases found to analyze.")
