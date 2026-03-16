import pandas as pd
import numpy as np
import os

file_poisson = 'conversion rate challenge/submission_POISSON_SUPREMACY.csv'
file_recovery = 'conversion rate challenge/submission_AUDIT_RECOVERY.csv'

print(f"Comparing:")
print(f"  A: {file_poisson}")
print(f"  B: {file_recovery}")

if not os.path.exists(file_poisson):
    print(f"Error: {file_poisson} not found!")
    exit()

if not os.path.exists(file_recovery):
    print(f"Error: {file_recovery} not found!")
    exit()

df_a = pd.read_csv(file_poisson)
df_b = pd.read_csv(file_recovery)

# Stats
conv_a = df_a['converted'].sum()
conv_b = df_b['converted'].sum()

# Agreement
agreement = (df_a['converted'] == df_b['converted']).mean()
confusion = pd.crosstab(df_a['converted'], df_b['converted'], rownames=['Poisson'], colnames=['Recovery'])

print("\n" + "="*40)
print(f"📊 Comparison Results")
print("="*40)
print(f"Total Conversions Poisson   : {conv_a}")
print(f"Total Conversions Recovery  : {conv_b}")
print(f"Overlap (Agreement)         : {agreement:.2%}")
print("-" * 40)
print("Confusion Matrix:")
print(confusion)
print("-" * 40)

# Detail Disagreements
only_poisson = confusion.loc[1, 0]
only_recovery = confusion.loc[0, 1]

print(f"Predictions uniques à Poisson (Lost in Recovery) : {only_poisson}")
print(f"Predictions uniques à Recovery (Gained)          : {only_recovery}")

# Jaccard
intersection = confusion.loc[1, 1]
union = conv_a + conv_b - intersection
jaccard = intersection / union if union > 0 else 0
print(f"Jaccard Similarity (on 1s) : {jaccard:.4f}")
