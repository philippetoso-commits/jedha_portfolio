import pandas as pd
import numpy as np

file_mariage = 'submission_mariagefrere_tosophilippe.csv'
file_audit = 'submission_AUDIT_PIPELINE.csv'

print(f"Comparing:")
print(f"  A: {file_mariage}")
print(f"  B: {file_audit}")

df_a = pd.read_csv(file_mariage)
df_b = pd.read_csv(file_audit)

# Stats
conv_a = df_a['converted'].sum()
conv_b = df_b['converted'].sum()

# Agreement
agreement = (df_a['converted'] == df_b['converted']).mean()
confusion = pd.crosstab(df_a['converted'], df_b['converted'], rownames=['Mariage'], colnames=['Audit'])

print("\n" + "="*40)
print(f"📊 Comparison Results")
print("="*40)
print(f"Total Conversions Mariage : {conv_a}")
print(f"Total Conversions Audit   : {conv_b}")
print(f"Overlap (Agreement)       : {agreement:.2%}")
print("-" * 40)
print("Confusion Matrix:")
print(confusion)
print("-" * 40)

# Detail Disagreements
only_mariage = confusion.loc[1, 0]
only_audit = confusion.loc[0, 1]

print(f"Predictions uniques à Mariage (Lost) : {only_mariage}")
print(f"Predictions uniques à Audit (Gained) : {only_audit}")

# Jaccard
intersection = confusion.loc[1, 1]
union = conv_a + conv_b - intersection
jaccard = intersection / union if union > 0 else 0
print(f"Jaccard Similarity (on 1s) : {jaccard:.4f}")
