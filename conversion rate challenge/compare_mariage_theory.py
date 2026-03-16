import pandas as pd
import numpy as np
import glob

file_theory = 'submission_THEORY_SINGLE.csv'
# Attempt to find the Mariage file
candidates = glob.glob('submission_mariage*.csv')
if candidates:
    file_mariage = candidates[0]
else:
    # Fallback to the specific name if glob fails (unlikely)
    file_mariage = 'submission_mariagefrere_tosophilippe (1).csv'

print(f"Comparing:")
print(f"  A (Theory): {file_theory}")
print(f"  B (Mariage): {file_mariage}\n")

try:
    df_a = pd.read_csv(file_theory)
    df_b = pd.read_csv(file_mariage)
except FileNotFoundError as e:
    print(f"Error loading files: {e}")
    exit()

preds_a = df_a['converted'].values
preds_b = df_b['converted'].values

conv_a = np.sum(preds_a)
conv_b = np.sum(preds_b)

print("="*40)
print("📊 Comparison Results")
print("="*40)
print(f"Total Conv Theory      : {conv_a}")
print(f"Total Conv Mariage     : {conv_b}")
print("-" * 40)
print("Confusion Matrix:")
# Rows = Theory, Cols = Mariage
print(pd.crosstab(preds_a, preds_b, rownames=['Theory'], colnames=['Mariage']))
print("-" * 40)

# Mariage says YES, Theory says NO -> Missed Opportunity?
missed = np.sum((preds_b == 1) & (preds_a == 0))

# Theory says YES, Mariage says NO -> New Opportunity?
new_finds = np.sum((preds_a == 1) & (preds_b == 0))

print(f"❌ Missed by Theory (Mariage=1, Theory=0) : {missed}")
print(f"✨ New in Theory    (Theory=1, Mariage=0) : {new_finds}")

intersection = np.sum((preds_a == 1) & (preds_b == 1))
union = np.sum((preds_a == 1) | (preds_b == 1))
jaccard = intersection / union if union > 0 else 0

print(f"Conclusion: Overlap Jaccard = {jaccard:.4f}")
