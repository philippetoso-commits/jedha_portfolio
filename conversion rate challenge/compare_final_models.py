import pandas as pd
import numpy as np

file_theory = 'submission_THEORY_SINGLE.csv'
file_constrained = 'submission_POISSON_CONSTRAINED.csv'

print(f"Comparing:")
print(f"  A: {file_theory}")
print(f"  B: {file_constrained}\n")

df_a = pd.read_csv(file_theory)
df_b = pd.read_csv(file_constrained)

preds_a = df_a['converted'].values
preds_b = df_b['converted'].values

conv_a = np.sum(preds_a)
conv_b = np.sum(preds_b)

print("="*40)
print("📊 Comparison Results")
print("="*40)
print(f"Total Conv Theory      : {conv_a}")
print(f"Total Conv Constrained : {conv_b}")
print("-" * 40)
print("Confusion Matrix:")
print(pd.crosstab(preds_a, preds_b, rownames=['Theory'], colnames=['Constrained']))
print("-" * 40)
diff_a_only = np.sum((preds_a == 1) & (preds_b == 0))
diff_b_only = np.sum((preds_a == 0) & (preds_b == 1))

print(f"Predictions uniques à Theory (Potential Noise)      : {diff_a_only}")
print(f"Predictions uniques à Constrained (Potential Miss)  : {diff_b_only}")

intersection = np.sum((preds_a == 1) & (preds_b == 1))
union = np.sum((preds_a == 1) | (preds_b == 1))
jaccard = intersection / union if union > 0 else 0

if jaccard > 0.99:
    print(f"Conclusion: Identical. Jaccard = {jaccard:.4f}")
else:
    print(f"Conclusion: Disagreement. Jaccard = {jaccard:.4f}")

