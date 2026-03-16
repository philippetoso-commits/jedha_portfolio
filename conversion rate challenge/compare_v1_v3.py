import pandas as pd
import numpy as np

file_v1 = 'submission_THEORY_SINGLE.csv'
file_v3 = 'submission_THEORY_V3_AMENDMENT.csv'

print(f"Comparing:")
print(f"  V1 (Base): {file_v1}")
print(f"  V3 (Amendment): {file_v3}\n")

try:
    df_v1 = pd.read_csv(file_v1)
    df_v3 = pd.read_csv(file_v3)
except FileNotFoundError:
    print("Waiting for files...")
    exit()

preds_v1 = df_v1['converted'].values
preds_v3 = df_v3['converted'].values

print("="*40)
print("📊 Comparison V1 vs V3")
print("="*40)
print(f"Total Conv V1 : {preds_v1.sum()}")
print(f"Total Conv V3 : {preds_v3.sum()}")
print("-" * 40)
print("Confusion Matrix (V1 Rows, V3 Cols):")
print(pd.crosstab(preds_v1, preds_v3, rownames=['V1'], colnames=['V3']))
print("-" * 40)

new_finds = np.sum((preds_v3 == 1) & (preds_v1 == 0))
dropped = np.sum((preds_v3 == 0) & (preds_v1 == 1))

print(f"✨ Gained by Amendment : {new_finds}")
print(f"❌ Dropped by Amendment : {dropped} (Should be 0 if purely additive)")
