import pandas as pd
import numpy as np

file_v1 = 'submission_THEORY_SINGLE.csv'
file_v2 = 'submission_THEORY_V2_SENSITIVE.csv'

print(f"Comparing:")
print(f"  V1 (Standard): {file_v1}")
print(f"  V2 (Weighted): {file_v2}\n")

try:
    df_v1 = pd.read_csv(file_v1)
    df_v2 = pd.read_csv(file_v2)
except FileNotFoundError:
    print("Waiting for V2 file...")
    exit()

preds_v1 = df_v1['converted'].values
preds_v2 = df_v2['converted'].values

print("="*40)
print("📊 Comparison V1 vs V2")
print("="*40)
print(f"Total Conv V1 : {preds_v1.sum()}")
print(f"Total Conv V2 : {preds_v2.sum()}")
print("-" * 40)
print("Confusion Matrix (V1 Rows, V2 Cols):")
print(pd.crosstab(preds_v1, preds_v2, rownames=['V1'], colnames=['V2']))
print("-" * 40)

# What did V2 find that V1 missed?
new_finds = np.sum((preds_v2 == 1) & (preds_v1 == 0))
dropped = np.sum((preds_v2 == 0) & (preds_v1 == 1))

print(f"✨ Gained by V2 : {new_finds}")
print(f"❌ Dropped by V2 : {dropped}")

if new_finds > dropped:
    print("👉 V2 is MORE AGGRESSIVE.")
else:
    print("👉 V2 is MORE CONSERVATIVE (Unexpected?).")
