import numpy as np

# 1. Inputs
f1_current = 0.764
pred_current = 935
delta_correct = 11

# 2. Assumptions on True Positives Limit (T)
# Train set has ~3.2% conversion.
# Test set size:
import pandas as pd
test_len = len(pd.read_csv('conversion_data_test.csv'))
print(f"Test Set Size: {test_len}")

# Range of possible True Conversions (T)
# 3.0% -> 948
# 3.2% -> 1012
# 3.4% -> 1075
t_candidates = range(950, 1100, 10)

print(f"\nSimulation: Starting at F1={f1_current} with {pred_current} predictions.")
print(f"Scenario: We add {delta_correct} predictions, and ALL are correct (+11 TP).")
print("-" * 65)
print(f"{'True Coins. (Est)':<20} | {'Current TP':<10} | {'Current Prec':<12} | {'NEW F1':<10} | {'GAIN':<10}")
print("-" * 65)

results = []

for t in t_candidates:
    # Reverse engineer Current TP
    # F1 = 2*TP / (Pred + T)
    # TP = F1 * (Pred + T) / 2
    tp_current = f1_current * (pred_current + t) / 2
    
    # Check consistency (TP cannot exceed Pred or T)
    if tp_current > pred_current or tp_current > t:
        continue
        
    prec_current = tp_current / pred_current
    
    # New Stats
    tp_new = tp_current + delta_correct
    pred_new = pred_current + delta_correct # We added 11 predictions
    t_new = t # Total true conversions doesn't change, we just found them
    
    # New F1
    f1_new = 2 * tp_new / (pred_new + t_new)
    
    gain = f1_new - f1_current
    results.append(gain)
    
    print(f"{t:<20} | {tp_current:<10.1f} | {prec_current:<12.1%} | {f1_new:.5f}    | +{gain:.5f}")

avg_gain = np.mean(results)
print("-" * 65)
print(f"🚀 AVERAGE PROJECTED GAIN : +{avg_gain:.5f}")
print(f"🎯 ESTIMATED NEW SCORE    : {f1_current + avg_gain:.5f}")
