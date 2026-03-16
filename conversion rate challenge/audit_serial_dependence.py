
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from scipy.stats import chi2

print("🕵️ SERIAL DEPENDENCE AUDIT: HUNTING FOR THE \"LAW OF SERIES\" 🕵️")

# 1. LOAD DATA
train_df = pd.read_csv('conversion rate challenge/conversion_data_train.csv')
test_df = pd.read_csv('conversion rate challenge/conversion_data_test.csv')

print(f"   Train Rows: {len(train_df)}")
print(f"   Test Rows:  {len(test_df)}")

# 2. CREATE SIGNATURES
# We need discrete buckets for continuous vars to form a "Barcode" for each user.
def create_signature(df):
    df_sig = df.copy()
    # Age: 5 year buckets
    df_sig['age_bin'] = (df['age'] // 5) * 5
    # Pages: 2 page buckets (to group 11-12, 13-14)
    df_sig['page_bin'] = (df['total_pages_visited'] // 2) * 2
    
    # Signature String
    df_sig['signature'] = (
        df_sig['country'] + "|" +
        df_sig['source'] + "|" +
        df_sig['new_user'].astype(str) + "|" +
        df_sig['age_bin'].astype(str) + "|" +
        df_sig['page_bin'].astype(str)
    )
    return df_sig

df_train_sig = create_signature(train_df)
df_test_sig = create_signature(test_df)

unique_sigs_train = set(df_train_sig['signature'].unique())
unique_sigs_test = set(df_test_sig['signature'].unique())

print("\n🔑 SIGNATURE ANALYSIS:")
print(f"   Unique Signatures in Train: {len(unique_sigs_train)}")
print(f"   Unique Signatures in Test:  {len(unique_sigs_test)}")
overlap = len(unique_sigs_train.intersection(unique_sigs_test))
print(f"   Overlap (Test sigs in Train): {overlap} ({overlap/len(unique_sigs_test)*100:.1f}%)")
# Interpretation: If Overlap is 100%, the Test set is just "More of the same exact profiles".

# 3. SPLIT-STABILITY TEST (Intra-Signature Variance)
print("\n📉 SPLIT-STABILITY TEST (Is the conversion rate 'Too Perfect'?)")
print("   Splitting Train into 5 Folds and checking variance of conversion rate per Signature.")

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_stats = []

# Prepare storage
# Structure: {signature: [p1, p2, p3, p4, p5]}
sig_tracker = {}
sig_counts = {}

# We iterate folds
for fold_idx, (train_idx, val_idx) in enumerate(kf.split(df_train_sig, df_train_sig['converted'])):
    val_fold = df_train_sig.iloc[val_idx]
    
    # Group by signature
    grouped = val_fold.groupby('signature')['converted'].agg(['mean', 'count'])
    
    for sig, row in grouped.iterrows():
        if sig not in sig_tracker:
            sig_tracker[sig] = []
            sig_counts[sig] = []
        sig_tracker[sig].append(row['mean'])
        sig_counts[sig].append(row['count'])

# Constants
k_folds = 5
results = []

for sig, rates in sig_tracker.items():
    # Only analyze signatures present in ALL 5 folds for stability
    if len(rates) == k_folds:
        counts = sig_counts[sig]
        # Weighted mean (Global p for this signature)
        # But for variance test, we treat folds as equal samples if n is roughly equal
        # Let's use simple mean/var for robustness
        
        p_mean = np.mean(rates)
        p_var_obs = np.var(rates, ddof=1) # Observed Variance
        
        avg_n = np.mean(counts)
        
        # Theoretical Variance if I.I.D Binomial:
        # Var(p_hat) = p(1-p) / n
        if avg_n < 10: continue # Skip tiny samples
        
        p_var_exp = (p_mean * (1 - p_mean)) / avg_n
        
        if p_var_exp == 0: continue # p=0 or p=1 (Deterministic)
        
        ratio = p_var_obs / p_var_exp
        
        results.append({
            'signature': sig,
            'p_mean': p_mean,
            'avg_n': avg_n,
            'obs_var': p_var_obs,
            'exp_var': p_var_exp,
            'ratio': ratio
        })

res_df = pd.DataFrame(results)

if len(res_df) == 0:
    print("⚠️ No signatures found with sufficient representation in all 5 folds.")
    exit()

print(f"\n📊 Analyzed {len(res_df)} signatures common to all 5 folds.")
print(f"   Mean Stability Ratio (Obs/Exp Var): {res_df['ratio'].mean():.4f}")
print(f"   Median Stability Ratio:             {res_df['ratio'].median():.4f}")

# Interpretation
# Ratio ~ 1.0 => I.I.D. (Random Noise dominates)
# Ratio << 1.0 => Under-dispersed (Too Stable / Deterministic Quotas)
# Ratio >> 1.0 => Over-dispersed (Hidden Variables / Drift / Instability)

print("\n🕵️ EXTREME CASES:")
print("   Most Result-Deterministic Signatures (Ratio -> 0):")
print(res_df.sort_values('ratio').head(5)[['signature', 'p_mean', 'avg_n', 'ratio']])

print("\n   Most Unstable Signatures (Ratio >> 1):")
print(res_df.sort_values('ratio', ascending=False).head(5)[['signature', 'p_mean', 'avg_n', 'ratio']])

# 4. DUPLICATE CONFIGURATION CHECK
print("\n👯 DUPLICATE CONFIGURATION CHECK:")
# Check frequency of exact row matches (excluding ID if any... oh wait there is no ID)
# We use the raw discrete columns (Country, Age, New, Source, Pages)
cols = ['country', 'age', 'new_user', 'source', 'total_pages_visited']
dupes = train_df.groupby(cols).size().reset_index(name='count')
dupes_dist = dupes['count'].value_counts().sort_index()

print("   Distribution of Identical User Configurations:")
print(dupes_dist.head(10))

# Do these duplicates have consistent labels?
# i.e. If 100 people have exact same config, do 100 convert? or 50?
# Variance within exact duplicates
print("\n   Variance within Exact Duplicates:")
dupes_var = train_df.groupby(cols)['converted'].var() 
# Fill NA for singles (var=NaN) or pure groups (var=0)
clean_vars = dupes_var.dropna()
print(f"   Zero Variance Groups (All 0 or All 1): {sum(clean_vars == 0)} / {len(clean_vars)} ({sum(clean_vars == 0)/len(clean_vars)*100:.1f}%)")
print(f"   Mean Variance: {clean_vars.mean():.4f} (Max possible 0.25)")

# CONCLUSION
global_ratio = res_df['ratio'].median()
if 0.8 < global_ratio < 1.2:
    print("\n✅ VERDICT: Data appears I.I.D. (Ratio ~ 1.0). No hidden series law detected.")
    print("   The generator is likely pure probabilistic (Bernoulli).")
elif global_ratio <= 0.8:
    print("\n🚨 VERDICT: Data is suspiciously STABLE (Ratio < 0.8).")
    print("   The generator implies 'Quotas' or deterministic balancing.")
    print("   Correction Strategy: Trust the Mean, ignore noise.")
else:
    print("\n⚠️ VERDICT: Data is UNSTABLE (Ratio > 1.2).")
    print("   Hidden variables are driving variance. Stratify more!")
