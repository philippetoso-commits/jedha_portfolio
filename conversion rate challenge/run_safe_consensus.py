import pandas as pd
import numpy as np

print("="*80)
print("🛡️ SAFE CONSENSUS : ADAPTING THE PROFESSOR'S INSIGHT")
print("="*80)

# Files
file_train = 'conversion_data_train.csv'
file_test = 'conversion_data_test.csv'
file_constrained = 'submission_POISSON_CONSTRAINED.csv'

df_train = pd.read_csv(file_train)
df_test = pd.read_csv(file_test)
df_const = pd.read_csv(file_constrained)

# 1. Feature Engineering (Exact Match Key)
def create_key(df):
    return (
        df['country'].astype(str) + "_" + 
        df['age'].astype(str) + "_" + 
        df['new_user'].astype(str) + "_" + 
        df['source'].astype(str) + "_" + 
        df['total_pages_visited'].astype(str)
    )

df_train['key'] = create_key(df_train)
df_test['key'] = create_key(df_test)

# 2. Build History (Lookup)
history = df_train.groupby('key')['converted'].agg(['sum', 'count']).reset_index()
history.rename(columns={'sum': 'n_conv', 'count': 'n_trials'}, inplace=True)

# 3. Bayesian Smoothing (The "Safe" Part)
# Global Mean roughly 0.03. 
# We want prior to be weak but stabilizing.
prior_alpha = 1.0  # Equivalent to ~1 success
prior_beta = 30.0  # Equivalent to ~30 trials (implies ~3.3% rate)

history['safe_prob'] = (history['n_conv'] + prior_alpha) / (history['n_trials'] + prior_beta)
history['raw_prob'] = history['n_conv'] / history['n_trials']

# Merge into Test
df_test = df_test.merge(history, on='key', how='left')

# 4. Decision Logic
# Base: Poisson Constrained
final_preds = df_const['converted'].values.copy()
updates_count = 0

# Threshold for Safe Probability
# If Safe Prob > 0.50, we override to 1.
# If Safe Prob < 0.20, we override to 0 (optional, but constrained is usually good at rejection).
# Let's focus on Positive Override (Recall Boost) which is the Professor's goal.

param_threshold_safe = 0.50
param_min_support = 5 # Need at least 5 historical twins to trust the consensus

# Identify Candidates for Override
mask_override = (
    (df_test['n_trials'] >= param_min_support) & 
    (df_test['safe_prob'] > param_threshold_safe) &
    (final_preds == 0) # Only interested in changing 0 -> 1
)

final_preds[mask_override] = 1
updates_count = mask_override.sum()

print(f"Stats Lookup:")
print(f"   Profils Test avec Historique : {df_test['n_trials'].notnull().sum()} / {len(df_test)}")
print(f"   Overrides Appliqués (0->1)   : {updates_count}")

# Save
submission = pd.DataFrame({'converted': final_preds.astype(int)})
submission.to_csv('submission_SAFE_CONSENSUS.csv', index=False)
print("✅ Submission Saved: submission_SAFE_CONSENSUS.csv")
print(f"   Total Conversions : {submission['converted'].sum()}")

# FORENSIC CHECK (Self-Contained)
if updates_count > 0:
    print("\n🔎 VÉRIFICATION IMMÉDIATE DES OVERRIDES")
    # Retrieve the Train stats of the overrides
    overridden_keys = df_test.loc[mask_override, 'key'].unique()
    subset_hist = history[history['key'].isin(overridden_keys)]
    
    avg_real_rate = subset_hist['n_conv'].sum() / subset_hist['n_trials'].sum()
    print(f"   Taux de Conversion Réel (Train) des profils repêchés : {avg_real_rate:.2%}")
    
    if avg_real_rate > 0.5:
        print("   ✅ VERDICT : L'Override est statistiquement JUSTIFIÉ.")
    else:
        print("   ❌ VERDICT : L'Override est risqué (Taux Réel < 50%).")
else:
    print("\nℹ️ Aucun changement par rapport au modèle Constrained.")
