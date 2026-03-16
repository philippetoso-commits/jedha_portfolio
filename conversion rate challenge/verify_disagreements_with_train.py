import pandas as pd
import numpy as np

# Files
file_audit = 'submission_AUDIT_RECOVERY.csv'
file_constrained = 'submission_POISSON_CONSTRAINED.csv'
file_test = 'conversion_data_test.csv'
file_train = 'conversion_data_train.csv'

print("🔎 VÉRIFICATION FORENSIQUE : Audit vs Constrained (via Train Truth)")
print("="*80)

# Load
df_audit = pd.read_csv(file_audit)
df_constrained = pd.read_csv(file_constrained)
df_test = pd.read_csv(file_test)
df_train = pd.read_csv(file_train)

# Merge Preds
df_test['pred_audit'] = df_audit['converted']
df_test['pred_constrained'] = df_constrained['converted']

# Define Groups
# Group A: Audit Says YES, Constrained Says NO (Audit Exclusive)
group_audit_wins = df_test[(df_test['pred_audit'] == 1) & (df_test['pred_constrained'] == 0)].copy()

# Group B: Audit Says NO, Constrained Says YES (Constrained Exclusive)
group_const_wins = df_test[(df_test['pred_audit'] == 0) & (df_test['pred_constrained'] == 1)].copy()

print(f"Group A (Audit=1, Constrained=0) : {len(group_audit_wins)} cas")
print(f"Group B (Audit=0, Constrained=1) : {len(group_const_wins)} cas")

# Function to find Truth in Train
def check_truth(df_disagreement, label):
    if len(df_disagreement) == 0:
        print(f"\n--- {label} : Vide ---")
        return

    print(f"\n🕵️ Analyse : {label} ({len(df_disagreement)} profils)")
    
    # We aggregate the profiles in the disagreement set to find the "archetypes"
    # Group by Country, Source, AgeBin, PageBin (approx) to handle sparsity
    # Let's use exact Age and Pages for precision if possible, or bin slightly.
    
    # Metrics
    total_simulated_trials = 0
    total_simulated_conversions = 0
    
    # Iterate unique profiles in disagreement
    # To save time, we group by (Country, Age, Pages)
    unique_profiles = df_disagreement.groupby(['country', 'age', 'total_pages_visited']).size().reset_index(name='count')
    
    matches_found = 0
    
    for _, row in unique_profiles.iterrows():
        c, a, p = row['country'], row['age'], row['total_pages_visited']
        
        # Search in Train
        # Exact match logic
        mask_train = (
            (df_train['country'] == c) & 
            (df_train['age'] == a) & 
            (df_train['total_pages_visited'] == p)
        )
        
        # Relaxed match (Age +/- 1, Pages +/- 0) if no exact match?
        # Let's stay Strict first.
        
        subset_train = df_train[mask_train]
        
        if len(subset_train) > 0:
            matches_found += 1
            conv = subset_train['converted'].sum()
            trials = len(subset_train)
            total_simulated_conversions += conv
            total_simulated_trials += trials
    
    if total_simulated_trials == 0:
        print("   ⚠️ Pas de correspondance exacte trouvée dans le Train.")
        return

    real_conversion_rate = total_simulated_conversions / total_simulated_trials
    
    print(f"   Correspondances Train trouvées : {matches_found} profils types")
    print(f"   Volume Train équivalent        : {total_simulated_trials} utilisateurs")
    print(f"   Conversions Réelles (Train)    : {total_simulated_conversions}")
    print(f"   TO JUSTIFY 'YES' (Rate > 0.5)  : {real_conversion_rate:.4f}")
    
    if real_conversion_rate > 0.5:
        print(f"   ✅ VERDICT : LE OUI EST LÉGITIME. (Taux réel {real_conversion_rate:.2%} > 50%)")
        print(f"      -> Le modèle qui dit 1 a RAISON.")
    else:
        print(f"   ❌ VERDICT : LE OUI EST EXAGÉRÉ. (Taux réel {real_conversion_rate:.2%} < 50%)")
        print(f"      -> Le modèle qui dit 0 a RAISON.")

check_truth(group_audit_wins, "Group A : Audit Oui / Constrained Non")
check_truth(group_const_wins, "Group B : Constrained Oui / Audit Non")
