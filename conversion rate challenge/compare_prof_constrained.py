import pandas as pd
import numpy as np

# Files
file_constrained = 'submission_POISSON_CONSTRAINED.csv'
file_prof = 'submission_METHODE_PROF_v2.csv'
file_test = 'conversion_data_test.csv'
file_train = 'conversion_data_train.csv'

print("🔎 DUEL FORENSIQUE : Poisson Constrained vs Méthode Prof V2")
print("="*80)

# Load
df_const = pd.read_csv(file_constrained)
df_prof = pd.read_csv(file_prof)
df_test = pd.read_csv(file_test)
df_train = pd.read_csv(file_train)

# Merge Preds
df_test['pred_const'] = df_const['converted']
df_test['pred_prof'] = df_prof['converted']

print(f"Total Conv Constrained : {df_const['converted'].sum()}")
print(f"Total Conv Prof V2     : {df_prof['converted'].sum()}")
print("-" * 60)

# Define Groups
# Group A: Prof Says YES, Constrained Says NO (Prof Exclusive)
group_prof_wins = df_test[(df_test['pred_prof'] == 1) & (df_test['pred_const'] == 0)].copy()

# Group B: Prof Says NO, Constrained Says YES (Constrained Exclusive)
group_const_wins = df_test[(df_test['pred_prof'] == 0) & (df_test['pred_const'] == 1)].copy()

print(f"Group A (Prof=1, Const=0) : {len(group_prof_wins)} cas")
print(f"Group B (Prof=0, Const=1) : {len(group_const_wins)} cas")

# Function to find Truth in Train
def check_truth(df_disagreement, label):
    if len(df_disagreement) == 0:
        print(f"\n--- {label} : Vide ---")
        return

    print(f"\n🕵️ Analyse : {label} ({len(df_disagreement)} profils)")
    
    unique_profiles = df_disagreement.groupby(['country', 'age', 'total_pages_visited']).size().reset_index(name='count')
    
    matches_found = 0
    total_simulated_trials = 0
    total_simulated_conversions = 0
    
    for _, row in unique_profiles.iterrows():
        c, a, p = row['country'], row['age'], row['total_pages_visited']
        
        mask_train = (
            (df_train['country'] == c) & 
            (df_train['age'] == a) & 
            (df_train['total_pages_visited'] == p)
        )
        
        subset_train = df_train[mask_train]
        if len(subset_train) > 0:
            matches_found += 1
            total_simulated_conversions += subset_train['converted'].sum()
            total_simulated_trials += len(subset_train)
    
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

# Profile Stats
def analyze_profiles(df, label):
    if len(df) == 0: return
    print(f"\n📊 Profil Moyen : {label}")
    print(f"   Age : {df['age'].mean():.1f}")
    print(f"   Pages : {df['total_pages_visited'].mean():.1f}")
    print(f"   Pays : {df['country'].value_counts().head(1).index[0]}")

analyze_profiles(group_prof_wins, "Prof Only")
check_truth(group_prof_wins, "Group A : Prof Oui / Const Non")

analyze_profiles(group_const_wins, "Const Only")
check_truth(group_const_wins, "Group B : Const Oui / Prof Non")
