import pandas as pd
import numpy as np

print("="*80)
print("🧠 VERIFICATION : EST-CE QUE 'APPRENDRE PAR CŒUR' MARCHE ?")
print("   (Analyse du Taux Réel des Profils Manqués)")
print("="*80)

# Load Data
df = pd.read_csv('conversion_data_train.csv')

# Create the "Profile Key" (Signature unique)
df['profile'] = (
    df['country'] + "_" + 
    df['age'].astype(str) + "_" + 
    df['source'] + "_" + 
    df['total_pages_visited'].astype(str)
)

# 1. Identify the Global Stats per Profile
profile_stats = df.groupby('profile').agg(
    count=('converted', 'count'),
    conversions=('converted', 'sum'),
    rate=('converted', 'mean')
).reset_index()

# 2. Focus on the "Missed Opportunity Zone" (US/UK, 8-11 Pages)
# We know from previous analysis that this is where we lose people.
# Let's filter for this demographic manually to see their stats.
target_mask = (
    df['country'].isin(['US', 'UK']) & 
    df['total_pages_visited'].between(8, 11) &
    (df['age'] >= 20) & (df['age'] <= 35)
)

targets = df[target_mask].copy()
targets['profile'] = (
    targets['country'] + "_" + 
    targets['age'].astype(str) + "_" + 
    targets['source'] + "_" + 
    targets['total_pages_visited'].astype(str)
)

# Group these specific targets
target_stats = targets.groupby('profile').agg(
    count=('converted', 'count'),
    conversions=('converted', 'sum'),
    rate=('converted', 'mean')
).sort_values('count', ascending=False).reset_index()

print(f"Analyse de la Zone Critique (US/UK, 20-35 ans, 8-11 pages)")
print(f"Nombre de profils uniques : {len(target_stats)}")
print("-" * 60)

# Check how many of these profiles have a rate > 0.5
# If rate > 0.5, we SHOULD predict 1. If we predicted 0, it's an error.
start_profiles = target_stats[target_stats['count'] > 5].head(20)

print("Top 20 Profils les plus fréquents dans cette zone :")
print(f"{'Profil':<35} | {'Count':<5} | {'Conv':<5} | {'Rate':<7} | {'Verdict'}")
print("-" * 80)

winnable_cases = 0
total_volume = 0
noise_volume = 0

for idx, row in start_profiles.iterrows():
    p, c, s, r = row['profile'], int(row['count']), int(row['conversions']), row['rate']
    
    verdict = "❓"
    if r > 0.5:
        verdict = "✅ OUI (Winner)"
        winnable_cases += c
    elif r < 0.4:
        verdict = "❌ NON (Bruit)"
        noise_volume += c
    else:
        verdict = "⚠️ GRIS"
    
    print(f"{p:<35} | {c:<5} | {s:<5} | {r:.1%}   | {verdict}")

print("-" * 80)

# Global Conclusion on the "Cheat" Strategy
print("\n🔎 CONCLUSION SUR LA STRATÉGIE 'TRICHE' (PAR CŒUR) :")
n_winners = len(target_stats[(target_stats['rate'] > 0.5) & (target_stats['count'] > 3)])
n_losers = len(target_stats[(target_stats['rate'] < 0.5) & (target_stats['count'] > 3)])

print(f"Dans la zone critique, il y a :")
print(f"   - {n_winners} profils 'Gagnants' (>50% succes)")
print(f"   - {n_losers} profils 'Perdants' (<50% succes)")

if n_winners > n_losers * 0.1: # Threshold of usefulness
    print("👉 CA VAUT LE COUP ! Il y a des pépites cachées.")
else:
    print("👉 FAUSSE PISTE. La plupart des profils ont des taux de 20-30%.")
    print("   Si on force 'Oui', on se trompe 7 fois sur 10.")
    print("   Les 'manqués' sont juste les 30% de chanceux d'un groupe perdant.")
