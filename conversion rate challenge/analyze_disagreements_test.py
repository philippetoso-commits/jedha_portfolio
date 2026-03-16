import pandas as pd
import numpy as np

# Files
file_audit = 'submission_AUDIT_RECOVERY.csv'
file_poisson = 'submission_POISSON_OPTIMIZED_V3.csv'
file_test = 'conversion_data_test.csv'

print("🕵️‍♂️ ANALYSE DES DÉSACCORDS : Audit Recovery vs Poisson Optimized")
print("="*80)

df_audit = pd.read_csv(file_audit)
df_poisson = pd.read_csv(file_poisson)
df_test = pd.read_csv(file_test)

# Merge predictions
df_test['pred_audit'] = df_audit['converted']
df_test['pred_poisson'] = df_poisson['converted']

# 1. Identify Disagreements
disagreements = df_test[df_test['pred_audit'] != df_test['pred_poisson']].copy()

n_disagreements = len(disagreements)
print(f"Nombre total de désaccords : {n_disagreements}")

if n_disagreements == 0:
    print("Aucun désaccord ! Les modèles sont identiques.")
    exit()

# 2. Categorize Disagreements
# Type A: Audit=1, Poisson=0 (Audit Only)
audit_only = disagreements[disagreements['pred_audit'] == 1]
# Type B: Audit=0, Poisson=1 (Poisson Only)
poisson_only = disagreements[disagreements['pred_poisson'] == 1]

print(f"\n👉 Cas Audit=1 / Poisson=0 (Exclusifs Audit) : {len(audit_only)}")
print(f"👉 Cas Audit=0 / Poisson=1 (Exclusifs Poisson) : {len(poisson_only)}")

# 3. Deep Dive Analysis
def analyze_profiles(df, label):
    if len(df) == 0: return
    print(f"\n🔍 Analyse des Profils : {label}")
    print("-" * 40)
    print(df[['country', 'age', 'total_pages_visited', 'source']].describe(include='all'))
    print("\n   Age Moyen   : ", df['age'].mean())
    print("   Pages Moy.  : ", df['total_pages_visited'].mean())
    print("   Pays Freq.  : ", df['country'].value_counts().to_dict())
    
    # Check Safety
    # Safe = Pages >= 7 or (Age >= 25 and Pages >= 5)
    safe = df[ (df['total_pages_visited'] >= 7) | ((df['age'] >= 25) & (df['total_pages_visited'] >= 5)) ]
    print(f"   🛡️ Profils 'Sûrs' (Pages>=7 ou Age>=25) : {len(safe)} / {len(df)}")
    
analyze_profiles(audit_only, "Gagnés par Audit (Perdus par Poisson)")
analyze_profiles(poisson_only, "Gagnés par Poisson (Perdus par Audit)")

# 4. Verdict Logic
print("\n" + "="*80)
print("⚖️ VERDICT AUTOMATIQUE")
print("="*80)

score_audit = 0
score_poisson = 0

# Evaluator: High Pages / Mature Age = Likely Conversion
# Young / Low Pages = Noise

# Audit Only Quality
if len(audit_only) > 0:
    avg_pages_audit = audit_only['total_pages_visited'].mean()
    if avg_pages_audit > 10:
        print("✅ Les exclusifs Audit ont beaucoup de pages (>10). Audit a raison de les garder.")
        score_audit += 1
    elif avg_pages_audit < 6:
         print("⚠️ Les exclusifs Audit ont peu de pages (<6). Risque de Faux Positifs.")
         score_poisson += 1 # Poisson was right to drop them
    else:
         print("ℹ️ Les exclusifs Audit sont en zone grise (6-10 pages).")

# Poisson Only Quality
if len(poisson_only) > 0:
    avg_pages_poisson = poisson_only['total_pages_visited'].mean()
    if avg_pages_poisson > 10:
         print("✅ Les exclusifs Poisson ont beaucoup de pages (>10). Poisson a raison de les prendre.")
         score_poisson += 1
    elif avg_pages_poisson < 6:
         print("⚠️ Les exclusifs Poisson ont peu de pages (<6). Risque de Faux Positifs (Bruit).")
         score_audit += 1 # Audit was right to reject them
    else:
         print("ℹ️ Les exclusifs Poisson sont en zone grise (6-10 pages).")

print("-" * 40)
if score_audit > score_poisson:
    print("🏆 CONSEIL : L'Audit semble plus cohérent.")
elif score_poisson > score_audit:
    print("🏆 CONSEIL : Poisson semble voir des opportunités manquées par l'Audit.")
else:
    print("🤝 CONSEIL : Difficile à dire (Match Nul). Privilégiez la robustesse (Audit).")
