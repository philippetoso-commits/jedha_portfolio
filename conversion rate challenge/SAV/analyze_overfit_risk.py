import pandas as pd
import numpy as np

# Files
file_audit = 'submission_AUDIT_RECOVERY.csv'
file_overfit = 'submission_POISSON_OVERFIT.csv'
file_test = 'conversion_data_test.csv'

print("📉 ANALYSE DE RISQUE : Audit Recovery vs Poisson Overfit")
print("="*80)

df_audit = pd.read_csv(file_audit)
df_overfit = pd.read_csv(file_overfit)
df_test = pd.read_csv(file_test)

# Merge
df_test['pred_audit'] = df_audit['converted']
df_test['pred_overfit'] = df_overfit['converted']

# Disagreements
# Risk A: Overfit rejects what Audit accepts (Potential False Negatives for Overfit)
rejected_by_overfit = df_test[(df_test['pred_audit'] == 1) & (df_test['pred_overfit'] == 0)]

# Risk B: Overfit accepts what Audit rejects (Potential False Positives for Overfit)
accepted_by_overfit = df_test[(df_test['pred_audit'] == 0) & (df_test['pred_overfit'] == 1)]

print(f"🚨 DANGER : Overfit rejette {len(rejected_by_overfit)} cas validés par Audit.")
print(f"🎲 PARI   : Overfit tente {len(accepted_by_overfit)} cas inédits.")

def analyze_risk_group(df, name):
    if len(df) == 0: return
    print(f"\n🔬 Zoom sur : {name}")
    print("-" * 40)
    print(f"   Age Moyen  : {df['age'].mean():.1f}")
    print(f"   Pages Moy. : {df['total_pages_visited'].mean():.1f}")
    
    # Heuristic: High Pages = Likely True Positive
    high_pages = df[df['total_pages_visited'] >= 12]
    print(f"   ❌ Profils 'Béton' (Pages >= 12) : {len(high_pages)} / {len(df)}")
    
    if len(high_pages) / len(df) > 0.5:
        print("   ⚠️ VERDICT : GROS RISQUE. Le modèle jette des conversions quasi-certaines.")
    else:
        print("   ✅ VERDICT : Risque modéré. Le modèle nettoie des cas limites.")

analyze_risk_group(rejected_by_overfit, "Ce que Overfit RATE (vs Audit)")
analyze_risk_group(accepted_by_overfit, "Ce que Overfit INVENTE (vs Audit)")
