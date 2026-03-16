
import pandas as pd
import numpy as np
import os

# Chemins attendus
FILES = {
    "V1": "conversion rate challenge/divination_v1_fast_predictions.csv",
    "V2": "conversion rate challenge/divination_v2_predictions.csv",
    "V3": "conversion rate challenge/divination_v3_features_predictions.csv"
}

print("📥 Chargement des bulletins de vote...")
votes = pd.DataFrame()
for name, path in FILES.items():
    if os.path.exists(path):
        df = pd.read_csv(path)
        col = 'converted' if 'converted' in df.columns else 'prediction'
        votes[name] = df[col]
        print(f"  ✅ {name} chargé ({df[col].sum()} OUI)")
    else:
        print(f"  ❌ {name} MANQUANT ({path})")

if len(votes.columns) < 3:
    print("\n⚠️ ATTENTION : Il manque des sénateurs ! Arrêt.")
    exit()
else:
    print("\n🔔 Quorum atteint (3/3).")

# 1. Vote Majoritaire Simple
votes['Somme'] = votes.sum(axis=1)
votes['Verdict_Base'] = (votes['Somme'] >= 2).astype(int)

# 2. Chargement des données Test pour le rattrapage
try:
    test_df = pd.read_csv('conversion rate challenge/conversion_data_test.csv')

    # Condition : Vote 1/3 (donc NON) MAIS Profil "Mariage Frères"
    condition_rattrapage = (
        (test_df['new_user'] == 0) & 
        (test_df['total_pages_visited'] >= 10) & 
        (test_df['total_pages_visited'] <= 16) &
        (votes['Somme'] == 1)
    )

    n_rep = condition_rattrapage.sum()
    print(f"⛑️ Amendement 'Mariage Frères' : {n_rep} utilisateurs repêchés.")

    votes['Verdict_Final'] = votes['Verdict_Base']
    votes.loc[condition_rattrapage, 'Verdict_Final'] = 1
    
except Exception as e:
    print(f"⚠️ Erreur lors du rattrapage (fichier test manquant?): {e}")
    votes['Verdict_Final'] = votes['Verdict_Base']

# Export
filename = 'conversion rate challenge/submission_LE_SENAT_V8.csv'
submission = votes[['Verdict_Final']].rename(columns={'Verdict_Final': 'converted'})
submission.to_csv(filename, index=False)

print(f"✅ Soumission prête : {filename}")
