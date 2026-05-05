import pandas as pd
import numpy as np
import os

# Chemins attendus
FILES = {
    "V1": "conversion rate challenge/divination_v1_fast_predictions.csv",
    "V2": "conversion rate challenge/divination_v2_predictions.csv",
    "V3": "conversion rate challenge/divination_v3_features_predictions.csv"
}

print("📥 [V9] Chargement des bulletins de vote...")
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

# 2. Chargement des données Test pour les amendements
try:
    test_df = pd.read_csv('conversion rate challenge/conversion_data_test.csv')

    # --- AMENDEMENT 1 : MARIAGE FRÈRES (V8) ---
    # Cible : Anciens, 10-16 pages, Vote=1/3
    cond_mariage_freres = (
        (test_df['new_user'] == 0) & 
        (test_df['total_pages_visited'] >= 10) & 
        (test_df['total_pages_visited'] <= 16) &
        (votes['Somme'] == 1)
    )
    n_mf = cond_mariage_freres.sum()
    print(f"⛑️ Amendement 'Mariage Frères' : {n_mf} utilisateurs repêchés.")

    # --- AMENDEMENT 2 : ERASMUS (V9) ---
    # Cible : Nouveaux, 8-16 pages, UK/DE, <25 ans, Vote < 2 (0 ou 1)
    # Note: On repêche même ceux qui ont 0 vote si le profil est parfait
    cond_erasmus = (
        (test_df['new_user'] == 1) & 
        (test_df['total_pages_visited'] >= 8) & 
        (test_df['total_pages_visited'] <= 16) &
        (test_df['country'].isin(['Germany', 'UK'])) &
        (test_df['age'] < 25) &
        (votes['Somme'] < 2)
    )
    n_erasmus = cond_erasmus.sum()
    print(f"🇪🇺 Amendement 'Erasmus' : {n_erasmus} utilisateurs repêchés.")

    # Application des deux amendements
    votes['Verdict_Final'] = votes['Verdict_Base']
    votes.loc[cond_mariage_freres, 'Verdict_Final'] = 1
    votes.loc[cond_erasmus, 'Verdict_Final'] = 1
    
    total_repec = (votes['Verdict_Final'] > votes['Verdict_Base']).sum()
    print(f"\n📊 TOTAL REPÊCHÉS V9 : {total_repec} (dont {total_repec - n_mf - n_erasmus} chevauchements)")
    
except Exception as e:
    print(f"⚠️ Erreur lors du rattrapage (fichier test manquant?): {e}")
    # En cas d'erreur, on garde le vote de base (pire cas)
    votes['Verdict_Final'] = votes['Verdict_Base']

# Export
filename = 'conversion rate challenge/submission_LE_SENAT_V9.csv'
submission = votes[['Verdict_Final']].rename(columns={'Verdict_Final': 'converted'})
submission.to_csv(filename, index=False)

print(f"✅ Soumission prête : {filename}")
