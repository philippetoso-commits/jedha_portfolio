import pandas as pd
import numpy as np

print("Chargement du dataset de base...")
df = pd.read_csv('SpeedDating.csv', encoding='ISO-8859-1')

print(f"Taille initiale : {df.shape}")

# 1. Traitement des valeurs aberrantes de la colonne 'wave'
# les vagues 6 à 9 utilisaient un système de notation différent (sur 100 au lieu de 10)
print("Suppression des vagues 6 à 9 (système de notation différent)...")
df_clean = df[~((df["wave"] > 5) & (df["wave"] < 10))].copy()

# 2. Gestion des valeurs manquantes critiques
# On supprime les lignes qui n'ont pas de décision finale (dec) puisqu'on ne peut pas les analyser
print("Suppression des lignes sans décision (dec ou dec_o)...")
df_clean = df_clean.dropna(subset=['dec', 'dec_o'])

# 3. Création des variables explicites pour l'analyse (Feature Engineering basique)
# Objectifs (goals)
goals_mapping = {
    1: 'Seamused a fun night out', 2: 'Meet new people', 3: 'Get a date',
    4: 'Looking for a serious relationship', 5: 'To say I did it', 6: 'Other'
}
df_clean['goal_desc'] = df_clean['goal'].map(goals_mapping)

print(f"Taille après nettoyage : {df_clean.shape}")

output_filename = 'SpeedDating_Cleaned.csv'
print(f"Sauvegarde dans {output_filename}...")
df_clean.to_csv(output_filename, index=False)
print("Terminé !")
