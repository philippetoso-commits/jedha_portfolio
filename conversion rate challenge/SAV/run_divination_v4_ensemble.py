"""
DIVINATION V4 - LE GRAND CONSEIL (SUPER ENSEMBLE)
Fusionne les prédictions de V1, V2 et V3 par vote majoritaire.
"""

import pandas as pd
import os
import numpy as np

print("="*80)
print("🧙‍♂️ DIVINATION V4 - LE GRAND CONSEIL")
print("="*80)

# Chemins
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
V1_PATH = os.path.join(SCRIPT_DIR, 'divination_v1_fast_predictions.csv') # Le champion régénéré
V2_PATH = os.path.join(SCRIPT_DIR, 'divination_v2_predictions.csv')      # Poisson Supremacy
V3_PATH = os.path.join(SCRIPT_DIR, 'divination_v3_features_predictions.csv') # Features Edition

# Vérification présence fichiers
missing = []
for p in [V1_PATH, V2_PATH, V3_PATH]:
    if not os.path.exists(p):
        missing.append(os.path.basename(p))

if missing:
    print(f"❌ ERREUR : Il manque des fichiers de prédictions : {missing}")
    print("   Attendez la fin de la génération de V1...")
    exit(1)

print("⏳ Chargement des prédictions...")
pred_v1 = pd.read_csv(V1_PATH)['converted']
pred_v2 = pd.read_csv(V2_PATH)['converted']
pred_v3 = pd.read_csv(V3_PATH)['converted']

print(f"   V1 (Original) : {pred_v1.sum()} conversions")
print(f"   V2 (Poisson)  : {pred_v2.sum()} conversions")
print(f"   V3 (Features) : {pred_v3.sum()} conversions")

# Calcul du consensus
print("\n🔍 Analyse des désaccords...")
# Combien de fois ils ne sont pas d'accord ?
disagreement = ((pred_v1 != pred_v2) | (pred_v1 != pred_v3) | (pred_v2 != pred_v3)).sum()
print(f"   Nombre de cas litigieux : {disagreement} / {len(pred_v1)} ({disagreement/len(pred_v1):.2%})")

# Vote Majoritaire (Hard Voting)
# Somme des votes (0, 1, 2 ou 3)
votes = pred_v1 + pred_v2 + pred_v3

# Si votes >= 2, alors majorité = 1
final_pred = (votes >= 2).astype(int)

# Analyse du résultat
print(f"\n✅ Vote terminé")
print(f"   Total conversions V4 : {final_pred.sum()}")
print(f"   (V1 seul en avait {pred_v1.sum()})")

# Sauvegarde
output_path = os.path.join(SCRIPT_DIR, 'divination_v4_grand_conseil.csv')
pd.DataFrame({'converted': final_pred}).to_csv(output_path, index=False)

print(f"\n💾 Fichier généré : {output_path}")
print(f"\n{'='*70}")
print("🏆 PRÊT POUR LA SOUMISSION")
print(f"{'='*70}")
