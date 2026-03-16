
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# File Paths
file_syndicate_us = 'conversion rate challenge/submission_SYNDICATE_USA.csv'
file_formula = 'conversion rate challenge/submission_THE_FORMULA.csv'
file_sniper = 'conversion rate challenge/submission_FN_SNIPER.csv'

# Load predictions
print("📥 Loading submissions...")
try:
    df_syn_us = pd.read_csv(file_syndicate_us)
    df_form = pd.read_csv(file_formula)
    df_sni = pd.read_csv(file_sniper)
except FileNotFoundError as e:
    print(f"❌ Error: {e}")
    exit(1)

# Vectors
p_syn_us = df_syn_us['converted'].values
p_form = df_form['converted'].values
p_sni = df_sni['converted'].values

# Totals
print(f"\n📊 Total Conversions Predicted:")
print(f"   - FN Sniper (Base):     {p_sni.sum()}")
print(f"   - The Formula (Math):   {p_form.sum()}")
print(f"   - Syndicate USA (Final): {p_syn_us.sum()}")

# Comparisons
def compare(name1, vec1, name2, vec2):
    intersection = ((vec1 == 1) & (vec2 == 1)).sum()
    only1 = ((vec1 == 1) & (vec2 == 0)).sum()
    only2 = ((vec1 == 0) & (vec2 == 1)).sum()
    iou = intersection / ((vec1 == 1) | (vec2 == 1)).sum()
    
    print(f"\n⚔️ {name1} vs {name2} :")
    print(f"   - Common Matches:      {intersection}")
    print(f"   - Only in {name1}:     {only1}")
    print(f"   - Only in {name2}:     {only2}")
    print(f"   - Jaccard Similarity:  {iou:.4f}")

compare("Sniper", p_sni, "Formula", p_form)
compare("Syndicate USA", p_syn_us, "Formula", p_form)

print(f"\n🔍 Deep Dive on Syndicate Extra Value:")
unique_to_syndicate = (p_syn_us == 1) & (p_form == 0)
print(f"   - Conversions found by Syndicate USA but MISSED by Formula: {unique_to_syndicate.sum()}")



