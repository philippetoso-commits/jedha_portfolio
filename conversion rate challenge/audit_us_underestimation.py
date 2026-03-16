
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import HistGradientBoostingClassifier

print("🕵️ AUDIT: IS THE MODEL BLIND TO AMERICANS? 🕵️")
print("📥 Loading Train Data...")
df = pd.read_csv('conversion rate challenge/conversion_data_train.csv')

# Feature Engineering Minimal (Standard Global Model)
X = df.drop('converted', axis=1)
y = df['converted']
X['pages_per_age'] = X['total_pages_visited'] / (X['age'] + 0.1)
# Simple encoding
for col in ['country', 'source']:
    X[col] = X[col].astype('category')

print("⚙️ Generating Unbiased OOF Predictions (Global Model)...")
# On utilise un modèle puissant mais standard
model = HistGradientBoostingClassifier(learning_rate=0.05, max_depth=6, random_state=42)

# Cross-Val Predict (Probas) pour avoir la vision du modèle sur tout le Train
y_pred_proba = cross_val_predict(model, X, y, cv=5, method='predict_proba')[:, 1]

# Ajout au DF pour analyse
df['proba_global'] = y_pred_proba

# --- SEGMENT CRITIQUE : US / 20-30 ans / 11-13 Pages ---
print("\n🇺🇸 ZOOM ON THE 'GREY ZONE' (US / 20-30yo / 11-13 Pages) 🇺🇸")
mask_target = (df['country'] == 'US') & (df['age'] >= 20) & (df['age'] <= 30) & (df['total_pages_visited'].between(11, 13))
segment = df[mask_target]

p_predicted = segment['proba_global'].mean()
p_actual = segment['converted'].mean()

print(f"   Population Size:      {len(segment)}")
print(f"   🤖 Model Prediction:  {p_predicted:.4f} (Avg Probability)")
print(f"   ✅ Actual Reality:    {p_actual:.4f} (Real Conversion Rate)")
print(f"   📉 Gap (Bias):        {p_predicted - p_actual:.4f}")

if p_predicted < p_actual:
    print(f"\n🚨 VERDICT: UNDERESTIMATION CONFIRMED!")
    print(f"   The model thinks probability is {p_predicted*100:.1f}%, but reality is {p_actual*100:.1f}%.")
    print(f"   It is missing {p_actual - p_predicted:.2%} of the signal.")
else:
    print(f"\n✅ VERDICT: No Underestimation. The model is calibrated.")

# Check thresholds
print("\n🚦 Threshold Analysis (Standard Threshold 0.50):")
print(f"   How many actually converted? {segment['converted'].sum()}")
print(f"   How many did the Global Model predict (Prob > 0.5)? {(segment['proba_global'] > 0.5).sum()}")
print(f"   => Missed Conversions in this tiny segment: {segment['converted'].sum() - (segment['proba_global'] > 0.5).sum()}")
