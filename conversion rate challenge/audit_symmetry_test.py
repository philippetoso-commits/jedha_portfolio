
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_predict, StratifiedKFold

print("🔬 SYMMETRY TEST: GLOBAL VS LOCAL BIAS 🔬")

# 1. Load Data
df = pd.read_csv('conversion rate challenge/conversion_data_train.csv')

# Feature Engineering
X = df.drop('converted', axis=1)
y = df['converted']
X['pages_sq'] = X['total_pages_visited'] ** 2
X['age_sq'] = X['age'] ** 2
X['interaction'] = X['total_pages_visited'] * X['age']
X['rate_concept'] = X['total_pages_visited'] / (X['age'] + 1)

# Preprocessing
numeric_features = ['age', 'total_pages_visited', 'pages_sq', 'age_sq', 'interaction', 'rate_concept']
categorical_features = ['country', 'source', 'new_user']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
    ]
)

model = LogisticRegression(solver='lbfgs', max_iter=2000, C=1e9)
pipeline = Pipeline([('preprocessor', preprocessor), ('model', model)])

print("⚙️ Generating OOF Probabilities...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_probs = cross_val_predict(pipeline, X, y, cv=cv, method='predict_proba', n_jobs=-1)[:, 1]

df['prob'] = y_probs
df['residual'] = df['converted'] - df['prob']

# 2. Define Clusters
# 🇩🇪 Allemagne 20–30
mask_de_2030 = (df['country'] == 'Germany') & (df['age'] > 20) & (df['age'] <= 30)
# 🇬🇧 UK 40–60
mask_uk_old = (df['country'] == 'UK') & (df['age'] > 40) & (df['age'] <= 60)
# 🇺🇸 US 20–30
mask_us_2030 = (df['country'] == 'US') & (df['age'] > 20) & (df['age'] <= 30)
# 🇺🇸 < 20
mask_us_young = (df['country'] == 'US') & (df['age'] <= 20)

clusters = {
    "DE 20-30": mask_de_2030,
    "UK 40-60": mask_uk_old,
    "US 20-30": mask_us_2030,
    "US < 20 ": mask_us_young
}

# 3. Define Zones
zones = {
    "LO (0.2-0.3)": (0.20, 0.30),
    "MID (0.4-0.6)": (0.40, 0.60),
    "HI (0.7-0.8)": (0.70, 0.80)
}

print("\n📊 RESIDUAL ANALYSIS BY ZONE (Actual - Predicted):")
print(f"{'Cluster':<10} | {'Zone':<15} | {'Count':<6} | {'Mean Resid':<10} | {'Verdict'}")
print("-" * 65)

for name, mask in clusters.items():
    bias_profile = []
    
    for zone_name, (p_min, p_max) in zones.items():
        # Zones
        mask_zone = (df['prob'] >= p_min) & (df['prob'] <= p_max)
        
        # Intersection
        final_mask = mask & mask_zone
        subset = df[final_mask]
        
        count = len(subset)
        mean_resid = subset['residual'].mean() if count > 0 else 0.0
        
        bias_profile.append(mean_resid)
        
        print(f"{name:<10} | {zone_name:<15} | {count:<6} | {mean_resid:+.5f}   |")
    
    # Interpretation
    lo, mid, hi = bias_profile
    # Check for "Peak at Mid" (Jackpot)
    # Allow some noise, but Mid should be significantly distinguishing
    is_jackpot = False
    
    # Heuristic: Mid is much stronger than LO and HI?
    # Or just distinctive?
    if abs(mid) > 0.03 and abs(mid) > abs(lo) and abs(mid) > abs(hi):
        verdict = "🎰 JACKPOT (Local Bias)"
    elif abs(lo) > 0.03 and abs(mid) > 0.03 and abs(hi) > 0.03 and np.sign(lo)==np.sign(mid)==np.sign(hi):
        verdict = "❌ GLOBAL (Underfitting)"
    else:
        verdict = "⚠️ NOISE / UNCLEAR"
        
    print(f"👉 VERDICT for {name}: {verdict}")
    print("-" * 65)
