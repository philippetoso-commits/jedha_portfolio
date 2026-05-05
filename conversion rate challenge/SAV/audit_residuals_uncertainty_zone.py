
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_predict, StratifiedKFold

print("🕵️ FORENSIC AUDIT: HUNTING FOR BIAS IN THE HESITATION ZONE 🕵️")
print("📥 Loading Data...")
df = pd.read_csv('conversion rate challenge/conversion_data_train.csv')

# Feature Engineering (Formula Style)
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

print("⚙️ Generating OOF Probabilities (Formula Model)...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_probs = cross_val_predict(pipeline, X, y, cv=cv, method='predict_proba', n_jobs=-1)[:, 1]

# Add to DF
df['prob'] = y_probs
df['residual'] = df['converted'] - df['prob'] # + means Underestimation, - means Overestimation

# DEFINE THE ZONE
p_min, p_max = 0.40, 0.60
zone_df = df[(df['prob'] >= p_min) & (df['prob'] <= p_max)]

print(f"\n🎯 THE ZONE ({p_min}-{p_max}) Analysis:")
print(f"   Samples in Zone: {len(zone_df)} (out of {len(df)})")
print(f"   Global Zone Residual: {zone_df['residual'].mean():.5f} (Should be ~0 if calibrated)")

if len(zone_df) < 50:
    print("⚠️ WARNING: Not enough samples in the zone. Widening or Aborting.")
    exit()

# CLUSTERING BIAS
# Check Country
print("\n🌍 Bias by Country (In The Zone):")
print(zone_df.groupby('country')['residual'].agg(['count', 'mean', 'sum']).sort_values('mean', ascending=False))

# Check Source
print("\n📢 Bias by Source (In The Zone):")
print(zone_df.groupby('source')['residual'].agg(['count', 'mean', 'sum']).sort_values('mean', ascending=False))

# Check Age Bin
zone_df['age_bin'] = pd.cut(zone_df['age'], bins=[0, 20, 30, 40, 60, 100])
print("\n🎂 Bias by Age (In The Zone):")
print(zone_df.groupby('age_bin', observed=False)['residual'].agg(['count', 'mean', 'sum']).sort_values('mean', ascending=False))

# Check Interaction (Country + AgeBin)
print("\n🤝 Bias by Country x Age (In The Zone):")
grouped = zone_df.groupby(['country', 'age_bin'], observed=False)['residual'].agg(['count', 'mean', 'sum'])
# Filter for meaningful sample size
print(grouped[grouped['count'] > 10].sort_values('mean', ascending=False))
