
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

print("🌪️ STRESS TEST: DRIFT & SENSITIVITY ANALYSIS 🌪️")

# 1. SETUP
train_df = pd.read_csv('conversion rate challenge/conversion_data_train.csv')
test_df = pd.read_csv('conversion rate challenge/conversion_data_test.csv')

train_len = len(train_df)
y_train = train_df['converted']
full_df = pd.concat([train_df.drop('converted', axis=1), test_df], axis=0).reset_index(drop=True)

# Feature Eng
full_df['pages_sq'] = full_df['total_pages_visited'] ** 2
full_df['age_sq'] = full_df['age'] ** 2
full_df['interaction'] = full_df['total_pages_visited'] * full_df['age']
full_df['rate_concept'] = full_df['total_pages_visited'] / (full_df['age'] + 1)

# Clusters
full_df['age_bin'] = pd.cut(full_df['age'], bins=[0, 25, 40, 100], labels=['Young', 'Mid', 'Sen'])
full_df['cluster'] = full_df['country'] + "_" + full_df['age_bin'].astype(str)

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

print("⚙️ Training Probability Model (Observer)...")
# Train on Train set only to establish baseline
pipeline.fit(full_df.iloc[:train_len], y_train)

# Get Probas
train_probs = pipeline.predict_proba(full_df.iloc[:train_len])[:, 1]
test_probs = pipeline.predict_proba(full_df.iloc[train_len:])[:, 1]

train_clusters = full_df.iloc[:train_len]['cluster']
test_clusters = full_df.iloc[train_len:]['cluster']

print("\n📊 1. CLUSTER SHIFT ANALYSIS (Train vs Test Means):")
print(f"{'Cluster':<15} | {'Train Mean':<10} | {'Test Mean':<10} | {'Delta':<8} | {'Status'}")
print("-" * 65)

drift_risks = []

unique_clusters = sorted(full_df['cluster'].unique())
for c in unique_clusters:
    p_train = train_probs[train_clusters == c]
    p_test = test_probs[test_clusters == c]
    
    if len(p_train) == 0 or len(p_test) == 0: continue
    
    m_train = p_train.mean()
    m_test = p_test.mean()
    delta = m_test - m_train
    pct_change = (delta / m_train) * 100 if m_train > 0 else 0
    
    status = "OK"
    if abs(pct_change) > 10: status = "⚠️ DRIFT"
    if abs(pct_change) > 20: status = "🚨 MAJOR"
    
    if status != "OK": drift_risks.append(c)
    
    print(f"{c:<15} | {m_train:.4f}     | {m_test:.4f}     | {pct_change:+.1f}%   | {status}")

print("\n📉 2. SENSITIVITY CHECK (Fragility of Thresholds):")
print(f"   Testing robustness to +/- 0.02 threshold shift on Train data.")
print(f"{'Cluster':<15} | {'Best Thresh':<10} | {'Max F1':<8} | {'F1 Drop (Noise)':<15} | {'Robustness'}")
print("-" * 80)

fragile_clusters = []

for c in unique_clusters:
    mask = (train_clusters == c)
    y_c = y_train[mask]
    prob_c = train_probs[mask]
    
    if len(y_c) < 50 or y_c.sum() == 0: continue
    
    # Find Best T
    best_t = 0.5
    best_f1 = 0
    for t in np.linspace(0.1, 0.9, 81):
        score = f1_score(y_c, (prob_c >= t).astype(int))
        if score > best_f1:
            best_f1 = score
            best_t = t
            
    # Check Noise
    f1_up = f1_score(y_c, (prob_c >= (best_t + 0.02)).astype(int))
    f1_down = f1_score(y_c, (prob_c >= (best_t - 0.02)).astype(int))
    
    worst_drop = best_f1 - min(f1_up, f1_down)
    
    robustness = "✅ Solid"
    if worst_drop > 0.02: robustness = "⚠️ Fragile"
    if worst_drop > 0.05: robustness = "🧱 BRITTLE"
    
    if "Fragile" in robustness or "BRITTLE" in robustness:
        fragile_clusters.append(c)
        
    print(f"{c:<15} | {best_t:.2f}       | {best_f1:.3f}    | -{worst_drop:.3f}          | {robustness}")

print("\n📋 STRATEGIC SUMMARY:")
print(f"   Drifting Clusters: {drift_risks}")
print(f"   Fragile Clusters:  {fragile_clusters}")

if len(drift_risks) == 0 and len(fragile_clusters) == 0:
    print("\n✅ GREEN LIGHT: Smart Betting is SAFE. The system is stable and robust.")
else:
    print("\n🚧 CAUTION: Smart Betting carries risks.")
    if len(fragile_clusters) > 0:
        print("   The optimal thresholds are brittle. A slight noise will tank F1.")
    if len(drift_risks) > 0:
        print("   The test set behaves differently in key segments.")
    print("   👉 RECOMMENDATION: Stick to Syndicate Final Cut (Broader/Safer).")
