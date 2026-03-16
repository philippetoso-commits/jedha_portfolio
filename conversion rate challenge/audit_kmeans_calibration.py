
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_predict, StratifiedKFold

print("🧭 K-MEANS CALIBRATION AUDIT 🧭")

# 1. Load Data
df_train = pd.read_csv('conversion rate challenge/conversion_data_train.csv')
df_test = pd.read_csv('conversion rate challenge/conversion_data_test.csv')

# 2. Get Predicted Probabilities (Baseline Model)
# We use the "Formula" (Logistic Regression) as the reference well-calibrated global model.
print("⚙️ Generating Reference Probabilities (Formula Model)...")

# Feature Eng for Model
def eng_features(df):
    d = df.copy()
    d['pages_sq'] = d['total_pages_visited'] ** 2
    d['age_sq'] = d['age'] ** 2
    d['interaction'] = d['total_pages_visited'] * d['age']
    d['rate_concept'] = d['total_pages_visited'] / (d['age'] + 1)
    return d

train_eng = eng_features(df_train)
test_eng = eng_features(df_test)

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

# OOF Probs for Train
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_probs = cross_val_predict(pipeline, train_eng, df_train['converted'], cv=cv, method='predict_proba', n_jobs=-1)[:, 1]

# Probs for Test
pipeline.fit(train_eng, df_train['converted'])
test_probs = pipeline.predict_proba(test_eng)[:, 1]

# 3. K-MEANS CLUSTERING on BEHAVIORAL VARIABLES ONLY
# "Exclure toute variable cible, géographique ou catégorielle"
# Variables: Age, Pages, and maybe derived intensity stats.
print("\n🧬 Clustering on Behavioral Features (Age, Pages)...")

behavior_cols = ['age', 'total_pages_visited']

# We add some intensity derived vars for the clustering to match "Usage Intensity"
# Be careful not to leak target proxy too much, but pages/age is fair.
train_cluster_data = df_train[behavior_cols].copy()
train_cluster_data['intensity'] = train_cluster_data['total_pages_visited'] / (train_cluster_data['age'] + 1)

test_cluster_data = df_test[behavior_cols].copy()
test_cluster_data['intensity'] = test_cluster_data['total_pages_visited'] / (test_cluster_data['age'] + 1)

scaler = StandardScaler()
X_cluster_train = scaler.fit_transform(train_cluster_data)
X_cluster_test = scaler.transform(test_cluster_data)

# K-Means K=5
K = 5
kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
train_labels = kmeans.fit_predict(X_cluster_train)
test_labels = kmeans.predict(X_cluster_test)

# 4. ANALYSIS
def analyze_clusters(name, labels, true_y, pred_p, df_raw):
    print(f"\n📊 {name} CLUSTER ANALYSIS:")
    
    results = []
    unique_labels = np.sort(np.unique(labels))
    
    for k in unique_labels:
        mask = (labels == k)
        count = mask.sum()
        pct = count / len(labels) * 100
        
        # Means of features for interpretation
        avg_age = df_raw.loc[mask, 'age'].mean()
        avg_pages = df_raw.loc[mask, 'total_pages_visited'].mean()
        
        # Probabilities
        p_pred = pred_p[mask].mean()
        
        if true_y is not None:
            p_real = true_y[mask].mean()
            delta = p_real - p_pred
            delta_pct = delta * 100
        else:
            p_real = np.nan
            delta = np.nan
            delta_pct = np.nan
            
        results.append({
            'Cluster': k,
            'Count': count,
            'Pct': pct,
            'Avg_Age': avg_age,
            'Avg_Pages': avg_pages,
            'P_Real': p_real,
            'P_Pred': p_pred,
            'Delta': delta,
            'Delta%': delta_pct
        })
        
    res_df = pd.DataFrame(results)
    print(res_df.to_string(float_format=lambda x: f"{x:.4f}"))
    return res_df

res_train = analyze_clusters("TRAIN", train_labels, df_train['converted'], train_probs, df_train)
res_test = analyze_clusters("TEST", test_labels, None, test_probs, df_test)

# 5. DRIFT CHECK
print("\n👀 DRIFT CHECK (Train vs Test Clusters):")
print(f"{'Cluster':<8} | {'Train Pct':<10} | {'Test Pct':<10} | {'Train P_Pred':<12} | {'Test P_Pred':<12} | {'Status'}")
print("-" * 70)

for k in range(K):
    train_row = res_train[res_train['Cluster'] == k].iloc[0]
    test_row = res_test[res_test['Cluster'] == k].iloc[0]
    
    pct_drift = test_row['Pct'] - train_row['Pct']
    pred_drift = test_row['P_Pred'] - train_row['P_Pred']
    
    status = "✅ Stable"
    if abs(pct_drift) > 2.0: status = "⚠️ Size Drift"
    if abs(pred_drift) > 0.01: status += " / ⚠️ Prob Drift"
    
    print(f"{k:<8} | {train_row['Pct']:<5.1f}%     | {test_row['Pct']:<5.1f}%     | {train_row['P_Pred']:<12.4f} | {test_row['P_Pred']:<12.4f} | {status}")

# 6. CALIBRATION CHECK
print("\n🔎 CALIBRATION CHECK (Train Deltas):")
bad_clusters = res_train[res_train['Delta'].abs() > 0.005] # 0.5% tolerance
if len(bad_clusters) > 0:
    print("🚨 FOUND MISCALIBRATED SEGMENTS:")
    print(bad_clusters[['Cluster', 'Avg_Age', 'Avg_Pages', 'P_Real', 'P_Pred', 'Delta%']].to_string())
    print("\n👉 Interpretation: Standard Model fails on these behaviors.")
else:
    print("✅ Model is Locally Calibrated (All Deltas < 0.5%).")
