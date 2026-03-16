
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score

print("🎰 SMART BETTING: STRATIFIED THRESHOLD OPTIMIZATION 🎰")

# 1. SETUP & DATA
print("📥 Loading Data...")
train_df = pd.read_csv('conversion rate challenge/conversion_data_train.csv')
test_df = pd.read_csv('conversion rate challenge/conversion_data_test.csv')

# Feature Engineering (Equation Style)
full_df = pd.concat([train_df.drop('converted', axis=1), test_df], axis=0).reset_index(drop=True)
full_df['pages_sq'] = full_df['total_pages_visited'] ** 2
full_df['age_sq'] = full_df['age'] ** 2
full_df['interaction'] = full_df['total_pages_visited'] * full_df['age']
full_df['rate_concept'] = full_df['total_pages_visited'] / (full_df['age'] + 1)

# Define Clusters
# We use Country + AgeBin (Young/Old) as the "Games"
full_df['age_bin'] = pd.cut(full_df['age'], bins=[0, 25, 40, 100], labels=['Young', 'Mid', 'Sen'])
full_df['cluster'] = full_df['country'] + "_" + full_df['age_bin'].astype(str)

print(f"📊 Defined {full_df['cluster'].nunique()} distinct Betting Markets (Clusters):")
print(full_df['cluster'].unique())

# 2. GET PROBABILITIES (Using the Robust Formula Model)
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

print("\n⚙️ Training Global Probability Model...")
train_len = len(train_df)
X_train = full_df.iloc[:train_len]
y_train = train_df['converted']
X_test = full_df.iloc[train_len:]

pipeline.fit(X_train, y_train)
train_probs = pipeline.predict_proba(X_train)[:, 1]
test_probs = pipeline.predict_proba(X_test)[:, 1]

# 3. OPTIMIZE THRESHOLDS PER CLUSTER
print("\n🎯 Optimizing Thresholds per Market...")
clusters = full_df['cluster'].unique()
cluster_thresholds = {}
validation_f1_scores = []
global_preds_train = np.zeros(train_len)

for c in clusters:
    # Indices for this cluster
    # Note: We need careful indexing because cluster column is in full_df
    mask_train = (full_df.iloc[:train_len]['cluster'] == c).values
    
    if mask_train.sum() < 50:
        # Too small to optimize, use default 0.5
        cluster_thresholds[c] = 0.5
        continue
        
    y_c = y_train[mask_train]
    prob_c = train_probs[mask_train]
    
    # Grid Search Best Threshold
    best_t = 0.5
    best_f1 = 0
    
    if y_c.sum() == 0:
        # No conversions in this cluster (e.g. China Seniors?)
        # Set threshold high to predict 0
        best_t = 0.99
    else:
        for t in np.linspace(0.1, 0.9, 81):
            score = f1_score(y_c, (prob_c >= t).astype(int))
            if score > best_f1:
                best_f1 = score
                best_t = t
    
    cluster_thresholds[c] = best_t
    
    # Store predictions for global score
    global_preds_train[mask_train] = (prob_c >= best_t).astype(int)
    
    print(f"   Category '{c}': Best Threshold = {best_t:.2f} (F1: {best_f1:.3f})")

# 4. APPLY TO TEST
print("\n🔮 Placing Final Bets on Test Set...")
test_preds = np.zeros(len(test_df), dtype=int)

for c in clusters:
    mask_test = (full_df.iloc[train_len:]['cluster'] == c).values
    if mask_test.sum() == 0: continue
    
    thresh = cluster_thresholds.get(c, 0.5)
    prob_c = test_probs[mask_test]
    test_preds[mask_test] = (prob_c >= thresh).astype(int)

# 5. RESULT Analysis
print(f"\n📊 Global Optimized Train F1: {f1_score(y_train, global_preds_train):.5f}")
print(f"   (vs Flat Threshold ~0.769)")

sub = pd.DataFrame({'converted': test_preds})
sub.to_csv('conversion rate challenge/submission_SMART_BETTING.csv', index=False)
print(f"✅ Generated 'submission_SMART_BETTING.csv' with {test_preds.sum()} conversions.")
