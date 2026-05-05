import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score

# 1. Load Data (Train Set to find generalizable patterns)
print("📥 Loading Training Data...")
df = pd.read_csv('conversion_data_train.csv')

# 2. Isolate the "Problem Segment" (New Users, moderate activity)
# We saw in errors that the missed conversions are often New Users with 8-16 pages
segment_mask = (df['new_user'] == 1) & (df['total_pages_visited'] >= 8) & (df['total_pages_visited'] <= 16)
segment = df[segment_mask].copy()

print(f"🔬 Segment Identified: 'New Users (8-16 pages)'")
print(f"   Total Size: {len(segment)}")
print(f"   Conversion Rate in Segment: {segment['converted'].mean():.4%}")
print(f"   Converters: {segment['converted'].sum()} | Non-Converters: {(segment['converted']==0).sum()}")

# 3. Can we distinguish them?
# Let's train a simple decision tree ONLY on this segment
# Features: Age, Country, Source (encoded)
segment['country_code'] = segment['country'].astype('category').cat.codes
segment['source_code'] = segment['source'].astype('category').cat.codes

X = segment[['age', 'country_code', 'source_code', 'total_pages_visited']]
y = segment['converted']

clf = DecisionTreeClassifier(max_depth=3, class_weight='balanced', random_state=42)
clf.fit(X, y)

# 4. Interpret Results
print("\n🌳 Decision Tree Rules (Can we split them?):")
print(export_text(clf, feature_names=['age', 'country', 'source', 'pages']))

# 5. Correlation Check
print("\n📊 Correlations with Conversion (within this segment):")
corr = segment[['age', 'country_code', 'source_code', 'total_pages_visited', 'converted']].corr()['converted']
print(corr.sort_values(ascending=False))
