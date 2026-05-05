
import pandas as pd
import numpy as np

print("📢 SOURCE ANALYSIS: ADS vs SEO vs DIRECT 📢")
print("📥 Loading Data...")
df = pd.read_csv('conversion rate challenge/conversion_data_train.csv')

# 1. Global Stats
print("\n📊 Global Conversion by Source:")
stats = df.groupby('source')['converted'].agg(['count', 'sum', 'mean']).sort_values('mean', ascending=False)
stats.columns = ['Users', 'Conversions', 'Rate']
print(stats)

# 2. Source x Country
print("\n🌍 Source x Country Interaction:")
print(df.groupby(['country', 'source'])['converted'].mean().unstack())

# 3. Source x Age (is one source better for young/old?)
df['age_bin'] = pd.cut(df['age'], bins=[0, 20, 30, 40, 50, 60, 100])
print("\n🎂 Source x Age Bin Interaction:")
print(df.groupby(['age_bin', 'source'])['converted'].mean().unstack())

# 4. Source x Pages (The Critical Check)
# Does a SEO user convert with fewer pages than an Ads user?
print("\n📄 Conversion Rate at Critical Page Counts (10-15):")
mask_critical = (df['total_pages_visited'] >= 10) & (df['total_pages_visited'] <= 15)
critical_df = df[mask_critical]
print(critical_df.groupby('source')['converted'].agg(['count', 'mean']))

# 5. The "Validation" Question
# Are there weird outliers? E.g. Direct with 0 pages converting?
print("\n🕵️ Anomalies Search (Conversions with < 2 pages):")
anomalies = df[(df['converted'] == 1) & (df['total_pages_visited'] < 2)]
print(anomalies.groupby('source').size())
