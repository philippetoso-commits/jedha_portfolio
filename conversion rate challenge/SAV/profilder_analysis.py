
import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print("📥 Loading Data...")
df = pd.read_csv('conversion rate challenge/conversion_data_train.csv')

# Création de Bins pour l'analyse
df['age_group'] = pd.cut(df['age'], bins=[10, 20, 30, 40, 50, 60, 100], labels=['10-20', '20-30', '30-40', '40-50', '50-60', '60+'])
df['pages_group'] = pd.cut(df['total_pages_visited'], bins=[0, 5, 10, 15, 20, 30, 100], labels=['0-5', '5-10', '10-15', '15-20', '20-30', '30+'])

print(f"\n📊 --- TOP PROFILES BY CONVERSION RATE (Min 100 users) ---")
# On groupe par Country, Age, Source
groups = df.groupby(['country', 'age_group', 'source']).agg(
    users=('total_pages_visited', 'count'),
    conversions=('converted', 'sum'),
    conversion_rate=('converted', 'mean')
).reset_index()

# Filtre pour éviter le bruit (groupes trop petits)
top_rate = groups[groups['users'] > 100].sort_values('conversion_rate', ascending=False).head(10)
print(top_rate.to_string(index=False))

print(f"\n💰 --- TOP PROFILES BY VOLUME (Absolute Number of Conversions) ---")
top_vol = groups.sort_values('conversions', ascending=False).head(10)
print(top_vol.to_string(index=False))

print(f"\n🔥 --- SUPER ANALYSIS: PAGES VISITED IMPACT ---")
page_groups = df.groupby(['pages_group']).agg(
    users=('total_pages_visited', 'count'),
    conversions=('converted', 'sum'),
    conversion_rate=('converted', 'mean')
).reset_index()
print(page_groups.to_string(index=False))

print(f"\n🧐 --- THE 'CHINESE MYSTERY' CHECK ---")
china_stats = df[df['country'] == 'China'].groupby('converted').size()
print(f"Total China Users: {len(df[df['country']=='China'])}")
print(f"China Conversions: {df[(df['country']=='China') & (df['converted']==1)].shape[0]}")
