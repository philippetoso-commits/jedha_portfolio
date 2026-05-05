
import pandas as pd
import itertools

print("💎 TREASURE HUNDER: SEARCHING FOR GOLDEN POCKETS 💎")
print("📥 Loading Data...")
df = pd.read_csv('conversion rate challenge/conversion_data_train.csv')

# Discretization pour l'analyse brute
df['age_bin'] = pd.cut(df['age'], bins=[0, 18, 25, 35, 45, 60, 100], labels=['<18', '18-25', '25-35', '35-45', '45-60', '60+'])
df['pages_bin'] = pd.cut(df['total_pages_visited'], bins=[0, 8, 12, 16, 20, 100], labels=['Low (<8)', 'Mid (8-12)', 'High (12-16)', 'Super (16-20)', 'Ultra (>20)'])
df['is_new'] = df['new_user'].map({1: 'New', 0: 'Returning'})

# colonnes à combiner
cols_to_combine = ['country', 'source', 'age_bin', 'pages_bin', 'is_new']

# Recherche de niches
print("\n🔍 Scanning combinations (Min 20 users, Rate > 40%)...")

results = []

# On teste des combinaisons de 3 et 4 critères
for r in [3, 4]:
    for combo in itertools.combinations(cols_to_combine, r):
        # GroupBy sur la combinaison
        grouped = df.groupby(list(combo)).agg(
            users=('converted', 'count'),
            conversions=('converted', 'sum'),
            rate=('converted', 'mean')
        ).reset_index()
        
        # Filtrage : On veut des niches significatives mais très fortes
        niches = grouped[(grouped['users'] >= 20) & (grouped['rate'] > 0.40)]
        
        for idx, row in niches.iterrows():
            desc = " + ".join([f"{col}={row[col]}" for col in list(combo)])
            results.append({
                'Description': desc,
                'Users': row['users'],
                'Rate': row['rate'],
                'Conversions': row['conversions']
            })

# Tri et Affichage
results_df = pd.DataFrame(results).sort_values('Rate', ascending=False)
# On dédoublonne un peu (garder le top)
results_df = results_df.drop_duplicates(subset=['Rate', 'Users'])

print(results_df.head(20).to_string(index=False))

print(f"\n💡 Total Niches Found: {len(results_df)}")
