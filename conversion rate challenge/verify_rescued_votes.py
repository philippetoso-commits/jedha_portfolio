import pandas as pd
import os

files = {
    "V1": "conversion rate challenge/divination_v1_fast_predictions.csv",
    "V2": "conversion rate challenge/divination_v2_predictions.csv",
    "V3": "conversion rate challenge/divination_v3_features_predictions.csv"
}

df = pd.DataFrame()
for name, path in files.items():
    if os.path.exists(path):
        temp = pd.read_csv(path)
        col = 'converted' if 'converted' in temp.columns else 'prediction'
        df[name] = temp[col]

df['Sum'] = df.sum(axis=1)

test = pd.read_csv('conversion rate challenge/conversion_data_test.csv')

# Rule: New User=0 AND Pages [10, 16] AND Sum=1 (meaning only one senator voted YES)
mask = (test['new_user'] == 0) & (test['total_pages_visited'] >= 10) & (test['total_pages_visited'] <= 16) & (df['Sum'] == 1)

rescued = df[mask]
print(f"Total rescued by rule: {len(rescued)}")
if len(rescued) > 0:
    print("Who was the lone voter for these rescued users?")
    print(rescued[['V1', 'V2', 'V3']].sum())
