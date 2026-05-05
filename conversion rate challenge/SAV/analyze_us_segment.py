
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

print("📥 Loading Data...")
df = pd.read_csv('conversion rate challenge/conversion_data_train.csv')

# Filter for the Target Segment: US, Age 20-30
target_segment = df[(df['country'] == 'US') & (df['age'] >= 20) & (df['age'] <= 30)]
print(f"🇺🇸 Focus Segment: US Users aged 20-30")
print(f"   Population: {len(target_segment)}")
print(f"   Conversions: {target_segment['converted'].sum()}")
print(f"   Rate: {target_segment['converted'].mean():.4f}")

# Compare Page Thresholds
print("\n📊 Conversion Rate by Pages Visited (US 20-30 vs Global):")
for pages in range(1, 25):
    rate_target = target_segment[target_segment['total_pages_visited'] == pages]['converted'].mean()
    rate_global = df[df['total_pages_visited'] == pages]['converted'].mean()
    count_target = len(target_segment[target_segment['total_pages_visited'] == pages])
    
    # Highlight potential opportunities
    marker = ""
    if rate_target > 0.3 and count_target > 50:
        marker = "🔥 High Potential"
    elif rate_target > 0.05 and rate_target < 0.3:
        marker = "⚠️ Grey Zone"
        
    print(f"   Pages: {pages:2d} | US 20-30 Rate: {rate_target:.4f} (n={count_target:<4}) | Global Rate: {rate_global:.4f} {marker}")

# New User Impact for this segment?
print("\n🆕 New User Impact on US 20-30:")
print(target_segment.groupby('new_user')['converted'].mean())

# Source Impact?
print("\n📢 Source Impact on US 20-30:")
print(target_segment.groupby('source')['converted'].mean())
