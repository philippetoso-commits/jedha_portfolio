
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
try:
    from statsmodels.tsa.stattools import acf
except ImportError:
    # Minimal fallback if statsmodels is not installed
    def acf(x, nlags=20):
        n = len(x)
        mean = np.mean(x)
        var = np.var(x)
        xp = x - mean
        corr = np.correlate(xp, xp, mode='full')[-n:]
        return corr[:nlags+1] / (var * n)

print("🧪 VERIFICATION BERNOULLI I.I.D. 🧪")

# 1️⃣ Charger le dataset
df = pd.read_csv('conversion rate challenge/conversion_data_train.csv')  
y = df['converted'].values 

N = len(y)
print(f"Dataset Size: {N}")

# 2️⃣ Moyenne globale et par segment
segments = {
    'Début ': y[:N//3],
    'Milieu': y[N//3:2*N//3],
    'Fin   ': y[2*N//3:]
}

print("\n--- Analyse par Tiers ---")
for name, seg in segments.items():
    print(f"{name} : Moyenne={seg.mean():.5f}, Nb de 1={seg.sum()}, Nb de 0={len(seg)-seg.sum()}")

# 3️⃣ Run-length analysis
def run_lengths(arr):
    # Fast numpy based run length
    # Find indices where value changes
    # Append -1 and N to handle start/end
    loc_run_start = np.empty(N, dtype=bool)
    loc_run_start[0] = True
    np.not_equal(arr[:-1], arr[1:], out=loc_run_start[1:])
    run_starts = np.nonzero(loc_run_start)[0]
    run_lengths = np.diff(np.append(run_starts, N))
    return run_lengths

runs = run_lengths(y)
mean_run = np.mean(runs)
max_run = np.max(runs)
print(f"\n--- Run Lengths ---")
print(f"Moyenne Longueur Suite : {mean_run:.2f}")
print(f"Max Longueur Suite     : {max_run}")

# Expected Mean Run Length for Bernoulli(p)
p = y.mean()
q = 1-p
# E[Run] = 1/P(change) ? Approximately...
# Actually P(same) = p^2 + q^2. P(change) = 2pq.
# Expected Run Length ~ 1 / (2pq) ??? 
# Let's just print stats.
print(f"Global Conversion Rate (p): {p:.5f}")

# 4️⃣ Autocorrélation
print(f"\n--- Autocorrélation (20 lags) ---")
try:
    autocorr = acf(y, nlags=20)
    print(autocorr)
except Exception as e:
    print(f"Error computing ACF: {e}")

# 5️⃣ Visualisation
plt.figure(figsize=(12,4))
plt.plot(np.cumsum(y)/np.arange(1, N+1), label="Moyenne cumulative")
plt.axhline(y=y.mean(), color='r', linestyle='--', label="Moyenne globale")
plt.title("Vérification Bernoulli i.i.d. : moyenne cumulative vs globale")
plt.xlabel("Index")
plt.ylabel("Conversion rate")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('conversion rate challenge/bernoulli_check.png')
print("\n✅ Plot saved to 'conversion rate challenge/bernoulli_check.png'")
