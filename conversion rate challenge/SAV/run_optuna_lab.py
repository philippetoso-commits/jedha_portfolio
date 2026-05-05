
import optuna
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
import warnings
import time

warnings.filterwarnings('ignore')

# 1. Load Data
print("📥 Loading Data...")
train_df = pd.read_csv('conversion rate challenge/conversion_data_train.csv')
target = train_df['converted']
train_df = train_df.drop('converted', axis=1)

# 2. Base Preprocessing
def get_base_data(df):
    df = df.copy()
    # Basic encoding needed for HistGradientBoosting
    cat_cols = ['country', 'source']
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    return df

X_base = get_base_data(train_df)

# 3. Objective Function
def objective(trial):
    # --- A. Feature Synthesis Selection (Bayesian Choice) ---
    use_pages_per_age = trial.suggest_categorical('use_pages_per_age', [True, False])
    use_interaction = trial.suggest_categorical('use_interaction', [True, False])
    use_active_flag = trial.suggest_categorical('use_active_flag', [True, False])
    use_source_freq = trial.suggest_categorical('use_source_freq', [True, False])
    
    # Construct X for this trial
    X_trial = X_base.copy()
    
    if use_pages_per_age:
        X_trial['pages_per_age'] = X_trial['total_pages_visited'] / (X_trial['age'] + 0.1)
        
    if use_interaction:
        X_trial['age_x_pages'] = X_trial['age'] * X_trial['total_pages_visited']
        
    if use_active_flag:
        X_trial['is_active'] = (X_trial['total_pages_visited'] > 2).astype(int)
        
    if use_source_freq:
        # Frequency encoding for source (simple version)
        freq = X_trial['source'].value_counts(normalize=True)
        X_trial['source_freq'] = X_trial['source'].map(freq)

    # --- B. Hyperparameters Optimization ---
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'max_iter': trial.suggest_int('max_iter', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 20, 100),
        'l2_regularization': trial.suggest_float('l2_regularization', 1e-4, 1.0, log=True),
        'loss': 'log_loss',
        'random_state': 42
    }
    
    # Update categorical features indices if we added columns?
    # Actually HistGradientBoosting takes indices. 
    # Base columns are [country, age, new_user, source, total_pages_visited] -> indices 0, 1, 2, 3, 4
    # country=0, source=3. 
    # If we add columns, they are appended at the end, so 0 and 3 remain correct.
    cat_indices = [0, 3] 

    model = HistGradientBoostingClassifier(**params, categorical_features=cat_indices)
    
    # --- C. Evaluation (Stratified K-Fold) ---
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_trial, target, cv=cv, scoring='f1', n_jobs=-1)
    
    return scores.mean()

# 4. Optimization Loop
print("🧪 Starting Optuna Laboratory...")
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=30, show_progress_bar=True)

# 5. Report
print("\n🏆 BEST TRIAL RESULTS 🏆")
print(f"Best CV F1 Value: {study.best_value:.5f}")
print("Best Parameters:")
for key, value in study.best_params.items():
    print(f"  {key}: {value}")

# Interpretation
baseline_f1 = 0.764 # Rough baseline of Sniper (on Test, CV might be different)
print(f"\nComparing to Baseline (approx): {study.best_value:.5f}")
