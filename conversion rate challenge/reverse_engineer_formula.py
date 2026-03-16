
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import log_loss

print("🧪 REVERSE ENGINEERING THE GENERATOR FORMULA 🧪")
print("📥 Loading Data...")
df = pd.read_csv('conversion rate challenge/conversion_data_train.csv')

# Hypothesis: The generator uses a simple equation.
# P(conv) = Sigmoid( w1*Pages + w2*Age + w3*Country + ... )

# Let's clean the data for regression
X = df.drop('converted', axis=1)
y = df['converted']

# Feature Engineering "Math Style"
# The generator likely used "Pages per Age" or "Pages^2" or "Age^2"
X['pages_sq'] = X['total_pages_visited'] ** 2
X['age_sq'] = X['age'] ** 2
X['interaction'] = X['total_pages_visited'] * X['age']
X['rate_concept'] = X['total_pages_visited'] / (X['age'] + 1) # Avoid div/0

numeric_features = ['age', 'total_pages_visited', 'pages_sq', 'age_sq', 'interaction', 'rate_concept']
categorical_features = ['country', 'source', 'new_user'] # new_user is categorical

# Pipe
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ]
)

# Model: Logistic Regression (The Equation Finder)
# C=1e9 to remove regularization (we want the finding the EXACT coefficients of the generator)
model = LogisticRegression(solver='lbfgs', max_iter=1000, C=1e9)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', model)
])

print("\n⚙️ Fitting Equation Model (10-Fold CV)...")
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
scores = cross_val_score(pipeline, X, y, cv=cv, scoring='f1', n_jobs=-1)
neg_log_loss = cross_val_score(pipeline, X, y, cv=cv, scoring='neg_log_loss', n_jobs=-1)

print(f"\n📊 FORMULA MODEL RESULTS :")
print(f"   🎯 F1 Score : {scores.mean():.5f}")
print(f"   📉 Log Loss : {-neg_log_loss.mean():.5f}")

# Compare with XGBoost Benchmark
print(f"\n🆚 Comparison:")
print(f"   Gradient Boosting (Black Box) : F1 ~0.764 | LogLoss ~0.040")
print(f"   Equation Model    (White Box) : F1 {scores.mean():.5f} | LogLoss {-neg_log_loss.mean():.5f}")

# Extract Coefficients to see the "Secret Sauce"
pipeline.fit(X, y)
coeffs = pipeline.named_steps['model'].coef_[0]
feature_names = (numeric_features + 
                 list(pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out()))

print("\n🔑 THE SECRET COEFFICIENTS (Generator weights?):")
coef_df = pd.DataFrame({'Feature': feature_names, 'Weight': coeffs})
coef_df['AbsWeight'] = coef_df['Weight'].abs()
print(coef_df.sort_values('AbsWeight', ascending=False).to_string(index=False))

if scores.mean() > 0.75:
    print("\n✅ INCREDIBLE! The Equation Model is almost as good as XGBoost.")
    print("   We have almost reverse-engineered the formula.")
else:
    print("\n⚠️ CONCLUSION: The generator is NON-LINEAR (e.g. If/Else logic).")
    print("   A simple Sigmoid equation cannot capture it. Trees are needed.")
