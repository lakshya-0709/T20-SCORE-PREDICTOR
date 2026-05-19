import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None
from sklearn.metrics import r2_score, mean_absolute_error
import pickle
import time

# 1. Load data
print("Loading Dataset_04.csv...")
try:
    df = pd.read_csv('Dataset_04.csv')
except FileNotFoundError:
    print("Error: Dataset_04.csv not found. Please run feature_extraction.py first.")
    exit()

# 2. Setup targets and features
print("Recalibrating target variable to 'remaining_runs'...")
df = df[df['balls_left'] > 0]
df['remaining_runs'] = df['runs_x'] - df['current_score']

X = df.drop(columns=['runs_x', 'remaining_runs'])
y = df['remaining_runs']
groups = df['match_id']

# 3. Group-Based Train/Test Split
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=1)
train_idx, test_idx = next(gss.split(X, y, groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

# Drop match_id from features
X_train = X_train.drop(columns=['match_id'])
X_test = X_test.drop(columns=['match_id'])

# 4. Preprocessing Pipeline
trf = ColumnTransformer([
    ('trf', OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore'), ['batting_team', 'bowling_team', 'city'])
], remainder='passthrough')

# --- Model Tournament Setup ---
models = {}

# A. Tuned Random Forest
models['Random Forest (Tuned)'] = Pipeline(steps=[
    ('step1', trf),
    ('step2', RandomForestRegressor(n_estimators=300, max_depth=15, random_state=1, n_jobs=-1))
])

# B. Tuned XGBoost (if available)
if XGBRegressor:
    models['XGBoost (Tuned)'] = Pipeline(steps=[
        ('step1', trf),
        ('step2', XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=8, random_state=1, n_jobs=-1))
    ])
else:
    print("Warning: xgboost not installed. Skipping XGBoost in comparison.")

# 5. Training and Evaluation
results = []
best_r2 = -np.inf
best_model = None
best_model_name = ""

print("\n--- Model Optimization Tournament ---")
for name, pipe in models.items():
    print(f"\nTraining {name}...")
    start_time = time.time()
    pipe.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    y_pred = pipe.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"Results for {name}:")
    print(f"  R2 Score: {r2:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  Training Time: {training_time:.2f}s")
    
    if r2 > best_r2:
        best_r2 = r2
        best_model = pipe
        best_model_name = name

# 6. Save the Winner
if best_model:
    print(f"\nWINNER: {best_model_name} with R2 = {best_r2:.4f}")
    print("Saving the winning model to pipe.pkl...")
    with open('pipe.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    print("Done! Optimization complete.")
else:
    print("Error: No models were trained.")