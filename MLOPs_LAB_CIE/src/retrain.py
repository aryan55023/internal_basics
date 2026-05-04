import pandas as pd
import joblib
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load data
train_df = pd.read_csv("data/training_data.csv")
new_df = pd.read_csv("data/new_data.csv")

combined = pd.concat([train_df, new_df])

X = combined.drop("rental_price", axis=1)
y = combined["rental_price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Load old model
old_model = joblib.load("models/best_model.pkl")

old_preds = old_model.predict(X_test)
old_rmse = np.sqrt(mean_squared_error(y_test, old_preds))

# Retrain same model type
from sklearn.ensemble import RandomForestRegressor
new_model = RandomForestRegressor(random_state=42)

new_model.fit(X_train, y_train)
new_preds = new_model.predict(X_test)
new_rmse = np.sqrt(mean_squared_error(y_test, new_preds))

improvement = old_rmse - new_rmse

action = "promoted" if improvement >= 1.0 else "kept_champion"

if action == "promoted":
    joblib.dump(new_model, "models/best_model.pkl")

output = {
    "original_data_rows": len(train_df),
    "new_data_rows": len(new_df),
    "combined_data_rows": len(combined),
    "champion_rmse": old_rmse,
    "retrained_rmse": new_rmse,
    "improvement": improvement,
    "min_improvement_threshold": 1.0,
    "action": action,
    "comparison_metric": "rmse"
}

with open("results/step4_s8.json", "w") as f:
    json.dump(output, f, indent=4)