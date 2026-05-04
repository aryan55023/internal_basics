import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import json
import os
import numpy as np

# Load data
df = pd.read_csv("data/training_data.csv")

X = df.drop("rental_price", axis=1)
y = df["rental_price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

mlflow.set_experiment("propval-rental-price")

results = []

def evaluate_model(name, model):
    with mlflow.start_run(run_name=name):
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)

        mlflow.log_param("model", name)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.set_tag("domain", "real_estate")

        mlflow.sklearn.log_model(model, name)

        return {
            "name": name,
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "model_obj": model
        }

models = [
    evaluate_model("LinearRegression", LinearRegression()),
    evaluate_model("RandomForest", RandomForestRegressor(random_state=42))
]

best = min(models, key=lambda x: x["rmse"])

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

import joblib
joblib.dump(best["model_obj"], "models/best_model.pkl")

output = {
    "experiment_name": "propval-rental-price",
    "models": [
        {k: v for k, v in m.items() if k != "model_obj"} for m in models
    ],
    "best_model": best["name"],
    "best_metric_name": "rmse",
    "best_metric_value": best["rmse"]
}

with open("results/step1_s1.json", "w") as f:
    json.dump(output, f, indent=4)

print("Saving model to models/best_model.pkl")
joblib.dump(best["model_obj"], "models/best_model.pkl")