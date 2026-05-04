import argparse
import joblib
import numpy as np
import json

parser = argparse.ArgumentParser()

parser.add_argument("--area_sqft", type=float, required=True)
parser.add_argument("--bedrooms", type=int, required=True)
parser.add_argument("--floor_level", type=int, required=True)
parser.add_argument("--locality_score", type=float, required=True)

args = parser.parse_args()

model = joblib.load("models/best_model.pkl")

import pandas as pd

features = pd.DataFrame([{
    "area_sqft": args.area_sqft,
    "bedrooms": args.bedrooms,
    "floor_level": args.floor_level,
    "locality_score": args.locality_score
}])
prediction = model.predict(features)[0]

output = {
    "image_name": "propval-predictor",
    "image_tag": "v1",
    "base_image": "python:3.12-slim",
    "test_input": vars(args),
    "prediction": float(prediction)
}

with open("results/step2_s3.json", "w") as f:
    json.dump(output, f, indent=4)

print(prediction)