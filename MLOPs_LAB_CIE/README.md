⸻

:::writing{variant=“standard” id=“84219”}

🏠 PropVal Rental Price Prediction – MLOps Lab CIE

This project implements an end-to-end MLOps pipeline for predicting rental prices using machine learning. It includes experiment tracking, containerization, API serving, and a retraining pipeline.

⸻

📁 Repository Structure

Internals_Basics/
└── MLOPs_Lab_CIE/
    ├── data/
    ├── src/
    ├── models/
    ├── results/
    ├── requirements.txt
    ├── Dockerfile
    └── README.md

⸻

📊 Dataset

* Features:
    * area_sqft (300–3000)
    * bedrooms (1–5)
    * floor_level (1–20)
    * locality_score (1–10)
* Target:
    * rental_price (INR)

⸻

🚀 Task 1 — Experiment Tracking & Model Comparison

Models Used

* Linear Regression
* Random Forest Regressor

Tools

* MLflow (local tracking)

Run Training

python src/train.py

Output

* MLflow logs
* Best model saved in:

models/best_model.pkl

* Results file:

results/step1_s1.json

⸻

🐳 Task 2 — Docker Packaging

Build Docker Image

docker build -t propval-predictor:v1 .

Run Prediction

docker run -v $(pwd)/results:/app/results propval-predictor:v1 \
--area_sqft 1842.3 \
--bedrooms 3 \
--floor_level 14 \
--locality_score 6.9

Output

* Prediction printed in terminal
* JSON output:

results/step2_s3.json

⸻

🌐 Task 3 — FastAPI Serving

Run API

uvicorn src.api:app --port 9000

Endpoints

Health Check

GET /ping

Response:

{
  "status": "running",
  "model": "best_model",
  "version": "1.0"
}

Prediction

POST /score

Request:

{
  "area_sqft": 1842.3,
  "bedrooms": 3,
  "floor_level": 14,
  "locality_score": 6.9
}

Response:

{
  "prediction": 41069.83
}

Output

results/step3_s4.json

⸻

🔄 Task 4 — Retraining Pipeline

Run Retraining

python src/retrain.py

Logic

* Combine old + new data
* Retrain best model
* Compare RMSE
* Promote only if improvement ≥ 1.0

Output

results/step4_s8.json

⸻

📦 Requirements

Install dependencies:

pip install -r requirements.txt

⸻

⚙️ Technologies Used

* Python
* scikit-learn
* MLflow
* FastAPI
* Docker

⸻

✅ Key Features

✔ Experiment tracking with MLflow
✔ Model comparison (RMSE-based selection)
✔ Dockerized CLI prediction
✔ REST API with validation
✔ Automated retraining pipeline
✔ JSON-based outputs for evaluation

⸻

📌 Notes

* All train/test splits use:

random_state = 42
test_size = 0.2

* Dataset is not modified
* JSON files in results/ act as proof of execution

⸻

👨‍💻 Author

Aryan A
BMS College of Engineering
MLOps Lab CIE – 2026

⸻

:::

⸻

