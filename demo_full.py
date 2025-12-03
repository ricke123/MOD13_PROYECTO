# demo_full.py
import requests, mlflow
mlflow.set_tracking_uri("http://127.0.0.1:5001")
mlflow.set_experiment("demo-day")

# 1. Predicción real
resp = requests.post("http://localhost:8000/predict", json={
    "product_category": "electronics",
    "historical_demand": [100,120,110,130,125],
    "promotion_planned": True,
    "seasonality_factor": 1.2
}).json()

# 2. Registrar en MLflow como demo oficial
with mlflow.start_run(run_name="DEMO_DAY"):
    mlflow.log_params({"category": "electronics", "promotion": True})
    mlflow.log_metric("predicted_demand", resp["predicted_demand"])
    mlflow.log_dict(resp, "api_response.json")

print(f"✅ Predicción: {resp['predicted_demand']:.1f} unidades")
print("➡️ Abre MLflow → experiment 'demo-day' → run 'DEMO_DAY'")