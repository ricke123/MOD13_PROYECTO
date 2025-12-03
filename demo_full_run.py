import mlflow
import requests
import pandas as pd
from src.business_metrics import BusinessMetrics

# 1. Configurar MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5001")
mlflow.set_experiment("olist-demo-day")

# 2. Simular datos reales de 3 meses
actual = [100, 120, 110, 130, 125, 140, 135, 128]
categories = ['electronics'] * 4 + ['home_appliances'] * 4

# 3. Obtener predicciones reales desde tu API
predicted = []
for i, cat in enumerate(categories):
    hist = actual[max(0, i-4):i+1] or [100]
    payload = {
        "product_category": cat,
        "historical_demand": hist,
        "promotion_planned": cat == 'electronics' and len(hist) > 2,
        "seasonality_factor": 1.1 if cat == 'electronics' else 1.0
    }
    resp = requests.post("http://localhost:8000/predict", json=payload).json()
    predicted.append(resp["predicted_demand"])

# 4. Calcular métricas de negocio
calc = BusinessMetrics()
results = calc.calculate_forecast_impact(actual, predicted)

# 5. Loggear en MLflow como demo oficial
with mlflow.start_run(run_name="DEMO_DAY_OFFICIAL"):
    mlflow.log_params({
        "categories_tested": 2,
        "predictions_made": len(predicted),
        "promotion_tested": True
    })
    mlflow.log_metrics({
        "accuracy": results["accuracy_percentage"],
        "roi": results["roi_percentage"],
        "monthly_savings": results["monthly_savings_estimate"],
        "total_cost": results["total_costs_usd"]
    })
    mlflow.log_dict({
        "actual": actual,
        "predicted": predicted,
        "categories": categories
    }, "demo_inputs.json")
    mlflow.log_dict(results, "business_metrics.json")

print("✅ DEMO PREPARADA")
print(f"📊 Precisión: {results['accuracy_percentage']:.1f}%")
print(f"💰 ROI estimado: {results['roi_percentage']:.1f}%")
print(f"📦 Ahorro mensual: ${results['monthly_savings_estimate']:.0f}")
print("\n➡️ Abre MLflow para ver el run 'DEMO_DAY_OFFICIAL'")