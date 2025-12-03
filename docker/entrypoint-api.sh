
#!/bin/bash
# entrypoint-api.sh

echo "🚀 Iniciando API Olist Forecasting..."

# Esperar a MLflow si está configurado
if [ -n "$MLFLOW_TRACKING_URI" ]; then
    echo "⏳ Esperando MLflow..."
    sleep 5
fi

# Iniciar API
exec uvicorn api:app --host 0.0.0.0 --port 8000 --reload

