#!/bin/bash
# entrypoint-dashboard.sh

echo "📊 Iniciando Dashboard Olist Forecasting..."

# Esperar a la API
echo "⏳ Esperando API..."
sleep 10

# Iniciar Streamlit
exec streamlit run dashboard.py \
    --server.port=8501 \
    --server.address=0.0.0.0 \
    --theme.base="light" \
    --browser.gatherUsageStats=false



    