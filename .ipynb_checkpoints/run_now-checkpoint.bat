@echo off
echo ============================================
echo OLIST FORECASTING - INICIO RAPIDO
echo ============================================

cd /d "C:\Users\REECK\MOD13_PROYECTO"

echo.
echo 1. Iniciando MLflow (puerto 5001)...
start "MLflow" cmd /k "call conda activate mlflow-env && mlflow ui --host 127.0.0.1 --port 5001"

timeout /t 5 >nul

echo.
echo 2. Iniciando API FastAPI (puerto 8000)...
start "API" cmd /k "call conda activate mlflow-env && python api.py"

timeout /t 5 >nul

echo.
echo 3. Iniciando Dashboard Streamlit (puerto 8501)...
start "Dashboard" cmd /k "call conda activate mlflow-env && streamlit run dashboard.py --server.port 8501"

echo.
echo ============================================
echo SISTEMA INICIADO - LISTO PARA PRUEBAS
echo.
echo Acceso:
echo   Dashboard:  http://localhost:8501
echo   API Docs:   http://localhost:8000/docs
echo   MLflow:     http://127.0.0.1:5001
echo   Health:     http://localhost:8000/health
echo ============================================

timeout /t 3 >nul

start http://localhost:8501
start http://localhost:8000/docs
start http://127.0.0.1:5001

pause