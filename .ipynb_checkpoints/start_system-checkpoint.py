import subprocess
import time
import webbrowser

print("🚀 Iniciando sistema Olist Forecasting...")

# 1. Iniciar API
print("1. Iniciando API (puerto 8000)...")
api_process = subprocess.Popen([
    'python', 'api.py'
])

time.sleep(5)

# 2. Iniciar Dashboard
print("2. Iniciando Dashboard (puerto 8501)...")
dashboard_process = subprocess.Popen([
    'streamlit', 'run', 'dashboard.py', '--server.port', '8501'
])

time.sleep(5)

# 3. Iniciar MLflow (si está disponible)
print("3. Intentando iniciar MLflow (puerto 5000)...")
try:
    import mlflow
    mlflow_process = subprocess.Popen([
        'python', '-m', 'mlflow', 'server', 
        '--host', '127.0.0.1', 
        '--port', '5000',
        '--backend-store-uri', 'sqlite:///mlflow.db'
    ])
    print("   MLflow iniciado")
except:
    print("   MLflow no disponible, continuando sin él")

# 4. Abrir navegadores
print("4. Abriendo interfaces...")
time.sleep(3)

webbrowser.open("http://localhost:8501")
time.sleep(2)
webbrowser.open("http://localhost:8000/docs")

print("\n" + "="*50)
print("✅ SISTEMA INICIADO")
print("="*50)
print("📊 Dashboard: http://localhost:8501")
print("🔌 API Docs:  http://localhost:8000/docs")
print("🔬 MLflow:    http://localhost:5000 (si se inició)")
print("\nPresiona Ctrl+C para detener todo")
print("="*50)

try:
    # Mantener corriendo
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\n🛑 Deteniendo sistema...")
    api_process.terminate()
    dashboard_process.terminate()
    if 'mlflow_process' in locals():
        mlflow_process.terminate()


