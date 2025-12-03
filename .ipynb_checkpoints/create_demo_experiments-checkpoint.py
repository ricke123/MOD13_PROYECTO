import mlflow
import numpy as np
from datetime import datetime

mlflow.set_tracking_uri('http://localhost:5000')

# Crear varios experimentos
experiments = [
    ('Baseline-Models', 'Baseline'),
    ('XGBoost-Optimized', 'XGBoost'),
    ('Random-Forest', 'RF'),
    ('Neural-Network', 'NN'),
    ('Production-Best', 'Production')
]

for exp_name, run_prefix in experiments:
    mlflow.set_experiment(exp_name)
    
    for i in range(1, 4):  # 3 runs por experimento
        with mlflow.start_run(run_name=f'{run_prefix}_run_{i}'):
            # Parámetros
            mlflow.log_param('model', run_prefix.lower())
            mlflow.log_param('random_state', 42 + i)
            mlflow.log_param('version', f'1.0.{i}')
            
            # Métricas (simuladas)
            mlflow.log_metric('mae', 10 + np.random.rand() * 5)
            mlflow.log_metric('rmse', 15 + np.random.rand() * 8)
            mlflow.log_metric('r2', 0.8 + np.random.rand() * 0.15)
            mlflow.log_metric('training_time_sec', 30 + np.random.rand() * 60)
            
            # Métricas de negocio
            mlflow.log_metric('roi_percentage', 30 + np.random.rand() * 20)
            mlflow.log_metric('cost_reduction', 0.1 + np.random.rand() * 0.3)
            mlflow.log_metric('accuracy_business', 0.75 + np.random.rand() * 0.2)

print(f'✅ Creados {len(experiments)*3} experimentos demo!')
print('🔗 Ve a http://localhost:5000')