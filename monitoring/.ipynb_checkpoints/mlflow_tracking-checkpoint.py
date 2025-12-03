# monitoring/mlflow_tracking.py - VERSIÓN SIMPLIFICADA PARA WINDOWS
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
from datetime import datetime
import os
import json

class MLflowTracker:
    def __init__(self, experiment_name="olist-demand-forecasting"):
        """Inicializa el tracker de MLflow - versión simplificada"""
        self.experiment_name = experiment_name
        self.setup_mlflow()
    
    def setup_mlflow(self):
        """Configura MLflow local"""
        try:
            # Configuración ya se hizo en api.py
            print(f"✅ MLflow listo para: {self.experiment_name}")
        except Exception as e:
            print(f"⚠️ MLflow setup skipped: {e}")
    
    def start_run(self, run_name=None):
        """Inicia un nuevo experimento"""
        if run_name is None:
            run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        return mlflow.start_run(run_name=run_name)
    
    def log_params(self, params):
        """Loggea parámetros"""
        mlflow.log_params(params)
    
    def log_param(self, key, value):
        """Loggea un parámetro"""
        mlflow.log_param(key, value)
    
    def log_metrics(self, metrics):
        """Loggea múltiples métricas"""
        mlflow.log_metrics(metrics)
    
    def log_metric(self, key, value):
        """Loggea una métrica"""
        mlflow.log_metric(key, value)
    
    def set_tag(self, key, value):
        """Establece un tag"""
        mlflow.set_tag(key, value)
    
    def set_tags(self, tags):
        """Establece múltiples tags"""
        mlflow.set_tags(tags)