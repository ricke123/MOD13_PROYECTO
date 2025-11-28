
"""
Configuración global del proyecto - VERSIÓN MEJORADA
"""
import pandas as pd
from pathlib import Path

# Rutas del proyecto
PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed" 
MODEL_DIR = PROJECT_ROOT / "data" / "model"

# Crear directorios si no existen
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Archivos de datos
DATA_FILES = {
    'orders': DATA_RAW / 'olist_orders_dataset.csv',
    'items': DATA_RAW / 'olist_order_items_dataset.csv', 
    'products': DATA_RAW / 'olist_products_dataset.csv',
    'reviews': DATA_RAW / 'olist_order_reviews_dataset.csv',
    'payments': DATA_RAW / 'olist_order_payments_dataset.csv'
}

# Configuración de fechas
DATE_COLS = {
    'orders': ['order_purchase_timestamp', 'order_delivered_carrier_date',
               'order_delivered_customer_date', 'order_estimated_delivery_date']
}

# Configuración del modelo MEJORADA
MODEL_CONFIG = {
    'target_col': 'demand_next_month',
    'test_size': 0.2,
    'random_state': 42,
    'model_type': 'xgboost',  # xgboost o random_forest
    'feature_selection': True,
    'feature_importance_threshold': 0.0001
}

# Configuración de períodos temporales
TIME_PERIODS = {
    'train_start': '2016-09',
    'train_end': '2018-04', 
    'test_start': '2018-05',
    'test_end': '2018-07'
}

# Configuración de retraining mejorada
RETRAINING_CONFIG = {
    'retrain_interval_days': 30,           # Reentrenamiento incremental cada 30 días
    'full_retrain_interval_days': 90,      # Reentrenamiento completo cada 90 días
    'performance_threshold': 0.01,         # Mejora mínima del 1% en R² para mantener nuevo modelo
    'enable_auto_retrain': True,           # Habilitar reentrenamiento automático
    'backup_old_models': True,             # Hacer backup de modelos antiguos
    'max_history_size': 100,               # Máximo historial de reentrenamientos
    'run_on_startup': False,                # Ejecutar al iniciar el scheduler
    'verbose_logging': False               # Logging detallado
}













