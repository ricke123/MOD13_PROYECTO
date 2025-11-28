"""
Configuración global del proyecto - VERSIÓN MEJORADA
"""
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
    'orders':   DATA_RAW / 'olist_orders_dataset.csv',
    'items':    DATA_RAW / 'olist_order_items_dataset.csv',
    'products': DATA_RAW / 'olist_products_dataset.csv',
    'reviews':  DATA_RAW / 'olist_order_reviews_dataset.csv',
    'payments': DATA_RAW / 'olist_order_payments_dataset.csv',
    # Opcional: si usas clientes
    # 'customers': DATA_RAW / 'olist_customers_dataset.csv',
}

# Configuración de fechas por dataset
DATE_COLS = {
    'orders': [
        'order_purchase_timestamp',
        'order_delivered_carrier_date',
        'order_delivered_customer_date',
        'order_estimated_delivery_date'
    ]
}

# Configuración del modelo
MODEL_CONFIG = {
    'target_col': 'demand_next_month',
    # Nota: el split es temporal usando TIME_PERIODS; test_size se deja para compatibilidad futura
    'random_state': 42,
    'model_type': 'xgboost',  # opciones: 'xgboost' o 'random_forest'
    'feature_selection': False,        # poner True solo si implementas el paso
    'feature_importance_threshold': 0.0001
}

# Configuración de períodos temporales (split mensual)
TIME_PERIODS = {
    'train_start': '2016-09',
    'train_end':   '2018-04',
    'test_start':  '2018-05',
    'test_end':    '2018-07'
}

# Configuración de retraining
RETRAINING_CONFIG = {
    'retrain_interval_days':      30,   # Reentrenamiento incremental cada 30 días
    'full_retrain_interval_days': 90,   # Reentrenamiento completo cada 90 días
    'performance_threshold':      0.01, # Mejora mínima del 1% en R² para aceptar nuevo modelo
    'enable_auto_retrain':        True,
    'backup_old_models':          True,
    'max_history_size':           100,
    'run_on_startup':             False,
    'verbose_logging':            False
}













