"""

"""
from pathlib import Path

# ============================================================================
# 1. RUTAS DEL PROYECTO
# ============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "data" / "model"

# Crear directorios si no existen
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# 2. ARCHIVOS DE DATOS
# ============================================================================
DATA_FILES = {
    'orders':   DATA_RAW / 'olist_orders_dataset.csv',
    'items':    DATA_RAW / 'olist_order_items_dataset.csv',
    'products': DATA_RAW / 'olist_products_dataset.csv',
    'reviews':  DATA_RAW / 'olist_order_reviews_dataset.csv',
    'payments': DATA_RAW / 'olist_order_payments_dataset.csv',
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

# ============================================================================
# 3. CONFIGURACIÓN GENERAL DEL MODELO
# ============================================================================
MODEL_CONFIG = {
    'target_col': 'demand_next_month',
    'random_state': 42,
    'model_type': 'random_forest',  # 'xgboost' o 'random_forest' - CAMBIA AQUÍ
    'feature_selection': False,
    'feature_importance_threshold': 0.0001,
    'enable_feature_correlation': True,
    'correlation_threshold': 0.90,
    'max_features': 99
}

# ============================================================================
# 4. HIPERPARÁMETROS OPTIMIZADOS (DEL EDA)
# ============================================================================

# XGBOOST - PARÁMETROS OPTIMIZADOS DEL EDA
XGBOOST_PARAMS = {
    # Hiperparámetros optimizados
    'colsample_bylevel': 0.6072,
    'colsample_bytree': 0.7976,
    'gamma': 0.1788,
    'learning_rate': 0.0348,
    'max_depth': 10,
    'min_child_weight': 7,
    'n_estimators': 294,
    'reg_alpha': 0.0882,
    'reg_lambda': 0.0257,
    'subsample': 0.6375,
    
    # Parámetros fijos
    'random_state': 42,
    'n_jobs': -1,
    'tree_method': 'hist',
    'early_stopping_rounds': 50,
    'eval_metric': 'mae'
}

# RANDOM FOREST - PARÁMETROS OPTIMIZADOS DEL EDA
# AJUSTA estos valores con los resultados EXACTOS de tu EDA
RANDOM_FOREST_PARAMS = {
    # Hiperparámetros optimizados
    'n_estimators': 300,           # Del rango [200, 300, 400, 500]
    'max_depth': 20,               # Del rango [5, 10, 15, 20, None]
    'min_samples_split': 2,        # Del rango [2, 5, 10]
    'min_samples_leaf': 1,         # Del rango [1, 2, 4]
    'max_features': 'sqrt',        # Del rango ['sqrt', 'log2', 0.3, 0.5, 0.7]
    
    # Parámetros fijos
    'random_state': 42,
    'n_jobs': -1,
    'bootstrap': True
}

# ============================================================================
# 5. CONFIGURACIÓN DE RANDOMIZEDSEARCHCV
# ============================================================================
RANDOM_SEARCH_CONFIG = {
    # Activar/desactivar búsqueda automática de hiperparámetros
    'enable_random_search': False,  # True para desarrollo, False para producción
    
    # Configuración general
    'n_iter': 30,
    'cv_splits': 3,
    'scoring': 'neg_mean_absolute_error',
    'verbose': 1,
    'n_jobs': -1,
    
    # Rangos de búsqueda para Random Forest
    'rf_param_distributions': {
        'n_estimators': [200, 300, 400, 500],
        'max_depth': [5, 10, 15, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', 0.3, 0.5, 0.7]
    }
}

# ============================================================================
# 6. CONFIGURACIÓN DE PERÍODOS TEMPORALES
# ============================================================================
TIME_PERIODS = {
    'train_start': '2016-09',
    'train_end':   '2018-04',
    'test_start':  '2018-05',
    'test_end':    '2018-07'
}

# ============================================================================
# 7. CONFIGURACIÓN DE REENTRENAMIENTO
# ============================================================================
RETRAINING_CONFIG = {
    'retrain_interval_days': 30, ##0.0000001,
    'full_retrain_interval_days': 90,
    'performance_threshold': 0.01,
    'enable_auto_retrain': True,
    'backup_old_models': True,
    'max_history_size': 100,
    'run_on_startup': False,
    'verbose_logging': False
}

# ============================================================================
# 8. CONFIGURACIÓN DEL BENCHMARK
# ============================================================================
BENCHMARK_CONFIG = {
    'test_size': 0.2,
    'random_state': 42,
    'models_to_compare': [
        'naive',
        'random_forest',
        'xgboost_base',
        'xgboost_tuned',
        'catboost_tuned'
    ],
    'save_results': True,
    'results_file': DATA_PROCESSED / 'benchmark_results.csv'
}

# ============================================================================
# 9. FUNCIONES DE AYUDA
# ============================================================================
def get_model_params(model_type: str = None):
    """
    Obtener parámetros del modelo según el tipo especificado.
    """
    if model_type is None:
        model_type = MODEL_CONFIG['model_type']
    
    if model_type == 'xgboost':
        params = XGBOOST_PARAMS.copy()
    elif model_type == 'random_forest':
        params = RANDOM_FOREST_PARAMS.copy()
    else:
        raise ValueError(f"Tipo de modelo no soportado: {model_type}")
    
    # Eliminar parámetros que no son del constructor del modelo
    model_params = params.copy()
    # Eliminar parámetros específicos de entrenamiento si existen
    model_params.pop('early_stopping_rounds', None)
    model_params.pop('eval_metric', None)
    
    return model_params

def update_model_params(model_type: str, new_params: dict):
    """
    Actualizar parámetros de un modelo.
    """
    if model_type == 'xgboost':
        XGBOOST_PARAMS.update(new_params)
        print(f"✅ Parámetros de XGBoost actualizados: {list(new_params.keys())}")
    elif model_type == 'random_forest':
        RANDOM_FOREST_PARAMS.update(new_params)
        print(f"✅ Parámetros de Random Forest actualizados: {list(new_params.keys())}")
    else:
        raise ValueError(f"Tipo de modelo no soportado: {model_type}")

def print_config_summary():
    """Imprimir resumen de la configuración actual"""
    print("=" * 60)
    print("CONFIGURACIÓN ACTUAL DEL PROYECTO")
    print("=" * 60)
    print(f"📊 Modelo activo: {MODEL_CONFIG['model_type'].upper()}")
    print(f"🎯 Target: {MODEL_CONFIG['target_col']}")
    print(f"📅 Train: {TIME_PERIODS['train_start']} → {TIME_PERIODS['train_end']}")
    print(f"📅 Test:  {TIME_PERIODS['test_start']} → {TIME_PERIODS['test_end']}")
    print(f"🔍 RandomizedSearch: {'✅ ACTIVADO' if RANDOM_SEARCH_CONFIG['enable_random_search'] else '❌ DESACTIVADO'}")
    print(f"🏁 Benchmark models: {len(BENCHMARK_CONFIG['models_to_compare'])} modelos")
    print("=" * 60)