# src/config.py
import os
from pathlib import Path

class Config:
    """Configuración centralizada del proyecto"""
    
    # Paths de datos
    BASE_DIR = Path(__file__).parent.parent
    DATA_PATH = BASE_DIR / 'data' / 'raw'
    OUTPUT_PATH = BASE_DIR / 'data' / 'processed'
    
    # Crear directorios si no existen
    DATA_PATH.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    
    # Archivos de datos
    FILES = {
        'orders': 'olist_orders_dataset.csv',
        'items': 'olist_order_items_dataset.csv', 
        'products': 'olist_products_dataset.csv',
        'reviews': 'olist_order_reviews_dataset.csv',
        'payments': 'olist_order_payments_dataset.csv',
        'customers': 'olist_customers_dataset.csv',
        'sellers': 'olist_sellers_dataset.csv'
    }
    
    # Configuración de features
    TEMPORAL_CONFIG = {
        'lags': [1, 2, 3, 6, 12],
        'moving_windows': [2, 3, 6, 12],
        'ema_alphas': [0.3, 0.5, 0.7]
    }
    
    # Columnas relevantes
    ORDERS_COLUMNS = [
        'order_id', 'customer_id', 'order_purchase_timestamp',
        'order_delivered_carrier_date', 'order_delivered_customer_date',
        'order_estimated_delivery_date', 'order_status'
    ]
    
    # Configuración de modelos
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    
    @classmethod
    def get_file_path(cls, dataset_name):
        """Obtiene path completo del archivo"""
        return cls.DATA_PATH / cls.FILES[dataset_name]
    
    @classmethod
    def get_output_path(cls, filename):
        """Obtiene path de salida"""
        return cls.OUTPUT_PATH / filename