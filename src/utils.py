# src/utils.py

import pandas as pd
import numpy as np
import logging
import os
from datetime import datetime

# 🔧 IMPORT CORREGIDO (sin punto, válido para ejecución directa y scheduler)
from config import Config


# ============================================================
# LOGGING
# ============================================================

def setup_logging():
    """Configura logging para todo el proyecto."""
    
    # Evita configurar logging múltiples veces
    if len(logging.getLogger().handlers) == 0:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    return logging.getLogger(__name__)


# ============================================================
# VALIDACIÓN DE DATASETS
# ============================================================

def validate_dataframe(df, name="DataFrame"):
    """Valida la calidad básica de un DataFrame con protección contra errores."""
    logger = setup_logging()

    if df is None:
        logger.error(f" {name} es None")
        return None
    
    if not isinstance(df, pd.DataFrame):
        logger.error(f" {name} no es un DataFrame")
        return None

    if df.empty:
        logger.warning(f" {name} está vacío")
        return df

    logger.info(f" Validando {name}: {df.shape}")
    logger.info(f"   • Columnas: {len(df.columns)}")
    logger.info(f"   • Nulos totales: {df.isnull().sum().sum()}")
    logger.info(f"   • Filas duplicadas: {df.duplicated().sum()}")

    return df


# ============================================================
# VERIFICACIÓN DE ARCHIVOS RAW
# ============================================================

def check_data_files():
    """Verifica que todos los archivos raw existan según Config."""
    
    missing_files = []
    available_files = []

    print("\n Verificando archivos RAW...")
    print("=" * 60)

    for dataset_name, filename in Config.FILES.items():
        file_path = Config.get_file_path(dataset_name)

        if file_path.exists():
            available_files.append(filename)
        else:
            missing_files.append(filename)

    print(f"    Disponibles: {available_files}")

    if missing_files:
        print(f"    Faltantes: {missing_files}")
        print(f"    Coloca los CSV faltantes en: {Config.DATA_PATH}")

    print("=" * 60)

    return available_files, missing_files


# ============================================================
# SAVE DATASET
# ============================================================

def save_dataset(df, filename):
    """Guarda un dataset dentro de /processed con validación."""
    if df is None or df.empty:
        print(" ERROR: Intentando guardar un DataFrame vacío o None")
        return None

    output_path = Config.get_output_path(filename)

    try:
        df.to_csv(output_path, index=False)
        print(f" Dataset guardado correctamente en: {output_path}")
        return output_path

    except Exception as e:
        print(f" ERROR guardando dataset {filename}: {e}")
        return None


# ============================================================
# FEATURE SUMMARY
# ============================================================

def get_feature_summary(df, target_col='demand_next_month'):
    """
    Obtiene un resumen estructurado de todas las features.
    Clasifica automáticamente los features según tu pipeline de forecasting.
    """
    
    if df is None or df.empty:
        return [], {}

    non_feature_cols = [
        'order_month', 'product_category_name', 'date', 
        'purchase_year_month', 'month_year', target_col
    ]

    feature_cols = [col for col in df.columns if col not in non_feature_cols]

    feature_categories = {
        'Temporales': len([f for f in feature_cols if any(x in f for x in [
            'year', 'month_num', 'quarter', 'is_', 'sin', 'cos'
        ])]),

        'Series Temporales': len([f for f in feature_cols if any(x in f for x in [
            'lag', 'ma_', 'ema_', 'rolling', 'momentum'
        ])]),

        'Crecimiento y Variación': len([f for f in feature_cols if any(x in f for x in [
            'growth', 'pct_change', 'acceleration'
        ])]),

        'Ventas / Precios': len([f for f in feature_cols if any(x in f for x in [
            'price', 'sales', 'revenue'
        ])]),

        'Estadísticos': len([f for f in feature_cols if any(x in f for x in [
            'std', 'skew', 'percentile', 'iqr', 'cv'
        ])]),

        'Ratios de Negocio': len([f for f in feature_cols if any(x in f for x in [
            'ratio', 'rate', 'per_', 'index'
        ])]),

        'Reviews': len([f for f in feature_cols if 'review' in f]),

        'Entregas': len([f for f in feature_cols if any(x in f for x in [
            'delivery', 'freight', 'on_time'
        ])]),

        'Pagos': len([f for f in feature_cols if any(x in f for x in [
            'payment', 'pct_', 'install'
        ])]),

        'Categoría': len([f for f in feature_cols if 'category_' in f or 'z_score' in f])
    }

    return feature_cols, feature_categories


# ============================================================
# TEST RÁPIDO
# ============================================================

if __name__ == "__main__":
    print(" Utils funcionando correctamente\n")
    check_data_files()












