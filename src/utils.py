# src/utils.py
import pandas as pd
import numpy as np
import logging
import os
from datetime import datetime

# CAMBIAR ESTE IMPORT - Quitar el punto
from config import Config

def setup_logging():
    """Configura logging para el proyecto"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)

def validate_dataframe(df, name=""):
    """Valida la calidad básica de un DataFrame"""
    logger = setup_logging()
    
    logger.info(f"Validando DataFrame {name}: {df.shape}")
    logger.info(f"  • Nulos: {df.isnull().sum().sum()}")
    logger.info(f"  • Duplicados: {df.duplicated().sum()}")
    logger.info(f"  • Columnas: {len(df.columns)}")
    
    return df

def check_data_files():
    """Verifica que todos los archivos de datos existan"""
    # EL IMPORT YA ESTÁ CORREGIDO ARRIBA
    
    missing_files = []
    available_files = []
    
    for dataset, filename in Config.FILES.items():
        file_path = Config.get_file_path(dataset)
        if file_path.exists():
            available_files.append(dataset)
        else:
            missing_files.append(dataset)
    
    print("📁 Verificación de archivos de datos:")
    print(f"   ✅ Disponibles: {available_files}")
    if missing_files:
        print(f"   ❌ Faltantes: {missing_files}")
        print(f"   💡 Coloca los archivos CSV en: {Config.DATA_PATH}")
    
    return available_files, missing_files

def save_dataset(df, filename):
    """Guarda el dataset en la carpeta processed"""
    output_path = Config.get_output_path(filename)
    df.to_csv(output_path, index=False)
    print(f"💾 Dataset guardado en: {output_path}")
    return output_path

def get_feature_summary(df, target_col='demand_next_month'):
    """Obtiene resumen de features"""
    non_feature_cols = ['order_month', 'product_category_name', 'date', 'month_year', target_col]
    feature_cols = [col for col in df.columns if col not in non_feature_cols]
    
    feature_categories = {
        'Temporales': len([f for f in feature_cols if any(x in f for x in ['year', 'month', 'quarter', 'is_', 'sin', 'cos'])]),
        'Demanda Histórica': len([f for f in feature_cols if any(x in f for x in ['lag', 'ma_', 'ema_', 'momentum'])]),
        'Crecimiento': len([f for f in feature_cols if 'growth' in f or 'pct_change' in f]),
        'Ventas/Precios': len([f for f in feature_cols if any(x in f for x in ['price', 'sales', 'revenue'])]),
        'Estadísticos': len([f for f in feature_cols if any(x in f for x in ['std', 'skew', 'percentile', 'cv', 'iqr'])]),
        'Ratios Negocio': len([f for f in feature_cols if 'ratio' in f or 'rate' in f or 'per_' in f]),
        'Reviews': len([f for f in feature_cols if 'review' in f]),
        'Entregas': len([f for f in feature_cols if 'delivery' in f]),
        'Pagos': len([f for f in feature_cols if 'payment' in f or 'pct_' in f or 'installment' in f]),
        'Categoría': len([f for f in feature_cols if 'category_' in f or 'z_score' in f])
    }
    
    return feature_cols, feature_categories

# Test de las utilidades
if __name__ == "__main__":
    print("✅ Utils importado correctamente")
    check_data_files()
















