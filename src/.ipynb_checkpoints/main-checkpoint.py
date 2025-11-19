# src/main.py
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Cambiar imports relativos por absolutos
from data_loader import DataLoader
from feature_engineer import FeatureEngineer
from preprocessor import DataPreprocessor
from config import Config
from utils import check_data_files, save_dataset, get_feature_summary

def main():
    """Pipeline principal de ejecución"""
    print("🚀 INICIANDO PIPELINE DE PREDICCIÓN DE DEMANDA")
    print("=" * 60)
    
    # Verificar archivos de datos
    available_files, missing_files = check_data_files()
    
    print(f"📁 Archivos disponibles: {available_files}")
    print(f"📁 Archivos faltantes: {missing_files}")
    
    if not available_files:
        print("❌ No se encontraron archivos de datos")
        print(f"💡 Coloca los archivos CSV en: {Config.DATA_PATH}")
        return None
    
    # 1. Carga y limpieza de datos
    print("\n📥 ETAPA 1: CARGA Y LIMPIEZA")
    loader = DataLoader()
    data = loader.load_all_data()
    
    if not data or data.get('orders') is None or data['orders'].empty:
        print("❌ No se pudo cargar el dataset orders. Verifica los archivos.")
        return None
        
    loader.clean_all_data()
    
    # 2. Ingeniería de features
    print("\n🔧 ETAPA 2: INGENIERÍA DE FEATURES")
    engineer = FeatureEngineer(loader.data)
    engineer.create_base_features()
    engineer.create_payment_features()
    engineer.create_review_features()
    engineer.create_dataset_principal()
    agg_data = engineer.aggregate_by_month_category()
    
    # 3. Preprocesamiento
    print("\n⚙️ ETAPA 3: PREPROCESAMIENTO")
    preprocessor = DataPreprocessor()
    
    # Features temporales
    agg_data = preprocessor.create_temporal_features(agg_data)
    
    # Features de series temporales
    master = preprocessor.apply_temporal_features(agg_data)
    
    # Ratios de negocio
    master = preprocessor.create_business_ratios(master)
    
    # Features YoY
    master = master.groupby('product_category_name', group_keys=False).apply(preprocessor.add_yoy_features)
    
    # Features estadísticos
    master = master.groupby('product_category_name', group_keys=False).apply(preprocessor.add_statistical_features)
    
    # Features de categoría
    master = preprocessor.add_category_features(master)
    
    # Limpieza final
    master_final = preprocessor.clean_final_dataset(master)
    
    # 4. Verificación final
    print("\n✅ ETAPA 4: VERIFICACIÓN FINAL")
    
    target_col = 'demand_next_month'
    feature_cols, feature_categories = get_feature_summary(master_final, target_col)
    
    print(f"🎯 RESUMEN FINAL:")
    print(f"   • Filas: {master_final.shape[0]}")
    print(f"   • Columnas totales: {master_final.shape[1]}")
    print(f"   • Features: {len(feature_cols)}")
    print(f"   • Target: {target_col}")
    
    print(f"\n📊 DISTRIBUCIÓN DE FEATURES:")
    for category, count in feature_categories.items():
        print(f"   {category}: {count} features")
    
    total_features = sum(feature_categories.values())
    print(f"\n🎯 TOTAL FEATURES: {total_features}")
    
    # 5. Guardar resultados
    print("\n💾 GUARDANDO RESULTADOS...")
    output_path = save_dataset(master_final, 'TABLA_FINAL_MODULAR.csv')
    
    print("\n🎉 PIPELINE COMPLETADO EXITOSAMENTE!")
    return master_final

if __name__ == "__main__":
    final_data = main()





