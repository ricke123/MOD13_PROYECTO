
"""
Módulo para reentrenamiento - CORREGIDO CON train_best_model
"""
import argparse
import sys
import pandas as pd
from data_updater import ejecutar_actualizacion_mensual
from data_loader import DataLoader
from feature_engineer import FeatureEngineer
from model_trainer import ModelTrainer

def run_incremental_retraining():
    """Ejecutar reentrenamiento incremental con datos actualizados"""
    print("🎯 INICIANDO REENTRENAMIENTO INCREMENTAL")
    
    # 1. ACTUALIZAR DATOS
    print("\n" + "="*50)
    print("📥 PASO 1: ACTUALIZACIÓN DE DATOS")
    print("="*50)
    
    if not ejecutar_actualizacion_mensual():
        print("❌ No se pudieron actualizar los datos. Abortando...")
        return False
    
    # 2. CARGAR DATOS ACTUALIZADOS
    print("\n" + "="*50)
    print("📊 PASO 2: CARGA DE DATOS ACTUALIZADOS")
    print("="*50)
    
    data_loader = DataLoader()
    df = data_loader.load_processed_data()
    
    if df is None:
        print("❌ No se pudieron cargar los datos procesados")
        return False
    
    # 3. FEATURE ENGINEERING COMPLETO
    print("\n" + "="*50)
    print("🔧 PASO 3: FEATURE ENGINEERING")
    print("="*50)
    
    feature_engineer = FeatureEngineer()
    
    # Crear target y features (se regeneran TODOS los 157 features)
    monthly_data = feature_engineer.create_target_variable(df)
    features_data = feature_engineer.create_advanced_features(monthly_data)
    
    # Preparar features para el modelo
    final_features = feature_engineer.prepare_model_features(features_data)
    
    print(f"✅ Features preparados: {final_features.shape}")
    
    # 4. REENTRENAR MODELO
    print("\n" + "="*50)
    print("🧠 PASO 4: REENTRENAMIENTO DEL MODELO")
    print("="*50)
    
    model_trainer = ModelTrainer()
    
    # CORRECCIÓN: Usar train_best_model en lugar de train_model
    try:
        model, train_metrics, test_metrics, model_path = model_trainer.train_best_model(final_features)
        success = model is not None
        
        if success:
            print("✅ Reentrenamiento incremental completado exitosamente")
        else:
            print("❌ Reentrenamiento incremental falló")
        
        return success
        
    except Exception as e:
        print(f"❌ Error en reentrenamiento incremental: {e}")
        return False

def run_full_retraining():
    """Reentrenamiento completo con todos los datos"""
    print("🎯 INICIANDO REENTRENAMIENTO COMPLETO")
    
    # 1. CARGAR Y PROCESAR TODOS LOS DATOS DESDE CERO
    print("\n" + "="*50)
    print("📊 PASO 1: CARGA Y PROCESAMIENTO COMPLETO")
    print("="*50)
    
    data_loader = DataLoader()
    
    # Cargar datos raw y procesar desde cero
    orders, items, products, reviews, payments = data_loader.load_raw_data()
    df = data_loader.clean_data(orders, items, products, reviews, payments)
    data_loader.save_processed_data(df)
    
    # 2. FEATURE ENGINEERING COMPLETO
    print("\n" + "="*50)
    print("🔧 PASO 2: FEATURE ENGINEERING")
    print("="*50)
    
    feature_engineer = FeatureEngineer()
    monthly_data = feature_engineer.create_target_variable(df)
    features_data = feature_engineer.create_advanced_features(monthly_data)
    final_features = feature_engineer.prepare_model_features(features_data)
    
    print(f"✅ Features preparados: {final_features.shape}")
    
    # 3. REENTRENAR MODELO
    print("\n" + "="*50)
    print("🧠 PASO 3: REENTRENAMIENTO DEL MODELO")
    print("="*50)
    
    model_trainer = ModelTrainer()
    
    # CORRECCIÓN: Usar train_best_model en lugar de train_model
    try:
        model, train_metrics, test_metrics, model_path = model_trainer.train_best_model(final_features)
        success = model is not None
        
        if success:
            print("✅ Reentrenamiento completo exitoso")
        else:
            print("❌ Reentrenamiento completo falló")
        
        return success
        
    except Exception as e:
        print(f"❌ Error en reentrenamiento completo: {e}")
        return False

def main():
    """Función principal con manejo de argumentos"""
    parser = argparse.ArgumentParser(description='Sistema de Reentrenamiento de Modelos')
    parser.add_argument('--incremental', action='store_true', 
                       help='Reentrenamiento incremental con datos nuevos')
    parser.add_argument('--full', action='store_true', 
                       help='Reentrenamiento completo con todos los datos')
    
    args = parser.parse_args()
    
    print("🚀 SISTEMA DE REENTRENAMIENTO - OLIST DEMAND FORECASTING")
    print("=" * 60)
    
    # Verificar que se proporcione algún argumento
    if not args.incremental and not args.full:
        print("❌ Error: Debes especificar el tipo de reentrenamiento")
        print("💡 Uso:")
        print("   python src/run_retraining.py --incremental    # Solo datos nuevos")
        print("   python src/run_retraining.py --full           # Todos los datos")
        sys.exit(1)
    
    # Ejecutar según el argumento
    if args.incremental:
        print("🔄 MODO: Reentrenamiento Incremental")
        success = run_incremental_retraining()
    elif args.full:
        print("🔄 MODO: Reentrenamiento Completo")
        success = run_full_retraining()
    
    # Resultado final
    print("\n" + "="*60)
    if success:
        print("🎉 ¡PROCESO TERMINADO EXITOSAMENTE!")
    else:
        print("💥 PROCESO TERMINADO CON ERRORES")
    print("="*60)
    
    return success

# Las funciones originales para compatibilidad con el scheduler
if __name__ == "__main__":
    main()


