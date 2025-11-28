"""
ORQUESTADOR PRINCIPAL del pipeline - VERSIÓN CORREGIDA
"""
import sys
import os
sys.path.append(os.path.dirname(__file__))

from data_loader import DataLoader
from feature_engineer import FeatureEngineer
from model_trainer import ModelTrainer

def run_pipeline():
    """Ejecutar pipeline completo de ML"""
    print("🚀 INICIANDO PIPELINE DE MACHINE LEARNING")
    print("=" * 60)
    
    try:
        # 1. CARGA Y LIMPIEZA DE DATOS
        print("\n1️⃣  ETAPA: CARGA Y LIMPIEZA DE DATOS")
        loader = DataLoader()
        
        # Intentar cargar datos procesados primero
        processed_data = loader.load_processed_data()
        
        if processed_data is None:
            # Cargar y procesar datos desde raw
            orders, items, products, reviews, payments = loader.load_raw_data()
            df_clean = loader.clean_data(orders, items, products, reviews, payments)
            loader.save_processed_data(df_clean)
        else:
            df_clean = processed_data
        
        # 2. INGENIERÍA DE FEATURES
        print("\n2️⃣  ETAPA: INGENIERÍA DE FEATURES")
        engineer = FeatureEngineer()
        
        # Crear variable target y features
        monthly_demand = engineer.create_target_variable(df_clean)
        monthly_demand_with_features = engineer.create_advanced_features(monthly_demand)
        features_df = engineer.prepare_model_features(monthly_demand_with_features)
        
        print(f"📊 Dataset final para modelo: {features_df.shape}")
        
        # 3. ENTRENAMIENTO DEL MODELO
        print("\n3️⃣  ETAPA: ENTRENAMIENTO DEL MODELO")
        trainer = ModelTrainer()
        model, train_score, test_score, model_path = trainer.train_best_model(features_df)
        
        # 4. RESUMEN FINAL
        print("\n🎉 PIPELINE COMPLETADO EXITOSAMENTE")
        print("=" * 60)
        print(f"📈 Modelo final: {trainer.config['model_type']}")
        print(f"📊 R2 Score (test): {test_score['R²']:.4f}")  # CORREGIDO: 'R²' en lugar de 'r2'
        print(f"📊 MAE (test): {test_score['MAE']:.2f}")
        print(f"📊 RMSE (test): {test_score['RMSE']:.2f}")
        print(f"📊 MAPE (test): {test_score['MAPE']:.2f}%")
        print(f"📁 Modelo guardado en: {model_path}")
        print("=" * 60)
        
        return model, train_score, test_score
        
    except Exception as e:
        print(f"❌ Error en el pipeline: {e}")
        raise

if __name__ == "__main__":
    run_pipeline()


