#!/usr/bin/env python3
"""
Script principal para ejecutar la actualización de datos y reentrenamiento de modelos
"""

import sys
import os
import pandas as pd

# Agregar el directorio src al path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_updater import DataUpdater
from model_trainer import ModelTrainer

def main():
    print("🚀 INICIANDO SISTEMA DE REENTRENAMIENTO")
    print("=" * 50)
    
    try:
        # 1. Actualizar datos
        print("📊 PASO 1: ACTUALIZANDO DATOS...")
        updater = DataUpdater()
        
        # Simular nuevos datos mensuales
        new_data = updater.simulate_new_monthly_data(months_to_add=1)
        
        if new_data is None or new_data.empty:
            print("❌ No se pudieron generar nuevos datos")
            return
        
        print(f"✅ Nuevos datos generados: {len(new_data)} filas")
        print(f"📅 Nuevos meses: {new_data['order_month'].unique()}")
        
        # Actualizar dataset
        updated_data = updater.update_dataset(new_data)
        
        if updated_data is not None and not updated_data.empty:
            # Guardar datos actualizados
            output_path = updater.save_updated_data(updated_data)
            print(f"✅ Dataset actualizado guardado: {output_path}")
        else:
            print("❌ Error actualizando dataset")
            return
        
        # 2. Reentrenar modelos
        print("\n🤖 PASO 2: REENTRENANDO MODELOS...")
        trainer = ModelTrainer()
        
        print(f"📊 Dataset actualizado: {updated_data.shape}")
        
        # Reentrenar modelos
        results = trainer.train_models(updated_data)
        
        if results:
            print("✅ Modelos reentrenados exitosamente")
            
            # Mostrar comparación de métricas
            print("\n📈 COMPARACIÓN DE MÉTRICAS:")
            for model_name, metrics in results.items():
                print(f"   {model_name:20} - MAE: {metrics['mae']:.2f}, RMSE: {metrics['rmse']:.2f}, R²: {metrics['r2']:.3f}")
        
        else:
            print("❌ Error en el reentrenamiento")
    
    except Exception as e:
        print(f"❌ Error en el sistema de reentrenamiento: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()