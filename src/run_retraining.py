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
        if 'order_month' in new_data.columns:
            print(f"📅 Nuevos meses: {new_data['order_month'].unique()}")
        
        # Actualizar dataset
        updated_data = updater.update_dataset(new_data)
        
        if updated_data is None or updated_data.empty:
            print("❌ Error actualizando dataset")
            return
        
        # Guardar datos actualizados
        output_path = updater.save_updated_data(updated_data)
        print(f"✅ Dataset actualizado guardado: {output_path}")
        
        # 2. Reentrenar modelos
        print("\n🤖 PASO 2: REENTRENANDO MODELOS...")
        trainer = ModelTrainer()
        
        print(f"📊 Dataset actualizado: {updated_data.shape}")
        
        # Usar DataFrame directamente
        results = trainer.train_models_from_dataframe(updated_data)
        
        if results:
            print("✅ Modelos reentrenados exitosamente")
            
            # Mostrar comparación de métricas
            print("\n📈 COMPARACIÓN DE MÉTRICAS:")
            for model_name, metrics in trainer.metrics.items():
                print(
                    f"   {model_name.upper():<20} - "
                    f"MAE: {metrics['mae']:.2f}, "
                    f"RMSE: {metrics['rmse']:.2f}, "
                    f"R²: {metrics['r2']:.3f}"
                )
            
            # Guardar los modelos actualizados
            trainer.save_models()
            
            # Mostrar resumen
            print("\n📊 RESUMEN DEL REENTRENAMIENTO:")
            print(f"   📈 Datos totales: {len(updated_data)} filas")
            if trainer.metrics:
                best_model_name = min(trainer.metrics.items(), key=lambda x: x[1]['mae'])[0]
                print(f"   🏆 Mejor modelo: {best_model_name}")
                print(f"   🎯 Mejor R²: {trainer.metrics[best_model_name]['r2']:.3f}")
        else:
            print("❌ Error en el reentrenamiento")
    
    except Exception as e:
        print(f"❌ Error en el sistema de reentrenamiento: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
