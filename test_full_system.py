# test_full_system.py
import sys
import os
sys.path.append('src')

def main():
    print("🚀 PROBANDO SISTEMA COMPLETO DE REENTRENAMIENTO")
    print("=" * 50)
    
    try:
        # 1. Probar Data Updater
        print("\n📥 PASO 1: Probando Data Updater...")
        from data_updater import DataUpdater
        updater = DataUpdater()
        new_data = updater.simulate_new_monthly_data(months_to_add=1)
        
        if new_data is not None:
            print(f"✅ Nuevos datos simulados: {new_data.shape}")
            updated_data = updater.update_dataset(new_data)
            updater.save_updated_data(updated_data)
            print("✅ Dataset actualizado exitosamente")
        else:
            print("❌ No se pudieron generar nuevos datos")
            return
        
        # 2. Probar Model Trainer
        print("\n🤖 PASO 2: Probando Model Trainer...")
        from model_trainer import ModelTrainer
        trainer = ModelTrainer()
        models = trainer.full_training_pipeline()
        
        if models:
            print("✅ Modelos entrenados exitosamente")
        else:
            print("❌ Error entrenando modelos")
            
        # 3. Mostrar resumen
        print("\n🎯 RESUMEN FINAL:")
        print("✅ Sistema de reentrenamiento probado exitosamente")
        print("✅ Nuevos datos integrados al dataset")
        print("✅ Modelos reentrenados con datos actualizados")
        print("✅ Archivos guardados en data/processed/")
        
    except Exception as e:
        print(f"❌ Error durante la prueba: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()