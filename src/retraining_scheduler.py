# src/retraining_scheduler.py
import schedule
import time
import threading
from datetime import datetime
from model_trainer import ModelTrainer
from data_updater import DataUpdater

class RetrainingScheduler:
    def __init__(self):
        self.is_running = False
        self.thread = None
    
    def monthly_retraining_job(self):
        """Trabajo programado para reentrenamiento mensual"""
        print(f"\n🔄 INICIANDO REENTRENAMIENTO AUTOMÁTICO - {datetime.now()}")
        print("=" * 60)
        
        try:
            # 1. Actualizar datos con nuevo mes
            updater = DataUpdater()
            new_data = updater.simulate_new_monthly_data(months_to_add=1)
            
            if new_data is not None:
                updated_data = updater.update_dataset(new_data)
                updater.save_updated_data(updated_data)
                
                # 2. Reentrenar modelos
                trainer = ModelTrainer()
                trainer.full_training_pipeline()
                
                print("✅ REENTRENAMIENTO COMPLETADO EXITOSAMENTE!")
            else:
                print("❌ No se pudieron generar nuevos datos")
                
        except Exception as e:
            print(f"❌ Error en reentrenamiento automático: {e}")
    
    def start_scheduler(self, interval_days=30):
        """Inicia el programador de reentrenamiento"""
        print(f"⏰ Iniciando programador de reentrenamiento (cada {interval_days} días)...")
        
        # Programar ejecución periódica
        schedule.every(interval_days).days.do(self.monthly_retraining_job)
        
        # Ejecutar inmediatamente la primera vez
        self.monthly_retraining_job()
        
        self.is_running = True
        print("🎯 Programador iniciado. Presiona Ctrl+C para detener.")
        
        # Mantener el programa en ejecución
        while self.is_running:
            schedule.run_pending()
            time.sleep(3600)  # Revisar cada hora
    
    def start_in_background(self, interval_days=30):
        """Inicia el programador en segundo plano"""
        def run_scheduler():
            self.start_scheduler(interval_days)
        
        self.thread = threading.Thread(target=run_scheduler)
        self.thread.daemon = True
        self.thread.start()
        print("🎯 Programador iniciado en segundo plano")
    
    def stop_scheduler(self):
        """Detiene el programador"""
        self.is_running = False
        print("⏹️ Programador detenido")

# Uso rápido
if __name__ == "__main__":
    scheduler = RetrainingScheduler()
    
    # Para ejecución inmediata (sin scheduling)
    scheduler.monthly_retraining_job()
    
    # Para scheduling automático (descomentar para usar)
    # scheduler.start_scheduler(interval_days=30)  # Cada 30 días