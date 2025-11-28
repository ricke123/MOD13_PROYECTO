
"""
Módulo para programación de reentrenamiento automático - VERSIÓN SIN HILOS DAEMON
"""
import schedule
import time
import threading
from datetime import datetime, timedelta
import sys
import os
import signal

sys.path.append(os.path.dirname(__file__))

from run_retraining import run_incremental_retraining, run_full_retraining
from config import RETRAINING_CONFIG

class RetrainingScheduler:
    def __init__(self):
        self.config = RETRAINING_CONFIG
        self.is_running = False
        self.retraining_history = []
        self._stop_event = threading.Event()
    
    def schedule_retraining(self):
        """Programar reentrenamiento automático"""
        if self.config['enable_auto_retrain']:
            # Limpiar schedule existente
            schedule.clear()
            
            # Programar reentrenamiento incremental cada X días
            schedule.every(self.config['retrain_interval_days']).days.at("02:00").do(
                self._execute_incremental_retraining
            )
            
            # Programar reentrenamiento completo cada 3 meses
            schedule.every(90).days.at("03:00").do(self._execute_full_retraining)
            
            print(f"⏰ Reentrenamiento programado:")
            print(f"   • Incremental: cada {self.config['retrain_interval_days']} días a las 02:00")
            print(f"   • Completo: cada 90 días a las 03:00")
    
    def _execute_incremental_retraining(self):
        """Ejecutar reentrenamiento incremental"""
        print(f"\n🔄 Ejecutando reentrenamiento incremental - {datetime.now()}")
        print("=" * 60)
        
        try:
            success = run_incremental_retraining()
            result = {
                'timestamp': datetime.now(),
                'type': 'incremental',
                'success': success,
                'details': 'Reentrenamiento incremental automático'
            }
            self.retraining_history.append(result)
            
            if success:
                print("✅ Reentrenamiento incremental completado")
            else:
                print("❌ Reentrenamiento incremental falló")
                
        except Exception as e:
            print(f"❌ Error en reentrenamiento incremental: {e}")
            result = {
                'timestamp': datetime.now(),
                'type': 'incremental',
                'success': False,
                'error': str(e)
            }
            self.retraining_history.append(result)
    
    def _execute_full_retraining(self):
        """Ejecutar reentrenamiento completo"""
        print(f"\n🔄 Ejecutando reentrenamiento COMPLETO - {datetime.now()}")
        print("=" * 60)
        
        try:
            success = run_full_retraining()
            result = {
                'timestamp': datetime.now(),
                'type': 'full',
                'success': success,
                'details': 'Reentrenamiento completo automático'
            }
            self.retraining_history.append(result)
            
            if success:
                print("✅ Reentrenamiento completo completado")
            else:
                print("❌ Reentrenamiento completo falló")
                
        except Exception as e:
            print(f"❌ Error en reentrenamiento completo: {e}")
            result = {
                'timestamp': datetime.now(),
                'type': 'full',
                'success': False,
                'error': str(e)
            }
            self.retraining_history.append(result)
    
    def get_retraining_status(self):
        """Obtener estado del programador"""
        status = {
            'is_running': self.is_running,
            'next_retraining': None,
            'history_count': len(self.retraining_history),
            'successful_runs': len([r for r in self.retraining_history if r['success']]),
            'failed_runs': len([r for r in self.retraining_history if not r['success']]),
            'last_execution': self.retraining_history[-1] if self.retraining_history else None
        }
        
        # Calcular próximo reentrenamiento
        try:
            jobs = schedule.get_jobs()
            if jobs:
                next_run = min(jobs, key=lambda x: x.next_run).next_run
                status['next_retraining'] = next_run
        except:
            status['next_retraining'] = "No programado"
        
        return status
    
    def print_status(self):
        """Imprimir estado actual del programador"""
        status = self.get_retraining_status()
        
        print("\n📊 ESTADO DEL PROGRAMADOR:")
        print("=" * 50)
        print(f"🔄 Ejecutándose: {'✅ Sí' if status['is_running'] else '❌ No'}")
        print(f"📅 Próximo reentrenamiento: {status['next_retraining']}")
        print(f"📈 Historial: {status['successful_runs']} exitosos, {status['failed_runs']} fallidos")
        print(f"📋 Total ejecuciones: {status['history_count']}")
        
        # Mostrar última ejecución
        if status['last_execution']:
            last = status['last_execution']
            status_icon = "✅" if last['success'] else "❌"
            print(f"⏱️  Última ejecución: {last['timestamp'].strftime('%Y-%m-%d %H:%M')} - {last['type']} - {status_icon}")
        
        print("=" * 50)
        
        # Mostrar últimos 5 reentrenamientos
        if self.retraining_history:
            print("\n📋 ÚLTIMOS REENTRENAMIENTOS:")
            for i, run in enumerate(self.retraining_history[-5:], 1):
                status_icon = "✅" if run['success'] else "❌"
                error_info = f" - Error: {run['error']}" if 'error' in run and run['error'] else ""
                print(f"   {i}. {run['timestamp'].strftime('%Y-%m-%d %H:%M')} - {run['type']} - {status_icon}{error_info}")
    
    def run_scheduler(self):
        """Ejecutar el programador continuamente - VERSIÓN SIMPLIFICADA"""
        print("🚀 INICIANDO PROGRAMADOR DE REENTRENAMIENTO AUTOMÁTICO")
        print("=" * 60)
        
        self.is_running = True
        self.schedule_retraining()
        
        # Ejecutar inicialmente si está configurado
        initial_run = self.config.get('run_on_startup', False)
        if initial_run:
            print("🚀 Ejecutando reentrenamiento inicial...")
            self._execute_incremental_retraining()
        
        print("\n⏰ Programador iniciado. Presiona Ctrl+C para detener.")
        self.print_status()
        
        last_status_print = datetime.now()
        
        try:
            while self.is_running and not self._stop_event.is_set():
                schedule.run_pending()
                
                # Imprimir estado cada 30 minutos en lugar de cada hora
                current_time = datetime.now()
                if (current_time - last_status_print).total_seconds() >= 1800:  # 30 minutos
                    self.print_status()
                    last_status_print = current_time
                
                time.sleep(60)  # Revisar cada minuto
                    
        except KeyboardInterrupt:
            print("\n🛑 Deteniendo programador por Ctrl+C...")
            self.is_running = False
        except Exception as e:
            print(f"❌ Error inesperado en el programador: {e}")
            self.is_running = False
    
    def stop_scheduler(self):
        """Detener el programador"""
        self.is_running = False
        self._stop_event.set()
        schedule.clear()
        print("⏹️  Programador detenido")
    
    def force_retraining(self, retraining_type='incremental'):
        """Forzar reentrenamiento manualmente"""
        print(f"🔄 Forzando reentrenamiento {retraining_type}...")
        
        if retraining_type == 'incremental':
            self._execute_incremental_retraining()
        elif retraining_type == 'full':
            self._execute_full_retraining()
        else:
            print(f"❌ Tipo de reentrenamiento no válido: {retraining_type}")

def main():
    """Función principal del programador - VERSIÓN SIMPLIFICADA"""
    print("🚀 SISTEMA DE REENTRENAMIENTO AUTOMÁTICO - OLIST")
    print("=" * 60)
    
    scheduler = RetrainingScheduler()
    
    # Configuración simple sin hilos complejos
    print("💡 Opciones:")
    print("   1. Iniciar programador automático")
    print("   2. Ejecutar reentrenamiento incremental ahora")
    print("   3. Ejecutar reentrenamiento completo ahora")
    print("   4. Salir")
    
    try:
        while True:
            choice = input("\nSelecciona una opción (1-4): ").strip()
            
            if choice == '1':
                print("\n🚀 Iniciando programador automático...")
                print("💡 Presiona Ctrl+C para detener")
                scheduler.run_scheduler()
                break
            elif choice == '2':
                print("\n🔄 Ejecutando reentrenamiento incremental...")
                scheduler._execute_incremental_retraining()
            elif choice == '3':
                print("\n🔄 Ejecutando reentrenamiento completo...")
                scheduler._execute_full_retraining()
            elif choice == '4':
                print("👋 Saliendo...")
                break
            else:
                print("❌ Opción no válida. Por favor selecciona 1-4.")
                
    except KeyboardInterrupt:
        print("\n👋 Saliendo...")
        scheduler.stop_scheduler()
    except Exception as e:
        print(f"❌ Error: {e}")
        scheduler.stop_scheduler()

if __name__ == "__main__":
    main()

