
import schedule
import time
import threading
from datetime import datetime
import sys
import os

# ============================================================
# CORRECCIÓN DE IMPORTS
# ============================================================

# Agregar el directorio padre al path para imports absolutos
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from run_retraining import run_incremental_retraining, run_full_retraining
    from config import RETRAINING_CONFIG
    print("✅ Todos los módulos importados correctamente")
except ImportError as e:
    print(f"❌ Error importando módulos: {e}")
    print("💡 Asegúrate de ejecutar desde la raíz del proyecto: python src/retraining_scheduler.py")
    sys.exit(1)


class RetrainingScheduler:
    def __init__(self):
        self.config = RETRAINING_CONFIG
        self.is_running = False
        self.retraining_history = []
        self._stop_event = threading.Event()

    # ------------------------------------------------------------------
    # PROGRAMACIÓN DE TAREAS
    # ------------------------------------------------------------------
    def schedule_retraining(self):
        """Programar reentrenamiento automático usando schedule."""
        if not self.config.get('enable_auto_retrain', False):
            print("⚠️  El reentrenamiento automático está deshabilitado en RETRAINING_CONFIG.")
            schedule.clear()
            return

        # Limpiar schedule existente
        schedule.clear()

        # Intervalos desde config
        retrain_interval_days = self.config.get('retrain_interval_days', 30)
        full_retrain_interval_days = self.config.get('full_retrain_interval_days', 90)

        # Reentrenamiento incremental
        schedule.every(retrain_interval_days).days.at("02:00").do(
            self._execute_incremental_retraining
        )

        # Reentrenamiento completo
        schedule.every(full_retrain_interval_days).days.at("03:00").do(
            self._execute_full_retraining
        )

        print("⏰ Reentrenamientos programados:")
        print(f"   • Incremental: cada {retrain_interval_days} días a las 02:00")
        print(f"   • Completo:   cada {full_retrain_interval_days} días a las 03:00")

    # ------------------------------------------------------------------
    # EJECUCIÓN DE REENTRENAMIENTOS
    # ------------------------------------------------------------------
    def _log_run(self, run_type: str, success: bool, error: str | None = None):
        """Registrar una ejecución en el historial."""
        result = {
            'timestamp': datetime.now(),
            'type': run_type,
            'success': success,
        }
        if error:
            result['error'] = error
        self.retraining_history.append(result)

    def _execute_incremental_retraining(self):
        """Ejecutar reentrenamiento incremental."""
        print(f"\n🔄 Ejecutando reentrenamiento INCREMENTAL - {datetime.now()}")
        print("=" * 60)

        try:
            success = run_incremental_retraining()
            self._log_run('incremental', success)

            if success:
                print("✅ Reentrenamiento incremental completado")
            else:
                print("❌ Reentrenamiento incremental falló (run_incremental_retraining devolvió False)")

        except Exception as e:
            print(f"❌ Error en reentrenamiento incremental: {e}")
            self._log_run('incremental', False, str(e))

    def _execute_full_retraining(self):
        """Ejecutar reentrenamiento completo."""
        print(f"\n🔄 Ejecutando reentrenamiento COMPLETO - {datetime.now()}")
        print("=" * 60)

        try:
            success = run_full_retraining()
            self._log_run('full', success)

            if success:
                print("✅ Reentrenamiento completo completado")
            else:
                print("❌ Reentrenamiento completo falló (run_full_retraining devolvió False)")

        except Exception as e:
            print(f"❌ Error en reentrenamiento completo: {e}")
            self._log_run('full', False, str(e))

    # ------------------------------------------------------------------
    # STATUS Y LOG
    # ------------------------------------------------------------------
    def get_retraining_status(self):
        """Obtener estado del programador."""
        successful_runs = len([r for r in self.retraining_history if r.get('success')])
        failed_runs = len([r for r in self.retraining_history if not r.get('success')])

        status = {
            'is_running': self.is_running,
            'next_retraining': None,
            'history_count': len(self.retraining_history),
            'successful_runs': successful_runs,
            'failed_runs': failed_runs,
            'last_execution': self.retraining_history[-1] if self.retraining_history else None
        }

        # Próxima ejecución
        try:
            jobs = schedule.get_jobs()
            if jobs:
                next_run = min(jobs, key=lambda x: x.next_run).next_run
                status['next_retraining'] = next_run
            else:
                status['next_retraining'] = "No hay jobs programados"
        except Exception:
            status['next_retraining'] = "Error al calcular próxima ejecución"

        return status

    def print_status(self):
        """Imprimir estado actual del programador."""
        status = self.get_retraining_status()

        print("\n📊 ESTADO DEL PROGRAMADOR:")
        print("=" * 50)
        print(f"🔄 Ejecutándose: {'✅ Sí' if status['is_running'] else '❌ No'}")
        print(f"📅 Próximo reentrenamiento: {status['next_retraining']}")
        print(f"📈 Historial: {status['successful_runs']} exitosos, {status['failed_runs']} fallidos")
        print(f"📋 Total ejecuciones: {status['history_count']}")

        # Última ejecución
        last = status['last_execution']
        if last:
            icon = "✅" if last.get('success') else "❌"
            print(f"⏱️  Última ejecución: {last['timestamp'].strftime('%Y-%m-%d %H:%M')} "
                  f"- {last['type']} - {icon}")
            if 'error' in last and last['error']:
                print(f"    Error: {last['error']}")

        print("=" * 50)

        # Últimos 5 reentrenamientos
        if self.retraining_history:
            print("\n📋 ÚLTIMOS REENTRENAMIENTOS:")
            for i, run in enumerate(self.retraining_history[-5:], 1):
                icon = "✅" if run.get('success') else "❌"
                error_info = f" - Error: {run.get('error')}" if run.get('error') else ""
                print(f"   {i}. {run['timestamp'].strftime('%Y-%m-%d %H:%M')} "
                      f"- {run['type']} - {icon}{error_info}")

    # ------------------------------------------------------------------
    # LOOP PRINCIPAL DEL SCHEDULER
    # ------------------------------------------------------------------
    def run_scheduler(self):
        """Ejecutar el programador continuamente (sin hilos daemon)."""
        print("🚀 INICIANDO PROGRAMADOR DE REENTRENAMIENTO AUTOMÁTICO")
        print("=" * 60)

        self.is_running = True
        self._stop_event.clear()
        self.schedule_retraining()

        # Ejecutar inicialmente si está configurado
        if self.config.get('run_on_startup', False):
            print("🚀 Ejecutando reentrenamiento inicial (run_on_startup=True)...")
            self._execute_incremental_retraining()

        print("\n⏰ Programador iniciado. Presiona Ctrl+C para detener.")
        self.print_status()

        last_status_print = datetime.now()

        try:
            while self.is_running and not self._stop_event.is_set():
                schedule.run_pending()

                # Imprimir estado cada 30 minutos
                current_time = datetime.now()
                if (current_time - last_status_print).total_seconds() >= 1800:
                    self.print_status()
                    last_status_print = current_time

                time.sleep(60)  # Revisar cada minuto

        except KeyboardInterrupt:
            print("\n🛑 Deteniendo programador por Ctrl+C...")
            self.stop_scheduler()
        except Exception as e:
            print(f"❌ Error inesperado en el programador: {e}")
            self.stop_scheduler()

    def stop_scheduler(self):
        """Detener el programador."""
        self.is_running = False
        self._stop_event.set()
        schedule.clear()
        print("⏹️  Programador detenido")

    # ------------------------------------------------------------------
    # EJECUCIONES MANUALES
    # ------------------------------------------------------------------
    def force_retraining(self, retraining_type: str = 'incremental'):
        """Forzar reentrenamiento manualmente."""
        print(f"🔄 Forzando reentrenamiento {retraining_type}...")

        if retraining_type == 'incremental':
            self._execute_incremental_retraining()
        elif retraining_type == 'full':
            self._execute_full_retraining()
        else:
            print(f"❌ Tipo de reentrenamiento no válido: {retraining_type}")


# ----------------------------------------------------------------------
# CLI SENCILLA
# ----------------------------------------------------------------------
def main():
    """Función principal del programador - VERSIÓN SIMPLIFICADA (sin hilos)."""
    print("🚀 SISTEMA DE REENTRENAMIENTO AUTOMÁTICO - OLIST")
    print("=" * 60)

    scheduler = RetrainingScheduler()

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
                scheduler.force_retraining('incremental')
            elif choice == '3':
                print("\n🔄 Ejecutando reentrenamiento completo...")
                scheduler.force_retraining('full')
            elif choice == '4':
                print("👋 Saliendo...")
                break
            else:
                print("❌ Opción no válida. Por favor selecciona 1-4.")

    except KeyboardInterrupt:
        print("\n👋 Saliendo por Ctrl+C...")
        scheduler.stop_scheduler()
    except Exception as e:
        print(f"❌ Error: {e}")
        scheduler.stop_scheduler()


if __name__ == "__main__":
    main()



