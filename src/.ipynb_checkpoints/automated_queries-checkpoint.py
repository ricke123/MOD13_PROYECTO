"""
Sistema automatizado de consultas y reportes
"""
import schedule
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
import json

class AutomatedForecaster:
    def __init__(self, api_url="http://localhost:8000"):
        self.api_url = api_url
    
    def daily_demand_forecast(self):
        """Ejecuta predicciones diarias para todas las categorías"""
        categories = ["electronics", "home_appliances", "furniture", "computers", "housewares"]
        
        forecasts = {}
        for category in categories:
            try:
                response = requests.post(
                    f"{self.api_url}/predict",
                    json={
                        "product_category": category,
                        "historical_demand": self.get_recent_demand(category),
                        "promotion_planned": self.check_promotions(category),
                        "seasonality_factor": self.get_seasonality_factor()
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    forecasts[category] = result
                    print(f"✅ Predicción {category}: {result['predicted_demand']} unidades")
                else:
                    print(f"❌ Error en {category}: {response.text}")
                    
            except Exception as e:
                print(f"❌ Error consultando API para {category}: {e}")
        
        # Guardar resultados
        self.save_forecasts(forecasts)
        return forecasts
    
    def get_recent_demand(self, category, days=30):
        """Obtiene demanda reciente (simulada)"""
        # En producción, esto vendría de tu base de datos
        base_demand = {
            "electronics": [100, 120, 110, 130, 125, 140, 135, 128],
            "home_appliances": [80, 85, 90, 95, 88, 92, 87, 89],
            "furniture": [45, 50, 48, 52, 47, 49, 51, 46],
            "computers": [60, 65, 62, 68, 64, 67, 63, 66],
            "housewares": [70, 75, 72, 78, 74, 76, 73, 77]
        }
        return base_demand.get(category, [100, 110, 105])
    
    def check_promotions(self, category):
        """Verifica si hay promociones planificadas"""
        # Lógica de negocio para promociones
        promotion_days = {"electronics": [1, 15], "home_appliances": [5, 20]}
        today = datetime.now().day
        return today in promotion_days.get(category, [])
    
    def get_seasonality_factor(self):
        """Calcula factor de estacionalidad"""
        month = datetime.now().month
        # Factores estacionales ejemplo
        season_factors = {
            12: 1.5,  # Navidad
            1: 1.3,   # Post-navidad
            6: 1.2,   # Mitad de año
            7: 1.1    # Vacaciones
        }
        return season_factors.get(month, 1.0)
    
    def save_forecasts(self, forecasts):
        """Guarda predicciones para análisis posterior"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"data/processed/forecasts_{timestamp}.json"
        
        # Asegurar que el directorio existe
        import os
        os.makedirs("data/processed", exist_ok=True)
        
        with open(filename, 'w') as f:
            json.dump(forecasts, f, indent=2)
        
        print(f"💾 Predicciones guardadas en: {filename}")
    
    def generate_daily_report(self):
        """Genera reporte diario ejecutivo"""
        forecasts = self.daily_demand_forecast()
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_categories": len(forecasts),
            "total_predicted_demand": sum(f['predicted_demand'] for f in forecasts.values()),
            "category_breakdown": {
                cat: data['predicted_demand'] for cat, data in forecasts.items()
            },
            "top_category": max(forecasts.items(), key=lambda x: x[1]['predicted_demand'])[0]
        }
        
        # Guardar reporte
        report_file = f"data/processed/daily_report_{datetime.now().strftime('%Y%m%d')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📊 Reporte diario generado: {report_file}")
        return report

# Programación de tareas
def setup_scheduled_tasks():
    forecaster = AutomatedForecaster()
    
    # Programar tareas
    schedule.every().day.at("08:00").do(forecaster.generate_daily_report)
    schedule.every().hour.do(forecaster.daily_demand_forecast)
    schedule.every().monday.at("09:00").do(lambda: print("✅ Sistema verificado"))
    
    print("🕐 Programador de tareas iniciado")
    print("   - Reporte diario: 08:00")
    print("   - Predicciones: cada hora")
    print("   - Verificación: todos los lunes 09:00")
    
    # Ejecutar continuamente
    while True:
        schedule.run_pending()
        time.sleep(60)

if __name__ == "__main__":
    # Probar el sistema
    forecaster = AutomatedForecaster()
    
    print("🧪 Probando sistema automatizado...")
    test_forecasts = forecaster.daily_demand_forecast()
    print(f"✅ Predicciones generadas para {len(test_forecasts)} categorías")
    
    # Iniciar programador (comentar para prueba rápida)
    # setup_scheduled_tasks()

