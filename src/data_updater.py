# src/data_updater.py
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from data_loader import DataLoader
from feature_engineer import FeatureEngineer
from preprocessor import DataPreprocessor
from config import Config

class DataUpdater:
    def __init__(self):
        self.new_data = None
        
    def simulate_new_monthly_data(self, months_to_add=1):
        """Simula la llegada de nuevos datos mensuales"""
        print(f"🔄 Simulando {months_to_add} mes(es) de nuevos datos...")
        
        # Cargar datos existentes para referencia
        existing_data = self.load_existing_data()
        if existing_data is None:
            print("❌ No se pudieron cargar datos existentes")
            return None
        
        # Obtener el último mes disponible
        last_date = pd.to_datetime(existing_data['order_month'] + '-01').max()
        new_dates = []
        
        for i in range(1, months_to_add + 1):
            new_month = last_date + pd.DateOffset(months=i)
            new_dates.append(new_month.strftime('%Y-%m'))
        
        print(f"📅 Nuevos meses a simular: {new_dates}")
        return self.generate_simulated_data(existing_data, new_dates)
    
    def load_existing_data(self):
        """Carga los datos existentes procesados"""
        try:
            file_path = Config.get_output_path('TABLA_FINAL_MODULAR.csv')
            return pd.read_csv(file_path)
        except:
            print("⚠️ No se encontraron datos existentes, ejecutando pipeline completo...")
            from main import main
            return main()
    
    def generate_simulated_data(self, existing_data, new_months):
        """Genera datos simulados para nuevos meses"""
        simulated_data = []
        
        for month in new_months:
            # Para cada categoría, generar datos similares al histórico
            categories = existing_data['product_category_name'].unique()
            
            for category in categories:
                # Filtrar datos históricos de esta categoría
                cat_data = existing_data[existing_data['product_category_name'] == category]
                
                if len(cat_data) > 0:
                    # Tomar la última observación como DataFrame (no Series)
                    last_row = cat_data.tail(1).copy()
                    
                    # Actualizar mes
                    last_row['order_month'] = month
                    last_row['date'] = month + '-01'
                    
                    # Simular variaciones (ruido gaussiano)
                    # CORRECCIÓN: Usar last_row como DataFrame, no como Series
                    numeric_cols = last_row.select_dtypes(include=[np.number]).columns
                    
                    for col in numeric_cols:
                        if col != 'demand_next_month':  # No modificar target
                            if last_row[col].iloc[0] > 0:  # Solo modificar valores positivos
                                variation = np.random.normal(1.0, 0.1)  # ±10% variación
                                new_value = max(0, last_row[col].iloc[0] * variation)
                                last_row[col] = new_value
                    
                    simulated_data.append(last_row)
        
        if simulated_data:
            return pd.concat(simulated_data, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def update_dataset(self, new_data):
        """Actualiza el dataset con nuevos datos"""
        print("📊 Actualizando dataset completo...")
        
        # Cargar datos existentes
        existing_data = self.load_existing_data()
        
        if existing_data is not None and not new_data.empty:
            # Combinar datos existentes con nuevos
            updated_data = pd.concat([existing_data, new_data], ignore_index=True)
            
            # Ordenar por categoría y fecha
            updated_data = updated_data.sort_values(['product_category_name', 'date']).reset_index(drop=True)
            
            # Recalcular features que dependen del tiempo
            updated_data = self.recalculate_temporal_features(updated_data)
            
            print(f"✅ Dataset actualizado: {existing_data.shape} → {updated_data.shape}")
            return updated_data
        else:
            print("❌ No hay datos nuevos para actualizar")
            return existing_data
    
    def recalculate_temporal_features(self, df):
        """Recalcula features temporales después de añadir nuevos datos"""
        print("🔄 Recalculando features temporales...")
        
        # Aquí puedes re-aplicar las transformaciones temporales necesarias
        # Por simplicidad, vamos a mantener las features existentes
        
        return df
    
    def save_updated_data(self, df):
        """Guarda el dataset actualizado"""
        output_path = Config.get_output_path('TABLA_FINAL_MODULAR.csv')
        
        # Guardar backup del anterior
        if os.path.exists(output_path):
            backup_path = Config.get_output_path(f'TABLA_FINAL_BACKUP_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
            os.rename(output_path, backup_path)
            print(f"💾 Backup guardado: {backup_path}")
        
        # Guardar nuevo dataset
        df.to_csv(output_path, index=False)
        print(f"💾 Dataset actualizado guardado: {output_path}")
        
        return output_path

# Uso rápido
if __name__ == "__main__":
    updater = DataUpdater()
    new_data = updater.simulate_new_monthly_data(months_to_add=1)
    
    if new_data is not None and not new_data.empty:
        updated_data = updater.update_dataset(new_data)
        updater.save_updated_data(updated_data)
        print(f"🎉 Simulación completada. Nuevo dataset: {updated_data.shape}")
    else:
        print("❌ No se pudieron generar datos nuevos")