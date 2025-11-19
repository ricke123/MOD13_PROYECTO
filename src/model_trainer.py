# src/model_trainer.py
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
from datetime import datetime
from config import Config

class ModelTrainer:
    def __init__(self):
        self.models = {}
        self.metrics = {}
        
    def load_training_data(self):
        """Carga el dataset para entrenamiento"""
        try:
            file_path = Config.get_output_path('TABLA_FINAL_MODULAR.csv')
            df = pd.read_csv(file_path)
            print(f"📊 Datos cargados: {df.shape}")
            return df
        except Exception as e:
            print(f"❌ Error cargando datos: {e}")
            return None
    
    def prepare_features(self, df):
        """Prepara features y target para entrenamiento"""
        # Columnas a excluir
        exclude_cols = ['order_month', 'product_category_name', 'date', 'month_year', 'demand_next_month']
        
        # Features (X) y target (y)
        X = df.drop(columns=[col for col in exclude_cols if col in df.columns])
        y = df['demand_next_month']
        
        print(f"🔧 Features: {X.shape[1]}, Target: {y.shape[0]}")
        return X, y
    
    def train_models(self, X, y):
        """Entrena múltiples modelos"""
        print("🤖 Entrenando modelos...")
        
        # Split temporal (último 20% para test)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # 1. Random Forest
        print("🌲 Entrenando Random Forest...")
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train, y_train)
        self.models['random_forest'] = rf_model
        
        # 2. XGBoost
        print("🚀 Entrenando XGBoost...")
        xgb_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=10,
            learning_rate=0.1,
            random_state=42
        )
        xgb_model.fit(X_train, y_train)
        self.models['xgboost'] = xgb_model
        
        # Evaluación
        self.evaluate_models(X_test, y_test)
        
        return self.models
    
    def evaluate_models(self, X_test, y_test):
        """Evalúa los modelos entrenados"""
        print("📈 Evaluando modelos...")
        
        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            
            self.metrics[name] = {
                'MAE': mae,
                'RMSE': rmse,
                'r2_score': model.score(X_test, y_test)
            }
            
            print(f"   {name.upper():<15} - MAE: {mae:.2f}, RMSE: {rmse:.2f}")
    
    def save_models(self):
        """Guarda los modelos entrenados"""
        print("💾 Guardando modelos...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for name, model in self.models.items():
            # Guardar modelo
            model_path = Config.get_output_path(f'model_{name}_{timestamp}.pkl')
            joblib.dump(model, model_path)
            
            # Guardar métricas
            metrics_path = Config.get_output_path(f'metrics_{name}_{timestamp}.json')
            import json
            with open(metrics_path, 'w') as f:
                json.dump(self.metrics.get(name, {}), f, indent=2)
            
            print(f"   ✅ {name} guardado en: {model_path}")
        
        # Guardar el modelo más reciente como "latest"
        best_model_name = min(self.metrics.items(), key=lambda x: x[1]['MAE'])[0]
        best_model = self.models[best_model_name]
        latest_path = Config.get_output_path('model_latest.pkl')
        joblib.dump(best_model, latest_path)
        print(f"   🏆 Mejor modelo ({best_model_name}) guardado como: {latest_path}")
    
    def full_training_pipeline(self):
        """Pipeline completo de entrenamiento"""
        print("🚀 INICIANDO ENTRENAMIENTO DE MODELOS")
        print("=" * 50)
        
        # Cargar datos
        df = self.load_training_data()
        if df is None:
            return None
        
        # Preparar features
        X, y = self.prepare_features(df)
        
        # Entrenar modelos
        self.train_models(X, y)
        
        # Guardar modelos
        self.save_models()
        
        print("🎉 ENTRENAMIENTO COMPLETADO!")
        return self.models

# Uso rápido
if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.full_training_pipeline()