# src/model_trainer.py (VERSIÓN COMPLETA CORREGIDA)
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import datetime
from config import Config

# Intentar importar xgboost, si no está disponible usar alternativa
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
    print("✅ XGBoost disponible")
except ImportError:
    XGB_AVAILABLE = False
    print("⚠️ XGBoost no disponible, usando alternativas")

class NaiveModel:
    """
    Modelo Naive que predice basado en:
    - Último valor conocido (Last Value)
    - Promedio histórico (Historical Mean)
    - Último valor por categoría (Last Value by Category)
    """
    
    def __init__(self, strategy='last_value'):
        self.strategy = strategy
        self.last_values = {}
        self.historical_mean = 0
        self.last_value = 0
        self.is_fitted = False
        
    def fit(self, X, y, categories=None):
        """Entrena el modelo naive"""
        if self.strategy == 'last_value':
            # Para last_value, simplemente guardamos el último valor
            self.last_value = y.iloc[-1] if len(y) > 0 else 0
            
        elif self.strategy == 'historical_mean':
            # Para historical_mean, calculamos la media
            self.historical_mean = y.mean()
            
        elif self.strategy == 'last_value_by_category' and categories is not None:
            # Para last_value_by_category, guardamos el último valor por categoría
            for category in categories.unique():
                mask = categories == category
                if mask.any() and len(y[mask]) > 0:
                    self.last_values[category] = y[mask].iloc[-1]
            # También calculamos la media global como fallback
            self.historical_mean = y.mean()
        
        self.is_fitted = True
        return self
    
    def predict(self, X, categories=None):
        """Predice usando la estrategia naive"""
        if not self.is_fitted:
            raise ValueError("Modelo no entrenado")
            
        n_samples = len(X)
        
        if self.strategy == 'last_value':
            return np.full(n_samples, self.last_value)
            
        elif self.strategy == 'historical_mean':
            return np.full(n_samples, self.historical_mean)
            
        elif self.strategy == 'last_value_by_category' and categories is not None:
            predictions = []
            for category in categories:
                # Usar último valor de la categoría o media histórica como fallback
                pred_value = self.last_values.get(category, self.historical_mean)
                predictions.append(pred_value)
            return np.array(predictions)
            
        else:
            # Fallback: ceros
            return np.zeros(n_samples)

class ModelTrainer:
    def __init__(self):
        self.models = {}
        self.metrics = {}
        
    def load_training_data(self):
        """Carga el dataset para entrenamiento"""
        try:
            file_path = Config.get_output_path('TABLA_FINAL_MODULAR.csv')
            df = pd.read_csv(file_path)
            print(f"📊 Datos de entrenamiento cargados: {df.shape}")
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
    
    def train_models(self, X, y, df=None):
        """Entrena múltiples modelos incluyendo Naive"""
        print("🤖 Entrenando modelos...")
        
        # Split temporal (último 20% para test)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        print(f"📈 Split: Train {X_train.shape}, Test {X_test.shape}")
        
        # 0. MODELOS NAIVE (Baselines)
        print("📊 Entrenando modelos Naive...")
        
        # Naive 1: Último valor
        try:
            naive_last = NaiveModel(strategy='last_value')
            naive_last.fit(X_train, y_train)
            self.models['naive_last_value'] = naive_last
            print("   ✅ Naive Last Value entrenado")
        except Exception as e:
            print(f"   ❌ Error entrenando Naive Last Value: {e}")
    
        # Naive 2: Media histórica
        try:
            naive_mean = NaiveModel(strategy='historical_mean')
            naive_mean.fit(X_train, y_train)
            self.models['naive_historical_mean'] = naive_mean
            print("   ✅ Naive Historical Mean entrenado")
        except Exception as e:
            print(f"   ❌ Error entrenando Naive Historical Mean: {e}")
        
        # Naive 3: Último valor por categoría (si tenemos información de categorías)
        categories_test = None
        try:
            if df is not None and 'product_category_name' in df.columns:
                categories_train = df['product_category_name'].iloc[:split_idx]
                categories_test = df['product_category_name'].iloc[split_idx:]
                
                naive_by_category = NaiveModel(strategy='last_value_by_category')
                naive_by_category.fit(X_train, y_train, categories_train)
                self.models['naive_by_category'] = naive_by_category
                print("   ✅ Naive By Category entrenado")
        except Exception as e:
            print(f"   ❌ Error entrenando Naive By Category: {e}")
        
        # 1. Random Forest
        print("🌲 Entrenando Random Forest...")
        try:
            rf_model = RandomForestRegressor(
                n_estimators=50,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            rf_model.fit(X_train, y_train)
            self.models['random_forest'] = rf_model
        except Exception as e:
            print(f"   ❌ Error entrenando Random Forest: {e}")
        
        # 2. Linear Regression
        print("📐 Entrenando Linear Regression...")
        try:
            lr_model = LinearRegression()
            lr_model.fit(X_train, y_train)
            self.models['linear_regression'] = lr_model
        except Exception as e:
            print(f"   ❌ Error entrenando Linear Regression: {e}")
        
        # 3. XGBoost
        if XGB_AVAILABLE:
            print("🚀 Entrenando XGBoost...")
            try:
                xgb_model = xgb.XGBRegressor(
                    n_estimators=50,
                    max_depth=8,
                    learning_rate=0.1,
                    random_state=42
                )
                xgb_model.fit(X_train, y_train)
                self.models['xgboost'] = xgb_model
            except Exception as e:
                print(f"   ❌ Error entrenando XGBoost: {e}")
        else:
            print("⏭️  Saltando XGBoost (no disponible)")
        
        # Evaluación
        self.evaluate_models(X_test, y_test, categories_test, df)
        
        return self.models
    
    def evaluate_models(self, X_test, y_test, categories=None, df=None):
        """Evalúa los modelos entrenados incluyendo Naive"""
        print("📈 Evaluando modelos...")
        
        for name, model in self.models.items():
            try:
                if 'naive' in name:
                    # Modelos Naive necesitan tratamiento especial
                    if name == 'naive_by_category' and categories is not None:
                        y_pred = model.predict(X_test, categories)
                    else:
                        y_pred = model.predict(X_test)
                else:
                    # Modelos sklearn estándar
                    y_pred = model.predict(X_test)
                
                # Asegurar que no hay valores NaN en las predicciones
                y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=0.0, neginf=0.0)
                
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                
                # R² manual para todos los modelos (más consistente)
                ss_res = np.sum((y_test - y_pred) ** 2)
                ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                
                self.metrics[name] = {
                    'mae': mae,
                    'rmse': rmse,
                    'r2': r2
                }
                
                # Icono especial para modelos naive
                icon = "📊" if 'naive' in name else "🤖"
                print(f"   {icon} {name.upper():<20} - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.3f}")
                
            except Exception as e:
                print(f"   ❌ Error evaluando {name}: {e}")
                self.metrics[name] = {
                    'mae': float('inf'),
                    'rmse': float('inf'),
                    'r2': -float('inf')
                }
    
    def train_models_from_dataframe(self, df, target_column='demand_next_month'):
        """
        Entrena modelos a partir de un DataFrame completo
        Extrae automáticamente features y target
        """
        print(f"🔧 Preparando datos desde DataFrame: {df.shape}")
        
        # Usar el método existente prepare_features
        X, y = self.prepare_features(df)
        
        # Verificar que no haya valores NaN
        if X.isna().any().any() or y.isna().any():
            print("⚠️ Limpiando valores NaN...")
            X = X.fillna(0)
            y = y.fillna(0)
        
        # Entrenar modelos pasando el DataFrame completo para los modelos naive
        return self.train_models(X, y, df)
    
    def save_models(self):
        """Guarda los modelos entrenados (excepto Naive)"""
        print("💾 Guardando modelos...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        models_saved = 0
        
        for name, model in self.models.items():
            # Solo guardar modelos que no sean Naive (son muy simples)
            if 'naive' not in name:
                model_path = Config.get_output_path(f'model_{name}_{timestamp}.pkl')
                joblib.dump(model, model_path)
                print(f"   ✅ {name} guardado en: {model_path}")
                models_saved += 1
            else:
                print(f"   ⏭️  {name} no guardado (modelo naive)")
        
        # Guardar el mejor modelo como "latest" (excluyendo Naive)
        if self.metrics:
            non_naive_metrics = {k: v for k, v in self.metrics.items() if 'naive' not in k}
            if non_naive_metrics:
                best_model_name = min(non_naive_metrics.items(), key=lambda x: x[1]['mae'])[0]
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
            print("❌ No se pudieron cargar datos para entrenamiento")
            return None
        
        # Preparar features
        X, y = self.prepare_features(df)
        
        # Entrenar modelos
        models = self.train_models(X, y, df)
        
        # Guardar modelos
        self.save_models()
        
        print("🎉 ENTRENAMIENTO COMPLETADO!")
        return models

# Prueba rápida
if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.full_training_pipeline()