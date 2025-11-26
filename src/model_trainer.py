# src/model_trainer.py (VERSIÓN MEJORADA CON SPLIT TEMPORAL + MAPE + BACKTEST)
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


# =========================
#   MÉTRICA MAPE GLOBAL
# =========================
def mape(y_true, y_pred):
    """
    MAPE robusto:
    - ignora valores donde y_true == 0 o NaN
    - devuelve NaN si no hay valores válidos
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    mask = (~np.isnan(y_true)) & (y_true != 0)
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0


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
        self.metrics = {}           # métricas globales (último split evaluado)
        self.metrics_by_split = {}  # métricas por split: backtest / test_final / holdout
    
    # ========================
    #   CARGA Y PREPARACIÓN
    # ========================
    def load_training_data(self):
        """Carga el dataset para entrenamiento"""
        try:
            file_path = Config.get_output_path('TABLA_FINAL_MODULAR.csv')
            df = pd.read_csv(file_path)
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
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

    # =================================
    #   SPLIT TEMPORAL (TRAIN/BACK/TEST)
    # =================================
    def temporal_split(self, df, backtest_months=3, test_months=1):
        """
        Split temporal:
        - train: todos los meses excepto (backtest_months + test_months) finales
        - backtest: últimos 'backtest_months' ANTES del test
        - test: últimos 'test_months' meses
        Si no hay suficientes meses, cae a un 80/20 por índice.
        """
        if 'date' not in df.columns:
            print("⚠️ No hay columna 'date', usando split 80/20 por índice.")
            split_idx = int(len(df) * 0.8)
            df_train = df.iloc[:split_idx].copy()
            df_backtest = pd.DataFrame(columns=df.columns)
            df_test = df.iloc[split_idx:].copy()
            return df_train, df_backtest, df_test

        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')

        periods = (
            df['date']
            .dt.to_period('M')
            .dropna()
            .drop_duplicates()
            .sort_values()
            .tolist()
        )

        total_periods = len(periods)
        needed = backtest_months + test_months

        if total_periods <= needed + 1:
            print("⚠️ Pocos meses, usando split 80/20 por índice.")
            split_idx = int(len(df) * 0.8)
            df_train = df.iloc[:split_idx].copy()
            df_backtest = pd.DataFrame(columns=df.columns)
            df_test = df.iloc[split_idx:].copy()
            return df_train, df_backtest, df_test

        test_periods = periods[-test_months:]
        backtest_periods = periods[-(backtest_months + test_months):-test_months]
        train_periods = periods[:-(backtest_months + test_months)]

        df_train = df[df['date'].dt.to_period('M').isin(train_periods)].copy()
        df_backtest = df[df['date'].dt.to_period('M').isin(backtest_periods)].copy()
        df_test = df[df['date'].dt.to_period('M').isin(test_periods)].copy()

        print(f"🧪 Periodos totales: {total_periods}")
        print(f"   Train meses:    {len(train_periods)} → filas: {df_train.shape[0]}")
        print(f"   Backtest meses: {len(backtest_periods)} → filas: {df_backtest.shape[0]}")
        print(f"   Test meses:     {len(test_periods)} → filas: {df_test.shape[0]}")
        return df_train, df_backtest, df_test

    # ==========================
    #   ENTRENAMIENTO MODELOS
    # ==========================
    def _fit_all_models(self, X_train, y_train, categories_train=None):
        """Entrena todos los modelos (Naive + ML) sobre el conjunto de entrenamiento."""
        self.models = {}
        print("🤖 Entrenando modelos...")

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
        try:
            if categories_train is not None:
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
    
    # ==========================
    #   EVALUACIÓN MODELOS
    # ==========================
    def evaluate_models(self, X_test, y_test, categories=None, split_name="test", df=None):
        """Evalúa los modelos entrenados incluyendo Naive (con MAPE y R²)."""
        print(f"📈 Evaluando modelos en split: {split_name}")
        self.metrics_by_split.setdefault(split_name, {})

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
                mape_val = mape(y_test, y_pred)

                # R² manual
                ss_res = np.sum((y_test - y_pred) ** 2)
                ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                
                met = {
                    'mae': mae,
                    'rmse': rmse,
                    'mape': mape_val,
                    'r2': r2
                }
                self.metrics[name] = met
                self.metrics_by_split[split_name][name] = met
                
                # Icono especial para modelos naive
                icon = "📊" if 'naive' in name else "🤖"
                print(
                    f"   {icon} {name.upper():<20} - "
                    f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape_val:.2f}%, R²: {r2:.3f}"
                )
                
            except Exception as e:
                print(f"   ❌ Error evaluando {name}: {e}")
                met = {
                    'mae': float('inf'),
                    'rmse': float('inf'),
                    'mape': float('inf'),
                    'r2': -float('inf')
                }
                self.metrics[name] = met
                self.metrics_by_split[split_name][name] = met

    # ==============================
    #   DIAGNÓSTICOS DEL BACKTEST
    # ==============================
    def _backtest_diagnostics(self, df_back, X_back, y_back):
        """
        Crea un DataFrame de evaluación y calcula:
        - MAPE por categoría (Naive vs Random Forest)
        - MAPE por mes (Random Forest)
        """
        if df_back is None or df_back.empty:
            print("⚠️ No hay datos de backtest para diagnósticos.")
            return

        print("\n🧾 Generando diagnósticos de backtest (por categoría y por mes)...")

        eval_df = df_back[['product_category_name', 'date']].copy()
        eval_df['y_true'] = y_back.values

        # Predicciones de modelos relevantes
        for model_name in ['naive_by_category', 'random_forest']:
            if model_name in self.models:
                model = self.models[model_name]
                if model_name == 'naive_by_category':
                    preds = model.predict(X_back, df_back['product_category_name'])
                else:
                    preds = model.predict(X_back)
                eval_df[f'y_pred_{model_name}'] = preds

        # MAPE por categoría
        for model_name in ['naive_by_category', 'random_forest']:
            col = f'y_pred_{model_name}'
            if col in eval_df.columns:
                mape_cat = (
                    eval_df
                    .groupby('product_category_name')
                    .apply(lambda g: mape(g['y_true'], g[col]))
                    .sort_values()
                    .to_frame('mape')
                )
                print(f"\n📊 MAPE por categoría (backtest) – {model_name}: Top 10 categorías con menor MAPE")
                print(mape_cat.head(10))

        # MAPE por mes (solo Random Forest)
        if 'y_pred_random_forest' in eval_df.columns:
            eval_df['month_year'] = eval_df['date'].dt.to_period('M').astype(str)
            mape_month = (
                eval_df
                .groupby('month_year')
                .apply(lambda g: mape(g['y_true'], g['y_pred_random_forest']))
                .to_frame('mape')
            )
            print("\n📊 MAPE por mes (backtest) – random_forest:")
            print(mape_month)

    # ================================================
    #   PIPELINE TEMPORAL COMPLETO (CUADERNO → CÓDIGO)
    # ================================================
    def train_models_temporal(self, df):
        """
        Entrena modelos respetando temporalidad:
        - Split por meses: train / backtest / test_final
        - Evaluación con MAPE + diagnósticos de backtest
        """
        # Asegurar fechas
        if 'date' in df.columns:
            df = df.copy()
            df['date'] = pd.to_datetime(df['date'])

        # Split temporal
        df_train, df_back, df_test = self.temporal_split(df, backtest_months=3, test_months=1)

        # Preparar features para cada subset
        X_train, y_train = self.prepare_features(df_train)
        categories_train = df_train['product_category_name'] if 'product_category_name' in df_train.columns else None

        self._fit_all_models(X_train, y_train, categories_train=categories_train)

        # BACKTEST
        if not df_back.empty:
            X_back, y_back = self.prepare_features(df_back)
            categories_back = df_back['product_category_name'] if 'product_category_name' in df_back.columns else None
            self.evaluate_models(X_back, y_back, categories=categories_back, split_name='backtest', df=df_back)
            self._backtest_diagnostics(df_back, X_back, y_back)

        # TEST FINAL
        X_test, y_test = self.prepare_features(df_test)
        categories_test = df_test['product_category_name'] if 'product_category_name' in df_test.columns else None
        self.evaluate_models(X_test, y_test, categories=categories_test, split_name='test_final', df=df_test)

        return self.models

    # ==================================
    #   MODO 80/20 (REENTRENAMIENTO)
    # ==================================
    def train_models(self, X, y, df=None):
        """
        Entrena múltiples modelos con un split 80/20 por índice.
        Este modo se sigue usando para reentrenamientos simples.
        """
        print("🤖 Entrenando modelos con split 80/20 (holdout)...")
        
        # Split temporal por índice (mantiene tu comportamiento original)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        print(f"📈 Split: Train {X_train.shape}, Test {X_test.shape}")
        
        # Categorías (si están disponibles)
        categories_train = None
        categories_test = None
        if df is not None and 'product_category_name' in df.columns:
            categories_train = df['product_category_name'].iloc[:split_idx]
            categories_test = df['product_category_name'].iloc[split_idx:]
        
        # Entrenar todos los modelos
        self._fit_all_models(X_train, y_train, categories_train=categories_train)
        
        # Evaluación holdout
        self.evaluate_models(X_test, y_test, categories_test, split_name="holdout_80_20", df=df)
        
        return self.models
    
    def train_models_from_dataframe(self, df, target_column='demand_next_month'):
        """
        Entrena modelos a partir de un DataFrame completo.
        Usa el modo 80/20 (holdout) para reentrenamiento.
        """
        print(f"🔧 Preparando datos desde DataFrame: {df.shape}")
        
        X, y = self.prepare_features(df)
        
        # Verificar que no haya valores NaN
        if X.isna().any().any() or y.isna().any():
            print("⚠️ Limpiando valores NaN...")
            X = X.fillna(0)
            y = y.fillna(0)
        
        # Entrenar modelos con split 80/20
        return self.train_models(X, y, df)
    
    # ======================
    #   GUARDADO DE MODELOS
    # ======================
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
    
    # =============================
    #   PIPELINE COMPLETO PRINCIPAL
    # =============================
    def full_training_pipeline(self):
        """Pipeline completo de entrenamiento con split temporal + backtest."""
        print("🚀 INICIANDO ENTRENAMIENTO DE MODELOS (CON BACKTEST)")
        print("=" * 60)
        
        # Cargar datos
        df = self.load_training_data()
        if df is None:
            print("❌ No se pudieron cargar datos para entrenamiento")
            return None
        
        # Entrenar respetando temporalidad y hacer diagnósticos tipo cuaderno
        models = self.train_models_temporal(df)
        
        # Guardar modelos
        self.save_models()
        
        print("🎉 ENTRENAMIENTO COMPLETADO!")
        return models


# Prueba rápida
if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.full_training_pipeline()
