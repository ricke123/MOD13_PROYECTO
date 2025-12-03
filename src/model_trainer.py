"""
Módulo para entrenamiento de modelos - VERSIÓN CON HIPERPARÁMETROS EN CONFIG
"""
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
import json
from pathlib import Path

warnings.filterwarnings('ignore')

from src.config import (
    MODEL_CONFIG, MODEL_DIR, TIME_PERIODS, 
    XGBOOST_PARAMS, RANDOM_FOREST_PARAMS, RANDOM_SEARCH_CONFIG
)


class ModelTrainer:
    def __init__(self):
        self.config = MODEL_CONFIG
        self.model_dir = MODEL_DIR
        self.xgb_params = XGBOOST_PARAMS
        self.rf_params = RANDOM_FOREST_PARAMS
        self.random_search_config = RANDOM_SEARCH_CONFIG

        # Crear carpeta de modelos si no existe
        Path(self.model_dir).mkdir(parents=True, exist_ok=True)

    def calculate_comprehensive_metrics(self, y_true, y_pred, y_train=None):
        """Calcular métricas comprehensivas"""
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)

        mape = np.mean(
            np.abs((y_true - y_pred) / np.where(y_true == 0, 1, y_true))
        ) * 100

        if y_train is not None:
            naive_forecast_errors = np.abs(np.diff(y_train))
            if len(naive_forecast_errors) > 0:
                mean_naive_error = np.mean(naive_forecast_errors)
                mase = mae / mean_naive_error if mean_naive_error != 0 else np.inf
            else:
                mase = np.inf
        else:
            mase = np.inf

        return {
            'MAE': float(mae),
            'RMSE': float(rmse),
            'R²': float(r2),
            'MAPE': float(mape),
            'MASE': float(mase) if mase != np.inf else float('inf')
        }

    def prepare_temporal_data(self, features_df):
        """Preparar datos con división temporal"""
        print("🧱 PREPARANDO DATOS CON DIVISIÓN TEMPORAL")

        target_col = self.config.get('target_col', 'demand_next_month')
        
        if 'purchase_year_month' in features_df.columns:
            print("📅 Usando división TEMPORAL por purchase_year_month")

            if not pd.api.types.is_datetime64_any_dtype(features_df['purchase_year_month']):
                features_df['purchase_year_month'] = pd.to_datetime(
                    features_df['purchase_year_month'].astype(str)
                )

            train_start = TIME_PERIODS['train_start']
            train_end   = TIME_PERIODS['train_end']
            test_start  = TIME_PERIODS['test_start']
            test_end    = TIME_PERIODS['test_end']

            print(f"📅 Train: {train_start} → {train_end}")
            print(f"📅 Test:  {test_start} → {test_end}")

            train_mask = (
                (features_df['purchase_year_month'] >= train_start) &
                (features_df['purchase_year_month'] <= train_end)
            )
            test_mask = (
                (features_df['purchase_year_month'] >= test_start) &
                (features_df['purchase_year_month'] <= test_end)
            )

            train_data = features_df[train_mask].copy()
            test_data  = features_df[test_mask].copy()

            print(f"📊 Train shape: {train_data.shape}")
            print(f"📊 Test  shape: {test_data.shape}")

            X_train = train_data.drop(columns=[target_col, 'purchase_year_month'], errors='ignore')
            X_test  = test_data.drop(columns=[target_col, 'purchase_year_month'], errors='ignore')
            y_train = train_data[target_col]
            y_test  = test_data[target_col]

        else:
            # Fallback: división aleatoria
            print("⚠️ Usando train_test_split aleatorio.")
            X = features_df.drop(columns=[target_col], errors='ignore')
            y = features_df[target_col].copy()

            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=0.2,
                random_state=self.config['random_state']
            )

        return X_train, X_test, y_train, y_test

    def train_optimized_xgboost(self, X_train, X_test, y_train, y_test):
        """Entrenar XGBoost con parámetros de config.py"""
        print("🧠 Entrenando XGBoost optimizado...")

        print("⚙️  HIPERPARÁMETROS XGBOOST (desde config.py):")
        for param, value in self.xgb_params.items():
            print(f"   • {param:20} : {value}")

        xgb_model = xgb.XGBRegressor(
            **self.xgb_params,
            random_state=self.config['random_state'],
            n_jobs=-1,
            tree_method='hist',
            early_stopping_rounds=50,
            eval_metric='mae'
        )

        xgb_model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )

        # Predicciones y métricas
        y_pred = xgb_model.predict(X_test)
        y_train_pred = xgb_model.predict(X_train)
        
        test_metrics = self.calculate_comprehensive_metrics(y_test, y_pred, y_train)
        train_metrics = self.calculate_comprehensive_metrics(y_train, y_train_pred, y_train)

        self.print_metrics("XGBoost Optimizado", train_metrics, test_metrics)
        
        return xgb_model, train_metrics, test_metrics


    

    def train_tuned_random_forest(self, X_train, X_test, y_train, y_test):
        """Entrenar Random Forest con parámetros de config.py - VERSIÓN CORREGIDA"""
        print("🎯 Entrenando Random Forest TUNEADO...")
        
        # 1. Hacer una COPIA de los parámetros para depuración
        rf_params = self.rf_params.copy()
        
        # 2. Mostrar qué parámetros tenemos
        print("⚙️  HIPERPARÁMETROS RANDOM FOREST (desde config.py):")
        for param, value in sorted(rf_params.items()):
            print(f"   • {param:25} : {value}")
        
        # 3. VERIFICACIÓN CRÍTICA: Mostrar advertencias si hay duplicados
        if 'random_state' in rf_params:
            print(f"   ✅ random_state ya está en parámetros: {rf_params['random_state']}")
        if 'n_jobs' in rf_params:
            print(f"   ✅ n_jobs ya está en parámetros: {rf_params['n_jobs']}")
        
        # 4. CORRECCIÓN: Crear el modelo SOLO con rf_params
        #    NO añadir random_state=self.config['random_state'] (duplicado)
        #    NO añadir n_jobs=-1 (duplicado)
        #    self.rf_params YA contiene ambos
        print("\n🔧 Creando modelo RandomForestRegressor...")
        try:
            # SOLUCIÓN: Pasar solo rf_params, nada más
            model = RandomForestRegressor(**rf_params)
            print("✅ Modelo creado exitosamente")
        except TypeError as e:
            print(f"❌ Error: {e}")
            print("🔄 Intentando solución alternativa...")
            # Crear con parámetros básicos como respaldo
            model = RandomForestRegressor(
                n_estimators=rf_params.get('n_estimators', 100),
                max_depth=rf_params.get('max_depth', None),
                min_samples_split=rf_params.get('min_samples_split', 2),
                min_samples_leaf=rf_params.get('min_samples_leaf', 1),
                max_features=rf_params.get('max_features', 'sqrt'),
                random_state=rf_params.get('random_state', 42),
                n_jobs=rf_params.get('n_jobs', -1),
                bootstrap=rf_params.get('bootstrap', True)
            )
        
        # 5. Entrenar el modelo
        print("\n📊 Entrenando modelo...")
        model.fit(X_train, y_train)
        
        # 6. Predicciones y métricas
        y_pred = model.predict(X_test)
        y_train_pred = model.predict(X_train)
        
        test_metrics = self.calculate_comprehensive_metrics(y_test, y_pred, y_train)
        train_metrics = self.calculate_comprehensive_metrics(y_train, y_train_pred, y_train)

        self.print_metrics("Random Forest Tuneado", train_metrics, test_metrics)
        
        return model, train_metrics, test_metrics












    
    
    def train_random_forest_with_randomsearch(self, X_train, X_test, y_train, y_test):
        """Entrenar Random Forest con RandomizedSearchCV (en tiempo real)"""
        print("🔍 Entrenando Random Forest con RandomizedSearchCV...")
        
        if not self.random_search_config['enable_random_search']:
            print("⚠️ RandomizedSearchCV deshabilitado en config. Usando parámetros fijos.")
            return self.train_tuned_random_forest(X_train, X_test, y_train, y_test)
        
        tscv = TimeSeriesSplit(n_splits=self.random_search_config['cv_splits'])
        
        rf = RandomForestRegressor(
            random_state=self.config['random_state'],
            n_jobs=-1
        )
        
        param_distributions = self.random_search_config['rf_param_distributions']
        
        rf_search = RandomizedSearchCV(
            rf,
            param_distributions=param_distributions,
            n_iter=self.random_search_config['n_iter'],
            cv=tscv,
            scoring=self.random_search_config['scoring'],
            n_jobs=-1,
            verbose=1,
            random_state=42
        )
        
        print("🔍 Buscando mejores hiperparámetros...")
        rf_search.fit(X_train, y_train)
        
        best_rf = rf_search.best_estimator_
        print(f"🎯 Mejores hiperparámetros: {rf_search.best_params_}")
        print(f"🎯 Mejor score MAE: {-rf_search.best_score_:.4f}")
        
        # ACTUALIZAR PARÁMETROS EN CONFIG (opcional)
        self.rf_params.update(rf_search.best_params_)
        print("📝 Parámetros optimizados actualizados en memoria")
        
        y_pred = best_rf.predict(X_test)
        y_train_pred = best_rf.predict(X_train)
        
        test_metrics = self.calculate_comprehensive_metrics(y_test, y_pred, y_train)
        train_metrics = self.calculate_comprehensive_metrics(y_train, y_train_pred, y_train)

        self.print_metrics("Random Forest con RandomizedSearch", train_metrics, test_metrics)
        
        return best_rf, train_metrics, test_metrics

    def print_metrics(self, model_name, train_metrics, test_metrics):
        """Imprimir métricas de forma formateada"""
        print(f"\n📊 RESULTADOS {model_name.upper()}")
        print("=" * 60)
        print("               |   TRAIN    |    TEST    ")
        print("=" * 60)
        print(f"MAE           | {train_metrics['MAE']:>10.2f} | {test_metrics['MAE']:>10.2f}")
        print(f"RMSE          | {train_metrics['RMSE']:>10.2f} | {test_metrics['RMSE']:>10.2f}")
        print(f"R²            | {train_metrics['R²']:>10.4f} | {test_metrics['R²']:>10.4f}")
        print(f"MAPE (%)      | {train_metrics['MAPE']:>10.2f} | {test_metrics['MAPE']:>10.2f}")
        print("=" * 60)

    def save_model(self, model, model_name="optimized_model"):
        """Guardar modelo entrenado"""
        model_type = self.config['model_type']
        model_path = self.model_dir / f"{model_type}_{model_name}.pkl"

        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

        print(f"💾 Modelo guardado en: {model_path}")
        return model_path

    def load_model(self, model_name="optimized_model"):
        """Cargar modelo entrenado"""
        model_type = self.config['model_type']
        model_path = self.model_dir / f"{model_type}_{model_name}.pkl"

        if model_path.exists():
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            print(f"📥 Modelo cargado desde: {model_path}")
            return model
        else:
            print(f"❌ No se encontró el modelo: {model_path}")
            return None

    def feature_importance_analysis(self, model, feature_names, top_n=20):
        """Análisis de importancia de features"""
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)

            print(f"\n🏆 TOP {top_n} FEATURES MÁS IMPORTANTES:")
            for i, row in importance_df.head(top_n).iterrows():
                print(f"  {i+1:2d}. {row['feature']:35} → {row['importance']:.8f}")

            importance_path = self.model_dir / "feature_importances.csv"
            importance_df.to_csv(importance_path, index=False)
            print(f"💾 Importancia guardada en: {importance_path}")

            return importance_df
        return None

    def save_selected_features(self, importance_df, threshold=0.0001):
        """Guardar features seleccionadas"""
        selected = importance_df[importance_df['importance'] >= threshold]['feature'].tolist()

        output_path = self.model_dir / "selected_features.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(selected, f, indent=4, ensure_ascii=False)

        print(f"💾 {len(selected)} features seleccionadas guardadas en: {output_path}")
        return selected

    def train_best_model(self, features_df):
        """
        Entrenar el mejor modelo según configuración
        """
        X_train, X_test, y_train, y_test = self.prepare_temporal_data(features_df)

        # SELECCIÓN DE MODELO
        if self.config['model_type'] == 'xgboost':
            model, train_metrics, test_metrics = self.train_optimized_xgboost(
                X_train, X_test, y_train, y_test
            )
        elif self.config['model_type'] == 'random_forest':
            # DECISIÓN: ¿Usar parámetros fijos o RandomizedSearch?
            if self.random_search_config['enable_random_search']:
                model, train_metrics, test_metrics = self.train_random_forest_with_randomsearch(
                    X_train, X_test, y_train, y_test
                )
            else:
                model, train_metrics, test_metrics = self.train_tuned_random_forest(
                    X_train, X_test, y_train, y_test
                )
        else:
            raise ValueError(f"Modelo no soportado: {self.config['model_type']}")

        # ANÁLISIS DE FEATURES
        feature_names = X_train.columns.tolist()
        importance_df = self.feature_importance_analysis(model, feature_names)

        if importance_df is not None:
            threshold = self.config.get('feature_importance_threshold', 0.0001)
            selected_features = self.save_selected_features(importance_df, threshold=threshold)
            print(f"📌 Features usadas: {len(selected_features)}")

        # GUARDAR MODELO
        model_path = self.save_model(model, "optimized_model")

        return model, train_metrics, test_metrics, model_path




