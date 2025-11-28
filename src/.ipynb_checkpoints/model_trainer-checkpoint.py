"""
Módulo para entrenamiento de modelos - CONFIGURACIÓN EXACTA DEL EDA
"""
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
import json
from pathlib import Path

warnings.filterwarnings('ignore')

from src.config import MODEL_CONFIG, MODEL_DIR, TIME_PERIODS


class ModelTrainer:
    def __init__(self):
        self.config = MODEL_CONFIG
        self.model_dir = MODEL_DIR

        # Crear carpeta de modelos si no existe
        Path(self.model_dir).mkdir(parents=True, exist_ok=True)

    def calculate_comprehensive_metrics(self, y_true, y_pred, y_train=None):
        """Calcular métricas comprehensivas como en tu EDA - MEJORADO"""
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)

        # MAPE (evitando división por cero)
        mape = np.mean(
            np.abs((y_true - y_pred) / np.where(y_true == 0, 1, y_true))
        ) * 100

        # MASE (Mean Absolute Scaled Error)
        if y_train is not None:
            naive_forecast_errors = np.abs(np.diff(y_train))
            if len(naive_forecast_errors) > 0:
                mean_naive_error = np.mean(naive_forecast_errors)
                mase = mae / mean_naive_error if mean_naive_error != 0 else np.inf
            else:
                mase = np.inf
        else:
            mase = np.inf

        # CONVERTIR a tipos nativos de Python y formatear
        return {
            'MAE': float(mae),
            'RMSE': float(rmse),
            'R²': float(r2),
            'MAPE': float(mape),
            'MASE': float(mase) if mase != np.inf else float('inf')
        }

    def prepare_temporal_data(self, features_df):
        """
        Preparar datos de entrenamiento/prueba.
        Si existe 'purchase_year_month' → split temporal.
        Si no → train_test_split aleatorio con defaults seguros.
        """
        print("🧱 PREPARANDO DATOS CON DIVISIÓN TEMPORAL")

        target_col = self.config.get('target_col', 'demand_next_month')
        test_size = self.config.get('test_size', 0.2)
        random_state = self.config.get('random_state', 42)

        # Si tenemos columna de fecha, usamos el split temporal
        if 'purchase_year_month' in features_df.columns:
            print("📅 Usando división TEMPORAL por purchase_year_month")

            if not pd.api.types.is_datetime64_any_dtype(features_df['purchase_year_month']):
                features_df['purchase_year_month'] = pd.to_datetime(
                    features_df['purchase_year_month'].astype(str)
                )

            train_start = '2016-09'
            train_end   = '2018-04'
            test_start  = '2018-05'
            test_end    = '2018-07'

            print(f"📅 Período entrenamiento: {train_start} a {train_end}")
            print(f"📅 Período prueba:       {test_start} a {test_end}")

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
            print("⚠️ No se encontró 'purchase_year_month'. Usando train_test_split aleatorio.")
            X = features_df.drop(columns=[target_col], errors='ignore')
            y = features_df[target_col].copy()

            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=test_size,
                random_state=random_state
            )

            print(f"📊 Train shape: {X_train.shape}")
            print(f"📊 Test  shape: {X_test.shape}")

        return X_train, X_test, y_train, y_test

    def train_optimized_xgboost(self, X_train, X_test, y_train, y_test):
        """Entrenar XGBoost con configuración EXACTA del EDA"""
        print("🧠 Entrenando modelo XGBoost optimizado (configuración EDA)...")

        # HIPERPARÁMETROS OPTIMIZADOS EXACTOS del EDA
        best_params = {
            'colsample_bylevel': 0.6072,
            'colsample_bytree': 0.7976,
            'gamma': 0.1788,
            'learning_rate': 0.0348,
            'max_depth': 10,
            'min_child_weight': 7,
            'n_estimators': 294,
            'reg_alpha': 0.0882,
            'reg_lambda': 0.0257,
            'subsample': 0.6375
        }

        # Configuración EXACTA del EDA
        xgb_optimized = xgb.XGBRegressor(
            **best_params,
            random_state=self.config['random_state'],
            n_jobs=-1,
            tree_method='hist',
            early_stopping_rounds=50,
            eval_metric='mae'
        )

        print("⚙️  HIPERPARÁMETROS OPTIMIZADOS (EDA):")
        for param, value in best_params.items():
            print(f"   • {param:20} : {value}")

        # Entrenar con early stopping EXACTO como en el EDA
        xgb_optimized.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )

        # Predicciones
        y_pred_optimized = xgb_optimized.predict(X_test)

        # Métricas COMPREHENSIVAS como en el EDA
        metrics_optimized = self.calculate_comprehensive_metrics(
            y_test, y_pred_optimized, y_train
        )

        print(f"\n📊 RESULTADOS MODELO OPTIMIZADO (Test Set):")
        print("=" * 80)
        print(f"   • MAE:  {metrics_optimized['MAE']:>10.2f}")
        print(f"   • RMSE: {metrics_optimized['RMSE']:>10.2f}")
        print(f"   • R²:   {metrics_optimized['R²']:>10.4f}")
        print(f"   • MAPE: {metrics_optimized['MAPE']:>10.2f}%")
        print(f"   • MASE: {metrics_optimized['MASE']:>10.4f}")
        print("=" * 80)

        # También mostrar métricas de entrenamiento para comparación
        y_train_pred = xgb_optimized.predict(X_train)
        train_metrics = self.calculate_comprehensive_metrics(
            y_train, y_train_pred, y_train
        )

        print(f"\n📊 RESULTADOS MODELO OPTIMIZADO (Train Set):")
        print("=" * 80)
        print(f"   • MAE:  {train_metrics['MAE']:>10.2f}")
        print(f"   • RMSE: {train_metrics['RMSE']:>10.2f}")
        print(f"   • R²:   {train_metrics['R²']:>10.4f}")
        print(f"   • MAPE: {train_metrics['MAPE']:>10.2f}%")
        print("=" * 80)

        return xgb_optimized, train_metrics, metrics_optimized

    def train_random_forest_model(self, X_train, X_test, y_train, y_test):
        """Entrenar modelo Random Forest (backup)"""
        print("🌲 Entrenando modelo Random Forest...")

        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=self.config['random_state'],
            n_jobs=-1
        )

        model.fit(X_train, y_train)

        # Predicciones
        y_pred = model.predict(X_test)
        y_train_pred = model.predict(X_train)

        # Métricas
        test_metrics = self.calculate_comprehensive_metrics(y_test, y_pred, y_train)
        train_metrics = self.calculate_comprehensive_metrics(y_train, y_train_pred, y_train)

        print(f"\n📊 RESULTADOS RANDOM FOREST (Test Set):")
        print("=" * 80)
        print(f"   • MAE:  {test_metrics['MAE']:>10.2f}")
        print(f"   • RMSE: {test_metrics['RMSE']:>10.2f}")
        print(f"   • R²:   {test_metrics['R²']:>10.4f}")
        print(f"   • MAPE: {test_metrics['MAPE']:>10.2f}%")
        print("=" * 80)

        return model, train_metrics, test_metrics

    def save_model(self, model, model_name="xgboost_optimized_model"):
        """Guardar modelo entrenado"""
        model_path = self.model_dir / f"{model_name}.pkl"

        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

        print(f"💾 Modelo guardado en: {model_path}")
        return model_path

    def load_model(self, model_name="xgboost_optimized_model"):
        """Cargar modelo entrenado"""
        model_path = self.model_dir / f"{model_name}.pkl"

        if model_path.exists():
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            print(f"📥 Modelo cargado desde: {model_path}")
            return model
        else:
            print(f"❌ No se encontró el modelo: {model_path}")
            return None

    def feature_importance_analysis(self, model, feature_names, top_n=20):
        """Análisis de importancia de features y guardado a CSV"""
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)

            print(f"\n🏆 TOP {top_n} FEATURES POR IMPORTANCIA:")
            print("=" * 60)
            for i, row in importance_df.head(top_n).iterrows():
                print(f"  {i+1:2d}. {row['feature']:35} → {row['importance']:.8f}")

            # Guardar importancias completas
            importance_path = self.model_dir / "feature_importances.csv"
            importance_df.to_csv(importance_path, index=False)
            print(f"💾 Importancia de features guardada en: {importance_path}")

            return importance_df
        else:
            print("❌ El modelo no tiene atributo feature_importances_")
            return None

    def save_selected_features(self, importance_df, threshold=0.0001):
        """
        Guarda las features seleccionadas según el threshold de importancia.
        - importance_df: DataFrame con columnas ['feature', 'importance']
        - threshold: umbral mínimo de importancia para conservar la feature
        """
        selected = importance_df[importance_df['importance'] >= threshold]['feature'].tolist()

        output_path = self.model_dir / "selected_features.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(selected, f, indent=4, ensure_ascii=False)

        print(f"💾 Features seleccionadas guardadas en: {output_path}")
        print(f"📌 Total seleccionadas: {len(selected)}")

        return selected

    def train_best_model(self, features_df):
        """
        Entrenar el mejor modelo con configuración optimizada EXACTA.
        Además:
        - Calcula importancia de features
        - Guarda feature_importances.csv
        - Guarda selected_features.json basado en un threshold
        """
        # 1. Preparar datos temporales
        X_train, X_test, y_train, y_test = self.prepare_temporal_data(features_df)

        # 2. Entrenar modelo según configuración
        if self.config['model_type'] == 'xgboost':
            model, train_metrics, test_metrics = self.train_optimized_xgboost(
                X_train, X_test, y_train, y_test
            )
        elif self.config['model_type'] == 'random_forest':
            model, train_metrics, test_metrics = self.train_random_forest_model(
                X_train, X_test, y_train, y_test
            )
        else:
            raise ValueError(f"Tipo de modelo no soportado: {self.config['model_type']}")

        # 3. Análisis y guardado de importancia de features
        print("\n🔍 Analizando importancia de features y guardando artefactos...")
        feature_names = X_train.columns.tolist()
        importance_df = self.feature_importance_analysis(model, feature_names)

        if importance_df is not None:
            threshold = self.config.get('feature_importance_threshold', 0.0001)
            selected_features = self.save_selected_features(importance_df, threshold=threshold)
            print(f"📌 Features finales usadas por el modelo (según threshold {threshold}): {len(selected_features)}")

        # 4. Guardar modelo
        model_path = self.save_model(model, f"{self.config['model_type']}_optimized_model")

        return model, train_metrics, test_metrics, model_path