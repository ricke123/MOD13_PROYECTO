# src/smoke_test.py

import os
import sys
from pathlib import Path

# Asegurar que src esté en el path
CURRENT_DIR = Path(__file__).parent
PROJECT_ROOT = CURRENT_DIR.parent
sys.path.append(str(CURRENT_DIR))

from data_loader import DataLoader
from feature_engineer import FeatureEngineer
from model_trainer import ModelTrainer
from feature_correlation import FeatureCorrelationAnalyzer
from config import MODEL_CONFIG


def test_data_loader():
    print("\n TEST 1: DataLoader (raw + processed)")
    loader = DataLoader()

    # Intentar procesados primero
    df_processed = loader.load_processed_data()
    if df_processed is not None:
        print(f" Procesados OK: {df_processed.shape}")
        return df_processed

    print(" No hay procesados, cargando RAW...")
    orders, items, products, reviews, payments = loader.load_raw_data()
    df_clean = loader.clean_data(orders, items, products, reviews, payments)
    loader.save_processed_data(df_clean)
    print(f" Limpieza OK: {df_clean.shape}")
    return df_clean


def test_feature_engineering(df_clean):
    print("\n TEST 2: FeatureEngineer (target + features avanzados)")
    engineer = FeatureEngineer()

    monthly = engineer.create_target_variable(df_clean)
    print(f" monthly_demand: {monthly.shape}")

    monthly_feat = engineer.create_advanced_features(monthly)
    print(f" monthly_demand_with_features: {monthly_feat.shape}")

    features_df = engineer.prepare_model_features(monthly_feat)
    print(f" features_df final: {features_df.shape}")

    # Para que el entrenamiento no explote por tiempo, usamos un subset
    if len(features_df) > 2000:
        features_df = features_df.head(2000).copy()
        print(f" Usando subset de 2000 filas para el smoke test: {features_df.shape}")

    return features_df


def test_feature_correlation(features_df):
    print("\n TEST 3: FeatureCorrelationAnalyzer (selección 99 features)")
    target_col = MODEL_CONFIG.get("target_col", "demand_next_month")

    analyzer = FeatureCorrelationAnalyzer(
        target_col=target_col,
        corr_threshold=0.90,
        top_n=99
    )

    result = analyzer.select_top_features(features_df, save_results=True)
    selected = result["selected_features"]
    print(f" Correlación OK. Features seleccionadas: {len(selected)}")

    # Verifica que se hayan creado archivos de salida
    corr_path = PROJECT_ROOT / "data" / "processed" / "feature_correlation_matrix.csv"
    top_path = PROJECT_ROOT / "data" / "processed" / "top_correlated_features.csv"
    if corr_path.exists():
        print(f" Matriz de correlación guardada en: {corr_path}")
    if top_path.exists():
        print(f" Lista de top features guardada en: {top_path}")

    # Construir DF solo con features seleccionadas + target
    cols_to_keep = [c for c in selected if c in features_df.columns] + [target_col]
    reduced_df = features_df[cols_to_keep].copy()
    print(f" DF reducido para modelo: {reduced_df.shape}")

    return reduced_df


def test_model_trainer(features_df):
    print("\n TEST 4: ModelTrainer (train_best_model con subset)")
    trainer = ModelTrainer()

    model, train_metrics, test_metrics, model_path = trainer.train_best_model(features_df)

    print("\n Entrenamiento OK (smoke test)")
    print(f" Modelo: {trainer.config['model_type']}")
    print(f" R² test:   {test_metrics['R²']:.4f}")
    print(f" MAE test:  {test_metrics['MAE']:.2f}")
    print(f" RMSE test: {test_metrics['RMSE']:.2f}")
    print(f" MAPE test: {test_metrics['MAPE']:.2f}%")
    print(f" Modelo guardado en: {model_path}")

    return model


def main():
    print(" SMOKE TEST OLIST – Pipeline interno")
    print("=" * 60)

    # 1) Carga y limpieza
    df_clean = test_data_loader()

    # 2) Feature engineering
    features_df = test_feature_engineering(df_clean)

    # 3) Selección por correlación (opcional pero útil para validar)
    reduced_df = test_feature_correlation(features_df)

    # 4) Entrenamiento con DF reducido (más rápido)
    test_model_trainer(reduced_df)

    print("\n SMOKE TEST COMPLETADO SIN ERRORES GRAVES")
    print("=" * 60)


if __name__ == "__main__":
    main()
