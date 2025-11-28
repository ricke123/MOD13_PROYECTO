#!/usr/bin/env python3
"""
Punto de entrada principal del proyecto Olist – Pipeline + Reentrenamiento + Scheduler + Benchmark
"""

import argparse

from src.data_loader import DataLoader
from src.feature_engineer import FeatureEngineer
from src.model_trainer import ModelTrainer
from src.run_retraining import run_incremental_retraining, run_full_retraining
from src.retraining_scheduler import RetrainingScheduler
from src.feature_correlation import FeatureCorrelationAnalyzer
from src.config import MODEL_CONFIG
from src.model_benchmark import run_benchmark  # 👈 benchmark de modelos


# ---------------------------------------------------------------------
# 1. ORQUESTADOR DEL PIPELINE COMPLETO
# ---------------------------------------------------------------------
def run_pipeline(use_corr_features: bool = False):
    """
    Ejecutar pipeline completo de ML: datos → features → modelo

    Parámetros
    ----------
    use_corr_features : bool
        Si True, antes de entrenar selecciona hasta 99 features poco colineales
        basadas en correlación con la target y entrena solo con esas.
    """
    print("🚀 INICIANDO PIPELINE DE MACHINE LEARNING")
    print("=" * 60)

    try:
        # 1️⃣ CARGA Y LIMPIEZA DE DATOS
        print("\n1️⃣  ETAPA: CARGA Y LIMPIEZA DE DATOS")
        loader = DataLoader()

        # Intentar cargar datos procesados primero
        processed_data = loader.load_processed_data()

        if processed_data is None:
            ##print("📂 No hay datos procesados. Cargando datos RAW...")
            orders, items, products, reviews, payments = loader.load_raw_data()
            df_clean = loader.clean_data(orders, items, products, reviews, payments)
            loader.save_processed_data(df_clean)
        else:
            print("📂 Usando datos procesados existentes.")
            df_clean = processed_data

        # 2️⃣ INGENIERÍA DE FEATURES
        print("\n2️⃣  ETAPA: INGENIERÍA DE FEATURES")
        engineer = FeatureEngineer()

        # Crear variable target mensual y features avanzados
        monthly_demand = engineer.create_target_variable(df_clean)
        monthly_demand_with_features = engineer.create_advanced_features(monthly_demand)
        features_df = engineer.prepare_model_features(monthly_demand_with_features)

        print(f"📊 Dataset final para modelo (antes de correlación): {features_df.shape}")

        # 2.1️⃣ OPCIONAL: selección de hasta 99 features por correlación con la target
        if use_corr_features:
            print("\n🧮 ETAPA OPCIONAL: SELECCIÓN DE FEATURES POR CORRELACIÓN")
            target_col = MODEL_CONFIG.get("target_col", "demand_next_month")

            analyzer = FeatureCorrelationAnalyzer(
                target_col=target_col,
                corr_threshold=0.90,
                top_n=99,
            )

            corr_result = analyzer.select_top_features(features_df, save_results=True)
            selected_features = corr_result["selected_features"]

            # Nos aseguramos de incluir también la target y la fecha
            cols_to_keep = selected_features + [target_col]

            if "purchase_year_month" in features_df.columns:
                cols_to_keep.append("purchase_year_month")

            # Evitar columnas inexistentes / duplicadas
            cols_to_keep = [c for c in dict.fromkeys(cols_to_keep) if c in features_df.columns]

            features_df = features_df[cols_to_keep].copy()
            print(f"📊 Dataset para modelo (SOLO features seleccionadas por correlación): {features_df.shape}")

        # 3️⃣ ENTRENAMIENTO DEL MEJOR MODELO (según MODEL_CONFIG)
        print("\n3️⃣  ETAPA: ENTRENAMIENTO DEL MODELO")
        trainer = ModelTrainer()
        model, train_score, test_score, model_path = trainer.train_best_model(features_df)

        # 4️⃣ RESUMEN FINAL
        print("\n🎉 PIPELINE COMPLETADO EXITOSAMENTE")
        print("=" * 60)
        print(f"📈 Modelo final: {trainer.config['model_type']}")
        print(f"📊 R² (test):   {test_score['R²']:.4f}")
        print(f"📊 MAE (test):  {test_score['MAE']:.2f}")
        print(f"📊 RMSE (test): {test_score['RMSE']:.2f}")
        print(f"📊 MAPE (test): {test_score['MAPE']:.2f}%")
        print(f"📁 Modelo guardado en: {model_path}")
        print("=" * 60)

        return model, train_score, test_score

    except Exception as e:
        print(f"❌ Error en el pipeline: {e}")
        raise


# ---------------------------------------------------------------------
# 2. CLI PRINCIPAL
# ---------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Sistema de Forecasting Olist – ejecución de pipeline, "
                    "reentrenamiento, scheduler y benchmark."
    )
    parser.add_argument(
        "--pipeline",
        action="store_true",
        help="Ejecutar pipeline completo (datos → features → modelo).",
    )
    parser.add_argument(
        "--retrain-incremental",
        action="store_true",
        help="Reentrenamiento incremental usando datos nuevos.",
    )
    parser.add_argument(
        "--retrain-full",
        action="store_true",
        help="Reentrenamiento completo desde cero con todos los datos RAW.",
    )
    parser.add_argument(
        "--scheduler",
        action="store_true",
        help="Iniciar scheduler de reentrenamiento automático.",
    )
    parser.add_argument(
        "--use-corr-features",
        action="store_true",
        help=(
            "Antes de entrenar, seleccionar hasta 99 features por correlación "
            "con la target y usar solo esas para el modelo."
        ),
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Ejecutar benchmark de modelos (Naive, RF, XGB base, XGB tunado, CatBoost tunado).",
    )
    return parser.parse_args()


def main():
    print("🚀 SISTEMA PRINCIPAL – OLIST DEMAND FORECASTING")
    print("=" * 60)

    args = parse_args()

    # Si no se pasa ningún flag → ejecutamos pipeline por defecto
    if not (
        args.pipeline
        or args.retrain_incremental
        or args.retrain_full
        or args.scheduler
        or args.benchmark
    ):
        print("ℹ️  No se especificó opción. Ejecutando pipeline completo por defecto.\n")
        run_pipeline(use_corr_features=args.use_corr_features)
        return

    # 1) Pipeline completo
    if args.pipeline:
        run_pipeline(use_corr_features=args.use_corr_features)

    # 2) Reentrenamiento incremental
    if args.retrain_incremental:
        print("\n🔄 MODO: Reentrenamiento incremental")
        success = run_incremental_retraining()
        print("✅ Incremental OK" if success else "💥 Error en reentrenamiento incremental")

    # 3) Reentrenamiento completo
    if args.retrain_full:
        print("\n🔄 MODO: Reentrenamiento completo")
        success = run_full_retraining()
        print("✅ Completo OK" if success else "💥 Error en reentrenamiento completo")

    # 4) Scheduler automático
    if args.scheduler:
        print("\n⏰ MODO: Scheduler automático de reentrenamiento")
        scheduler = RetrainingScheduler()
        scheduler.run_scheduler()

    # 5) Benchmark de modelos
    if args.benchmark:
        print("\n🏁 MODO: Benchmark de modelos")
        loader = DataLoader()

        processed_data = loader.load_processed_data()
        if processed_data is None:
            print("📂 No hay datos procesados. Cargando datos RAW...")
            orders, items, products, reviews, payments = loader.load_raw_data()
            df_clean = loader.clean_data(orders, items, products, reviews, payments)
            loader.save_processed_data(df_clean)
        else:
            print("📂 Usando datos procesados existentes.")
            df_clean = processed_data

        engineer = FeatureEngineer()
        monthly_demand = engineer.create_target_variable(df_clean)
        monthly_demand_with_features = engineer.create_advanced_features(monthly_demand)
        features_df = engineer.prepare_model_features(monthly_demand_with_features)

        # Opcionalmente también aplicar selección por correlación al benchmark
        if args.use_corr_features:
            print("\n🧮 Aplicando selección de features por correlación para el benchmark...")
            target_col = MODEL_CONFIG.get("target_col", "demand_next_month")
            analyzer = FeatureCorrelationAnalyzer(
                target_col=target_col,
                corr_threshold=0.90,
                top_n=99,
            )
            corr_result = analyzer.select_top_features(features_df, save_results=False)
            selected_features = corr_result["selected_features"]

            cols_to_keep = selected_features + [target_col]
            if "purchase_year_month" in features_df.columns:
                cols_to_keep.append("purchase_year_month")
            cols_to_keep = [c for c in dict.fromkeys(cols_to_keep) if c in features_df.columns]

            features_df = features_df[cols_to_keep].copy()
            print(f"📊 Dataset para benchmark (con correlación): {features_df.shape}")
        else:
            print(f"📊 Dataset para benchmark (sin selección por correlación): {features_df.shape}")

        benchmark_results = run_benchmark(features_df)

        print("\n✅ Benchmark finalizado. Resultados:")
        print(benchmark_results)


if __name__ == "__main__":
    main()
