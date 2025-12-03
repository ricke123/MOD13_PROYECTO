#!/usr/bin/env python3
"""
Punto de entrada principal del proyecto Olist – Pipeline + Reentrenamiento + Scheduler + Benchmark
"""

import argparse
import sys
import os

# Añadir el directorio actual al path para importaciones
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# También añadir el directorio padre
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

print(f"📂 Directorio actual: {current_dir}")
print(f"📂 Directorio padre: {parent_dir}")

# IMPORTACIONES ROBUSTAS
try:
    # Intentar importar desde src.
    from src.data_loader import DataLoader
    from src.feature_engineer import FeatureEngineer
    from src.model_trainer import ModelTrainer
    from src.run_retraining import run_incremental_retraining, run_full_retraining
    from src.retraining_scheduler import RetrainingScheduler
    from src.feature_correlation import FeatureCorrelationAnalyzer
    from src.config import MODEL_CONFIG, print_config_summary
    print("✅ Módulos importados desde src/")
    
except ImportError as e:
    print(f"⚠️  Error importando desde src/: {e}")
    print("🔄 Intentando importación directa...")
    
    # Intentar importación directa (para cuando se ejecuta como módulo)
    try:
        from data_loader import DataLoader
        from feature_engineer import FeatureEngineer
        from model_trainer import ModelTrainer
        from run_retraining import run_incremental_retraining, run_full_retraining
        from retraining_scheduler import RetrainingScheduler
        from feature_correlation import FeatureCorrelationAnalyzer
        from config import MODEL_CONFIG, print_config_summary
        print("✅ Módulos importados directamente")
    except ImportError as e2:
        print(f"❌ Error en importación directa: {e2}")
        print("💡 Soluciones posibles:")
        print("   1. Ejecuta desde la raíz del proyecto: python -m src.main")
        print("   2. O usa run_pipeline.py: python run_pipeline.py")
        sys.exit(1)

# IMPORTACIÓN DEL BENCHMARK
try:
    from src.model_benchmark import run_benchmark
    BENCHMARK_MODULE = "model_benchmark"
    print("✅ Benchmark importado desde src/model_benchmark.py")
except ImportError as e:
    try:
        from src.model_benchmarking import main as run_benchmark
        BENCHMARK_MODULE = "model_benchmarking"
        print("✅ Benchmark importado desde src/model_benchmarking.py")
    except ImportError:
        try:
            # Último intento: importación directa
            from model_benchmark import run_benchmark
            BENCHMARK_MODULE = "model_benchmark"
            print("✅ Benchmark importado directamente")
        except ImportError:
            print("⚠️  No se pudo importar módulo de benchmark")
            print("   El comando --benchmark no funcionará")
            run_benchmark = None
            BENCHMARK_MODULE = None


# ---------------------------------------------------------------------
# 1. ORQUESTADOR DEL PIPELINE COMPLETO
# ---------------------------------------------------------------------
def run_pipeline(use_corr_features: bool = False):
    """
    Ejecutar pipeline completo de ML: datos → features → modelo
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
            print("📂 No hay datos procesados. Cargando datos RAW...")
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

        print(f"📊 Dataset final para modelo: {features_df.shape}")

        # 2.1️⃣ OPCIONAL: selección de features por correlación
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

            cols_to_keep = selected_features + [target_col]
            if "purchase_year_month" in features_df.columns:
                cols_to_keep.append("purchase_year_month")

            cols_to_keep = [c for c in dict.fromkeys(cols_to_keep) if c in features_df.columns]
            features_df = features_df[cols_to_keep].copy()
            print(f"📊 Dataset con features seleccionadas: {features_df.shape}")

        # 3️⃣ ENTRENAMIENTO DEL MEJOR MODELO
        print("\n3️⃣  ETAPA: ENTRENAMIENTO DEL MODELO")
        trainer = ModelTrainer()
        model, train_score, test_score, model_path = trainer.train_best_model(features_df)

        # 4️⃣ RESUMEN FINAL
        print("\n🎉 PIPELINE COMPLETADO EXITOSAMENTE")
        print("=" * 60)
        print(f"📈 Modelo final: {MODEL_CONFIG.get('model_type', 'desconocido').upper()}")
        print(f"📊 R² (test):   {test_score.get('R²', 'N/A'):.4f}")
        print(f"📊 MAE (test):  {test_score.get('MAE', 'N/A'):.2f}")
        print(f"📊 RMSE (test): {test_score.get('RMSE', 'N/A'):.2f}")
        print(f"📊 MAPE (test): {test_score.get('MAPE', 'N/A'):.2f}%")
        print(f"📁 Modelo guardado en: {model_path}")
        print("=" * 60)

        return model, train_score, test_score

    except Exception as e:
        print(f"❌ Error en el pipeline: {e}")
        import traceback
        traceback.print_exc()
        raise


# ---------------------------------------------------------------------
# 2. FUNCIÓN PARA BENCHMARK
# ---------------------------------------------------------------------
def run_benchmark_mode(use_corr_features=False):
    """Ejecutar benchmark de modelos"""
    if run_benchmark is None:
        print("❌ Benchmark no disponible (módulo no encontrado)")
        return None
    
    print(f"\n🏁 EJECUTANDO BENCHMARK DE MODELOS ({BENCHMARK_MODULE})")
    print("=" * 60)
    
    # Mostrar configuración actual si está disponible
    try:
        print_config_summary()
    except:
        print(f"📊 Modelo activo: {MODEL_CONFIG.get('model_type', 'desconocido')}")
    
    # Cargar datos
    loader = DataLoader()
    processed_data = loader.load_processed_data()
    
    if processed_data is None:
        print("📂 Cargando datos RAW...")
        orders, items, products, reviews, payments = loader.load_raw_data()
        df_clean = loader.clean_data(orders, items, products, reviews, payments)
        loader.save_processed_data(df_clean)
    else:
        df_clean = processed_data
    
    # Ingeniería de features
    engineer = FeatureEngineer()
    monthly_demand = engineer.create_target_variable(df_clean)
    monthly_demand_with_features = engineer.create_advanced_features(monthly_demand)
    features_df = engineer.prepare_model_features(monthly_demand_with_features)
    
    print(f"📊 Dataset original: {features_df.shape}")
    
    # Aplicar selección por correlación si se solicita
    if use_corr_features:
        print("\n🧮 Aplicando selección de features por correlación...")
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
        print(f"📊 Dataset con features seleccionadas: {features_df.shape}")
    
    # Ejecutar benchmark
    print("\n🔍 Ejecutando benchmark...")
    try:
        results = run_benchmark(features_df)
        print("\n✅ BENCHMARK COMPLETADO")
        return results
    except Exception as e:
        print(f"❌ Error en el benchmark: {e}")
        import traceback
        traceback.print_exc()
        return None


# ---------------------------------------------------------------------
# 3. CLI PRINCIPAL
# ---------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Sistema de Forecasting Olist",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python run_pipeline.py                    # Ejecuta pipeline completo
  python -m src.main --pipeline            # Pipeline desde módulo
  python -m src.main --benchmark           # Benchmark de modelos
  python -m src.main --config-summary      # Ver configuración
  
Cambiar modelo en config.py:
  MODEL_CONFIG['model_type'] = 'random_forest'  # o 'xgboost'
        """
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
        help="Seleccionar hasta 99 features por correlación antes de entrenar.",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Ejecutar benchmark de modelos (Naive, RF, XGB base, XGB tunado, CatBoost).",
    )
    parser.add_argument(
        "--config-summary",
        action="store_true",
        help="Mostrar resumen de configuración y salir.",
    )
    
    return parser.parse_args()


def main():
    print("🚀 SISTEMA PRINCIPAL – OLIST DEMAND FORECASTING")
    print("=" * 60)

    args = parse_args()

    # Mostrar resumen de configuración
    if args.config_summary:
        try:
            print_config_summary()
        except:
            print("ℹ️  Función print_config_summary no disponible")
        return

    # Si no se pasa ningún flag → ejecutamos pipeline por defecto
    if not any([args.pipeline, args.retrain_incremental, 
                args.retrain_full, args.scheduler, args.benchmark]):
        print("ℹ️  No se especificó opción. Ejecutando pipeline completo por defecto.\n")
        run_pipeline(use_corr_features=args.use_corr_features)
        return

    # 1) Pipeline completo
    if args.pipeline:
        run_pipeline(use_corr_features=args.use_corr_features)

    # 2) Benchmark de modelos
    if args.benchmark:
        results = run_benchmark_mode(use_corr_features=args.use_corr_features)
        if results is not None:
            print("\n📊 RESULTADOS DEL BENCHMARK:")
            print("=" * 60)
            print(results)

    # 3) Reentrenamiento incremental
    if args.retrain_incremental:
        print("\n🔄 MODO: Reentrenamiento incremental")
        success = run_incremental_retraining()
        print("✅ Incremental OK" if success else "💥 Error en reentrenamiento incremental")

    # 4) Reentrenamiento completo
    if args.retrain_full:
        print("\n🔄 MODO: Reentrenamiento completo")
        success = run_full_retraining()
        print("✅ Completo OK" if success else "💥 Error en reentrenamiento completo")

    # 5) Scheduler automático
    if args.scheduler:
        print("\n⏰ MODO: Scheduler automático de reentrenamiento")
        scheduler = RetrainingScheduler()
        scheduler.run_scheduler()


if __name__ == "__main__":
    main()