"""
Módulo para reentrenamiento - CORREGIDO Y OPTIMIZADO PARA train_best_model
"""

import argparse
import sys
import pandas as pd

from src.data_updater import ejecutar_actualizacion_mensual
from src.data_loader import DataLoader
from src.feature_engineer import FeatureEngineer
from src.model_trainer import ModelTrainer


# ============================================================
# FUNCIONES PRINCIPALES DE REENTRENAMIENTO
# ============================================================

def run_incremental_retraining():
    """Ejecutar reentrenamiento incremental con datos nuevos."""
    print("\n🎯 INICIANDO REENTRENAMIENTO INCREMENTAL")

    # --------------------------------------------------------
    # 1. ACTUALIZAR DATOS
    # --------------------------------------------------------
    print("\n" + "=" * 60)
    print("📥 PASO 1: ACTUALIZACIÓN DE DATOS")
    print("=" * 60)

    updated = ejecutar_actualizacion_mensual()
    if not updated:
        print("❌ ERROR: No se pudieron actualizar los datos. Cancelando.")
        return False

    # --------------------------------------------------------
    # 2. CARGAR DATOS PROCESADOS
    # --------------------------------------------------------
    print("\n" + "=" * 60)
    print("📊 PASO 2: CARGA DE DATOS ACTUALIZADOS")
    print("=" * 60)

    data_loader = DataLoader()
    df = data_loader.load_processed_data()

    if df is None or df.empty:
        print("❌ ERROR: Datos procesados no disponibles o vacíos.")
        return False

    print(f"   ✔ Datos cargados: {df.shape}")

    # --------------------------------------------------------
    # 3. FEATURE ENGINEERING
    # --------------------------------------------------------
    print("\n" + "=" * 60)
    print("🔧 PASO 3: FEATURE ENGINEERING")
    print("=" * 60)

    feature_engineer = FeatureEngineer()

    try:
        monthly_data = feature_engineer.create_target_variable(df)
        features_data = feature_engineer.create_advanced_features(monthly_data)
        final_features = feature_engineer.prepare_model_features(features_data)

    except Exception as e:
        print(f"❌ ERROR en feature engineering: {e}")
        return False

    if final_features is None or final_features.empty:
        print("❌ ERROR: No se generaron features válidos.")
        return False

    print(f"   ✔ Features generados: {final_features.shape}")

    # --------------------------------------------------------
    # 4. REENTRENAMIENTO DEL MODELO
    # --------------------------------------------------------
    print("\n" + "=" * 60)
    print("🧠 PASO 4: REENTRENAMIENTO DEL MODELO")
    print("=" * 60)

    model_trainer = ModelTrainer()

    try:
        model, train_metrics, test_metrics, model_path = model_trainer.train_best_model(final_features)

        if model is None:
            print("❌ FAIL: train_best_model no devolvió un modelo válido.")
            return False

        print("\n🎉 ¡Reentrenamiento incremental completado exitosamente!")
        print(f"📁 Modelo guardado en: {model_path}")
        print(f"📈 Métricas train: {train_metrics}")
        print(f"📈 Métricas test: {test_metrics}")

        return True

    except Exception as e:
        print(f"❌ ERROR en reentrenamiento incremental: {e}")
        return False



# ============================================================
# REENTRENAMIENTO COMPLETO
# ============================================================

def run_full_retraining():
    """Reentrenar desde cero cargando todos los datos RAW."""
    print("\n🎯 INICIANDO REENTRENAMIENTO COMPLETO")

    # --------------------------------------------------------
    # 1. CARGA Y PROCESAMIENTO COMPLETO
    # --------------------------------------------------------
    print("\n" + "=" * 60)
    print("📊 PASO 1: CARGA Y LIMPIEZA COMPLETA")
    print("=" * 60)

    data_loader = DataLoader()

    try:
        orders, items, products, reviews, payments = data_loader.load_raw_data()
        df = data_loader.clean_data(orders, items, products, reviews, payments)
        data_loader.save_processed_data(df)

    except Exception as e:
        print(f"❌ ERROR en carga/procesamiento desde cero: {e}")
        return False

    print(f"   ✔ Dataset procesado: {df.shape}")

    # --------------------------------------------------------
    # 2. FEATURE ENGINEERING
    # --------------------------------------------------------
    print("\n" + "=" * 60)
    print("🔧 PASO 2: FEATURE ENGINEERING")
    print("=" * 60)

    feature_engineer = FeatureEngineer()

    try:
        monthly_data = feature_engineer.create_target_variable(df)
        features_data = feature_engineer.create_advanced_features(monthly_data)
        final_features = feature_engineer.prepare_model_features(features_data)

    except Exception as e:
        print(f"❌ ERROR en feature engineering: {e}")
        return False

    if final_features is None or final_features.empty:
        print("❌ ERROR: No se generaron features válidos.")
        return False

    print(f"   ✔ Features generados: {final_features.shape}")

    # --------------------------------------------------------
    # 3. REENTRENAR MODELO COMPLETO
    # --------------------------------------------------------
    print("\n" + "=" * 60)
    print("🧠 PASO 3: REENTRENAMIENTO COMPLETO DEL MODELO")
    print("=" * 60)

    model_trainer = ModelTrainer()

    try:
        model, train_metrics, test_metrics, model_path = model_trainer.train_best_model(final_features)

        if model is None:
            print("❌ FAIL: train_best_model no devolvió un modelo válido.")
            return False

        print("\n🎉 Reentrenamiento COMPLETO exitoso")
        print(f"📁 Modelo guardado en: {model_path}")
        print(f"📈 Métricas train: {train_metrics}")
        print(f"📈 Métricas test: {test_metrics}")

        return True

    except Exception as e:
        print(f"❌ ERROR en reentrenamiento completo: {e}")
        return False



# ============================================================
# MAIN CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Sistema de Reentrenamiento OLIST")
    parser.add_argument('--incremental', action='store_true', help='Reentrenamiento con datos nuevos')
    parser.add_argument('--full', action='store_true', help='Reentrenamiento desde cero')

    args = parser.parse_args()

    print("🚀 SISTEMA DE REENTRENAMIENTO - OLIST FORECASTING")
    print("=" * 60)

    if not args.incremental and not args.full:
        print("❌ Debes especificar --incremental o --full")
        sys.exit(1)

    if args.incremental:
        success = run_incremental_retraining()
    elif args.full:
        success = run_full_retraining()

    print("\n" + "=" * 60)
    if success:
        print("🎉 PROCESO COMPLETADO EXITOSAMENTE")
    else:
        print("💥 PROCESO TERMINADO CON ERRORES")
    print("=" * 60)

    return success


# Compatibilidad con el scheduler
if __name__ == "__main__":
    main()
