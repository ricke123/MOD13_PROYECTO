
import argparse
import sys
import os
import pandas as pd

# ============================================================
# CORRECCIÓN DE IMPORTS - USANDO IMPORT RELATIVOS
# ============================================================

# Agregar el directorio padre al path para imports absolutos
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from data_updater import ejecutar_actualizacion_mensual
    from data_loader import DataLoader
    from feature_engineer import FeatureEngineer
    from model_trainer import ModelTrainer
    print("✅ Todos los módulos importados correctamente")
except ImportError as e:
    print(f"❌ Error importando módulos: {e}")
    print("💡 Asegúrate de ejecutar desde la raíz del proyecto: python src/run_retraining.py")
    sys.exit(1)


# ============================================================
# FUNCIONES AUXILIARES PARA MÉTRICAS ORDENADAS
# ============================================================

def print_metrics_ordered(metrics_dict, title):
    """Imprime métricas en orden consistente y formateado"""
    print(f"\n{title}:")
    print("-" * 50)
    print(f"  • MAE:  {metrics_dict['MAE']:>10.2f}")
    print(f"  • RMSE: {metrics_dict['RMSE']:>10.2f}")
    print(f"  • R²:   {metrics_dict['R²']:>10.4f}")
    print(f"  • MAPE: {metrics_dict['MAPE']:>10.2f}%")
    print(f"  • MASE: {metrics_dict['MASE']:>10.4f}")


def print_training_comparison(train_metrics, test_metrics):
    """Muestra comparación entre train y test"""
    print("\n🔍 COMPARACIÓN ENTRENAMIENTO vs TEST:")
    print("-" * 50)
    
    # Calcular diferencias
    r2_gap = train_metrics['R²'] - test_metrics['R²']
    mae_gap = test_metrics['MAE'] - train_metrics['MAE']
    mape_gap = test_metrics['MAPE'] - train_metrics['MAPE']
    
    print(f"  📈 Diferencia R²:     {r2_gap:>10.4f}")
    print(f"  📊 Diferencia MAE:    {mae_gap:>10.2f}")
    print(f"  📉 Diferencia MAPE:   {mape_gap:>10.2f}%")
    
    # Análisis de sobreajuste
    if r2_gap > 0.1:
        print("  ⚠️  ALERTA: Posible sobreajuste detectado")
    else:
        print("  ✅ Buen equilibrio entre train y test")


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

    try:
        updated = ejecutar_actualizacion_mensual()
        if not updated:
            print("❌ ERROR: No se pudieron actualizar los datos. Cancelando.")
            return False
    except Exception as e:
        print(f"❌ ERROR en actualización de datos: {e}")
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

        # ============================================================
        # MÉTRICAS ORDENADAS - PARTE ACTUALIZADA
        # ============================================================
        print("\n" + "🎉" * 20)
        print("🎉 ¡REENTRENAMIENTO INCREMENTAL COMPLETADO EXITOSAMENTE!")
        print("🎉" * 20)
        
        print(f"\n📁 Modelo guardado en: {model_path}")
        
        # Mostrar métricas ordenadas
        print_metrics_ordered(train_metrics, "📈 MÉTRICAS DE ENTRENAMIENTO")
        print_metrics_ordered(test_metrics, "📊 MÉTRICAS DE TEST")
        
        # Mostrar comparación
        print_training_comparison(train_metrics, test_metrics)
        
        print("\n" + "=" * 60)
        print("✅ PROCESO INCREMENTAL FINALIZADO")
        print("=" * 60)

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

        # ============================================================
        # MÉTRICAS ORDENADAS - PARTE ACTUALIZADA
        # ============================================================
        print("\n" + "🎉" * 20)
        print("🎉 ¡REENTRENAMIENTO COMPLETO EXITOSO!")
        print("🎉" * 20)
        
        print(f"\n📁 Modelo guardado en: {model_path}")
        
        # Mostrar métricas ordenadas
        print_metrics_ordered(train_metrics, "📈 MÉTRICAS DE ENTRENAMIENTO")
        print_metrics_ordered(test_metrics, "📊 MÉTRICAS DE TEST")
        
        # Mostrar comparación
        print_training_comparison(train_metrics, test_metrics)
        
        print("\n" + "=" * 60)
        print("✅ PROCESO COMPLETO FINALIZADO")
        print("=" * 60)

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


if __name__ == "__main__":
    main()


