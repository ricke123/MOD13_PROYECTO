import argparse
import sys
import os
import pandas as pd

# ============================================================
# IMPORTS
# ============================================================

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from data_updater import ejecutar_actualizacion_mensual
    from data_loader import DataLoader
    from feature_engineer import FeatureEngineer
    from model_trainer import ModelTrainer
except ImportError as e:
    print(f"❌ Error importando módulos: {e}")
    sys.exit(1)


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def print_metrics_ordered(metrics_dict, title):
    """Imprime métricas en orden consistente"""
    print(f"\n{title}:")
    print("-" * 50)
    print(f"  • MAE:  {metrics_dict['MAE']:>10.2f}")
    print(f"  • RMSE: {metrics_dict['RMSE']:>10.2f}")
    print(f"  • R²:   {metrics_dict['R²']:>10.4f}")
    print(f"  • MAPE: {metrics_dict['MAPE']:>10.2f}%")
    print(f"  • MASE: {metrics_dict['MASE']:>10.4f}")


def crear_target_manual(monthly_data):
    """Crear target manualmente si es necesario"""
    if 'product_category_name' in monthly_data.columns and 'purchase_year_month' in monthly_data.columns:
        monthly_data = monthly_data.sort_values(['product_category_name', 'purchase_year_month'])
        monthly_data['demand_next_month'] = monthly_data.groupby('product_category_name')['demand'].shift(-1)
        monthly_data = monthly_data.dropna(subset=['demand_next_month'])
    return monthly_data


# ============================================================
# REENTRENAMIENTO INCREMENTAL
# ============================================================

def run_incremental_retraining():
    """Ejecutar reentrenamiento incremental con datos nuevos."""
    print("\n🎯 INICIANDO REENTRENAMIENTO INCREMENTAL")

    # PASO 1: ACTUALIZAR DATOS
    print("\n" + "=" * 60)
    print("📥 PASO 1: ACTUALIZACIÓN DE DATOS")
    print("=" * 60)

    try:
        updated = ejecutar_actualizacion_mensual()
        if not updated:
            print("❌ No se pudieron actualizar los datos")
            return False
    except Exception as e:
        print(f"❌ Error en actualización: {e}")
        return False

    # PASO 2: CARGAR DATOS
    print("\n" + "=" * 60)
    print("📊 PASO 2: CARGA DE DATOS ACTUALIZADOS")
    print("=" * 60)

    data_loader = DataLoader()
    df = data_loader.load_processed_data()

    if df is None or df.empty:
        print("❌ Datos procesados no disponibles")
        return False

    print(f"   Shape: {df.shape}")

    # PASO 3: FEATURE ENGINEERING
    print("\n" + "=" * 60)
    print("🔧 PASO 3: FEATURE ENGINEERING")
    print("=" * 60)

    feature_engineer = FeatureEngineer()
    TARGET = 'demand_next_month'

    try:
        print("Creando variable target...")
        monthly_data = feature_engineer.create_target_variable(df)
        
        # Verificar y crear target si es necesario
        if TARGET not in monthly_data.columns:
            monthly_data = crear_target_manual(monthly_data)

        features_data = feature_engineer.create_advanced_features(monthly_data)
        final_features = feature_engineer.prepare_model_features(features_data)

        if final_features is None or final_features.empty:
            print("❌ No se generaron features válidos")
            return False

        print(f"=== MASTER FINAL ===\nShape: {final_features.shape}")

    except Exception as e:
        print(f"❌ Error en feature engineering: {e}")
        return False

    # PASO 4: ENTRENAMIENTO DEL MODELO
    print("\n" + "=" * 60)
    print("🧠 PASO 4: ENTRENAMIENTO DEL MODELO")
    print("=" * 60)

    model_trainer = ModelTrainer()

    try:
        model, train_metrics, test_metrics, model_path = model_trainer.train_best_model(final_features)

        if model is None:
            print("❌ No se pudo entrenar el modelo")
            return False

        # RESULTADOS FINALES
     
        print("🎉 ¡REENTRENAMIENTO COMPLETADO EXITOSAMENTE!")
      
        
        print(f"\n📁 Modelo guardado en: {model_path}")
        
        print_metrics_ordered(train_metrics, "📈 MÉTRICAS DE ENTRENAMIENTO")
        print_metrics_ordered(test_metrics, "📊 MÉTRICAS DE TEST")
        
        # Análisis de sobreajuste
        r2_gap = train_metrics['R²'] - test_metrics['R²']
        if r2_gap > 0.1:
            print("\n⚠️  ALERTA: Posible sobreajuste detectado")
        else:
            print("\n✅ Buen equilibrio entre train y test")
        
        print("\n" + "=" * 60)
        print("✅ PROCESO INCREMENTAL FINALIZADO")
        print("=" * 60)

        return True

    except Exception as e:
        print(f"❌ Error en entrenamiento: {e}")
        return False


# ============================================================
# REENTRENAMIENTO COMPLETO
# ============================================================

def run_full_retraining():
    """Reentrenar desde cero cargando todos los datos RAW."""
    print("\n🎯 INICIANDO REENTRENAMIENTO COMPLETO")

    # PASO 1: CARGA COMPLETA
    print("\n" + "=" * 60)
    print("📊 PASO 1: CARGA Y LIMPIEZA COMPLETA")
    print("=" * 60)

    data_loader = DataLoader()

    try:
        orders, items, products, reviews, payments = data_loader.load_raw_data()
        df = data_loader.clean_data(orders, items, products, reviews, payments)
        data_loader.save_processed_data(df)
        print(f"   Shape: {df.shape}")
    except Exception as e:
        print(f"❌ Error en carga: {e}")
        return False

    # PASO 2: FEATURE ENGINEERING
    print("\n" + "=" * 60)
    print("🔧 PASO 2: FEATURE ENGINEERING")
    print("=" * 60)

    feature_engineer = FeatureEngineer()

    try:
        print("Creando variable target...")
        monthly_data = feature_engineer.create_target_variable(df)
        features_data = feature_engineer.create_advanced_features(monthly_data)
        final_features = feature_engineer.prepare_model_features(features_data)

        if final_features is None or final_features.empty:
            print("❌ No se generaron features válidos")
            return False

        print(f"=== MASTER FINAL ===\nShape: {final_features.shape}")

    except Exception as e:
        print(f"❌ Error en feature engineering: {e}")
        return False

    # PASO 3: ENTRENAMIENTO
    print("\n" + "=" * 60)
    print("🧠 PASO 3: ENTRENAMIENTO COMPLETO DEL MODELO")
    print("=" * 60)

    model_trainer = ModelTrainer()

    try:
        model, train_metrics, test_metrics, model_path = model_trainer.train_best_model(final_features)

        if model is None:
            print("❌ No se pudo entrenar el modelo")
            return False

        # RESULTADOS FINALES
        print("\n" + "🎉" * 20)
        print("🎉 ¡REENTRENAMIENTO COMPLETO EXITOSO!")
        print("🎉" * 20)
        
        print(f"\n📁 Modelo guardado en: {model_path}")
        
        print_metrics_ordered(train_metrics, "📈 MÉTRICAS DE ENTRENAMIENTO")
        print_metrics_ordered(test_metrics, "📊 MÉTRICAS DE TEST")
        
        print("\n" + "=" * 60)
        print("✅ PROCESO COMPLETO FINALIZADO")
        print("=" * 60)

        return True

    except Exception as e:
        print(f"❌ Error en entrenamiento: {e}")
        return False


# ============================================================
# MAIN
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