#!/usr/bin/env python3
"""
Benchmarking de modelos:
Naive vs RandomForest vs XGBoost (base) vs XGBoost (tunado) vs CatBoost (tunado)
"""

import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

import xgboost as xgb
from catboost import CatBoostRegressor

# Importar módulos del proyecto
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR))
sys.path.append(PROJECT_ROOT)

from data_loader import DataLoader
from feature_engineer import FeatureEngineer
from model_trainer import ModelTrainer
from config import MODEL_CONFIG, TIME_PERIODS, DATA_PROCESSED


def temporal_train_test_split(df: pd.DataFrame,
                              target_col: str = "demand_next_month"):
    """
    Split temporal EXACTO como en el EDA:
    - Train: 2016-09 a 2018-04
    - Test:  2018-05 a 2018-07
    Usamos la columna 'purchase_year_month' que viene del FeatureEngineer.
    """

    print("⏱️  Realizando split temporal EXACTO para benchmarking...")

    if 'purchase_year_month' not in df.columns:
        raise ValueError("El DataFrame necesita la columna 'purchase_year_month' para el split temporal.")

    # Asegurar tipo datetime
    if not np.issubdtype(df['purchase_year_month'].dtype, np.datetime64):
        df['purchase_year_month'] = pd.to_datetime(df['purchase_year_month'].astype(str))

    train_start = pd.to_datetime(TIME_PERIODS['train_start'])
    train_end = pd.to_datetime(TIME_PERIODS['train_end'])
    test_start = pd.to_datetime(TIME_PERIODS['test_start'])
    test_end = pd.to_datetime(TIME_PERIODS['test_end'])

    print(f"📅 Train: {train_start.date()} → {train_end.date()}")
    print(f"📅 Test : {test_start.date()} → {test_end.date()}")

    mask_train = (df['purchase_year_month'] >= train_start) & (df['purchase_year_month'] <= train_end)
    mask_test = (df['purchase_year_month'] >= test_start) & (df['purchase_year_month'] <= test_end)

    train_df = df[mask_train].copy()
    test_df = df[mask_test].copy()

    print(f"📊 Train shape: {train_df.shape}")
    print(f"📊 Test shape : {test_df.shape}")

    # Separar X, y. Dejamos 'demand' en X_test para Naive.
    drop_cols = ['purchase_year_month', 'product_category_name', target_col]

    X_train = train_df.drop(columns=drop_cols, errors='ignore')
    X_test = test_df.drop(columns=drop_cols, errors='ignore')

    y_train = train_df[target_col].values
    y_test = test_df[target_col].values

    # Guardamos también la demanda actual para Naive
    demand_test = test_df['demand'].values

    # Manejo de NaN/inf
    for X in (X_train, X_test):
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        X.fillna(0, inplace=True)

    return X_train, X_test, y_train, y_test, demand_test


def train_naive_baseline(y_test, demand_test, y_train, metrics_fn):
    """
    Naive: predicción = demanda del mes actual (columna demand).
    Como la target es demanda_next_month, esto replica el Naive del EDA.
    """
    print("\n🟡 Entrenando modelo Naive (baseline)...")
    y_pred_test = demand_test.copy()

    # Para train, el Naive sería demanda_next_month ≈ demanda actual también,
    # pero para métricas de comparación nos enfocamos en test.
    metrics_test = metrics_fn(y_test, y_pred_test, y_train)

    # Train metrics (opcional: usar mismos valores que test o construir un Naive simétrico)
    metrics_train = {k: np.nan for k in metrics_test.keys()}
    return metrics_train, metrics_test


def train_random_forest(X_train, X_test, y_train, y_test, metrics_fn):
    print("\n🌲 Entrenando Random Forest...")
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)

    metrics_test = metrics_fn(y_test, y_pred_test, y_train)
    metrics_train = metrics_fn(y_train, y_pred_train, y_train)
    return model, metrics_train, metrics_test


def train_xgb_baseline(X_train, X_test, y_train, y_test, metrics_fn):
    print("\n🔵 Entrenando XGBoost baseline...")
    params = {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": -1,
        "tree_method": "hist"
    }
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)

    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)

    metrics_test = metrics_fn(y_test, y_pred_test, y_train)
    metrics_train = metrics_fn(y_train, y_pred_train, y_train)
    return model, metrics_train, metrics_test


def train_xgb_tuned(X_train, X_test, y_train, y_test, metrics_fn):
    print("\n🟣 Entrenando XGBoost tunado (parámetros del EDA)...")

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

    model = xgb.XGBRegressor(
        **best_params,
        random_state=42,
        n_jobs=-1,
        tree_method='hist',
        early_stopping_rounds=50,
        eval_metric='mae'
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)

    metrics_test = metrics_fn(y_test, y_pred_test, y_train)
    metrics_train = metrics_fn(y_train, y_pred_train, y_train)
    return model, metrics_train, metrics_test


def train_catboost_tuned(X_train, X_test, y_train, y_test, metrics_fn):
    print("\n🐱 Entrenando CatBoost tunado...")
    model = CatBoostRegressor(
        depth=8,
        learning_rate=0.05,
        iterations=500,
        loss_function='RMSE',
        random_seed=42,
        verbose=False
    )
    model.fit(X_train, y_train)

    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)

    metrics_test = metrics_fn(y_test, y_pred_test, y_train)
    metrics_train = metrics_fn(y_train, y_pred_train, y_train)
    return model, metrics_train, metrics_test


def build_results_table(results_dict):
    """
    Convierte el diccionario:
    {
      'Naive': {'train': {...}, 'test': {...}},
      'RandomForest': {...},
      ...
    }
    en un DataFrame con columnas:
    modelo, MAE_train, RMSE_train, ..., MAE_test, ...
    """
    rows = []
    for model_name, metrics in results_dict.items():
        row = {"modelo": model_name}
        train_m = metrics.get("train", {})
        test_m = metrics.get("test", {})

        for k, v in train_m.items():
            row[f"{k}_train"] = v
        for k, v in test_m.items():
            row[f"{k}_test"] = v

        rows.append(row)

    df_results = pd.DataFrame(rows)
    return df_results


def main():
    print("🚀 BENCHMARKING DE MODELOS - SPRINT 3")
    print("=" * 80)

    # 1. Carga de datos y feature engineering (igual que en pipeline)
    loader = DataLoader()
    processed_df = loader.load_processed_data()
    if processed_df is None:
        print("📥 No se encontraron datos procesados. Cargando desde raw...")
        orders, items, products, reviews, payments = loader.load_raw_data()
        df_clean = loader.clean_data(orders, items, products, reviews, payments)
        loader.save_processed_data(df_clean)
    else:
        df_clean = processed_df

    engineer = FeatureEngineer()
    monthly_data = engineer.create_target_variable(df_clean)
    monthly_with_features = engineer.create_advanced_features(monthly_data)

    # No aplicamos selección de features de XGBoost aquí,
    # solo limpieza básica para mantener TODAS las columnas.
    target_col = MODEL_CONFIG.get("target_col", "demand_next_month")

    df_full = monthly_with_features.copy()
    # Limpieza de NaN / inf
    df_full.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_full = df_full.dropna(subset=[target_col])
    df_full = df_full.fillna(0)

    # 2. Split temporal EXACTO
    X_train, X_test, y_train, y_test, demand_test = temporal_train_test_split(
        df_full,
        target_col=target_col
    )

    # 3. Preparar función de métricas desde ModelTrainer
    trainer = ModelTrainer()
    metrics_fn = trainer.calculate_comprehensive_metrics

    # 4. Entrenar y evaluar todos los modelos
    results = {}

    # 4.1 Naive
    naive_train_m, naive_test_m = train_naive_baseline(
        y_test, demand_test, y_train, metrics_fn
    )
    results["Naive"] = {"train": naive_train_m, "test": naive_test_m}

    # 4.2 Random Forest
    rf_model, rf_train_m, rf_test_m = train_random_forest(
        X_train, X_test, y_train, y_test, metrics_fn
    )
    results["RandomForest"] = {"train": rf_train_m, "test": rf_test_m}

    # 4.3 XGBoost baseline
    xgb_base, xgb_base_train_m, xgb_base_test_m = train_xgb_baseline(
        X_train, X_test, y_train, y_test, metrics_fn
    )
    results["XGBoost_Base"] = {"train": xgb_base_train_m, "test": xgb_base_test_m}

    # 4.4 XGBoost tunado
    xgb_tuned, xgb_tuned_train_m, xgb_tuned_test_m = train_xgb_tuned(
        X_train, X_test, y_train, y_test, metrics_fn
    )
    results["XGBoost_Tuned"] = {"train": xgb_tuned_train_m, "test": xgb_tuned_test_m}

    # 4.5 CatBoost tunado
    cat_model, cat_train_m, cat_test_m = train_catboost_tuned(
        X_train, X_test, y_train, y_test, metrics_fn
    )
    results["CatBoost_Tuned"] = {"train": cat_train_m, "test": cat_test_m}

    # 5. Construir tabla de resultados
    df_results = build_results_table(results)

    print("\n📊 RESULTADOS RESUMIDOS (SÚPER ÚTIL PARA LAS DIAPOS):")
    print(df_results.round(4))

    # 6. Guardar en data/processed
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    out_path = DATA_PROCESSED / "model_benchmark_sprint3.csv"
    df_results.to_csv(out_path, index=False)
    print(f"\n💾 Resultados de benchmarking guardados en: {out_path}")

    print("\n✅ Benchmarking completado.")
    print("=" * 80)


if __name__ == "__main__":
    main()
