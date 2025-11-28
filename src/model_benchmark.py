# src/model_benchmark.py
"""
Benchmark de modelos para forecasting Olist.

Compara:
- Naive (último valor)
- RandomForest
- XGBoost base
- XGBoost Tunado (hiperparámetros del EDA)
- CatBoost Tunado

Devuelve un DataFrame con las métricas por modelo.
"""

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False


# ---------------------------------------------------------------------
# Función auxiliar para métricas
# ---------------------------------------------------------------------
def compute_metrics(y_true, y_pred):
    """Calcula MAE, RMSE, R² y MAPE."""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)   # sin squared=False
    rmse = np.sqrt(mse)                        # RMSE a mano
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R²": r2,
        "MAPE": mape,
    }


# ---------------------------------------------------------------------
# Benchmark principal
# ---------------------------------------------------------------------
def run_benchmark(features_df: pd.DataFrame, target_col: str = "demand_next_month"):
    """
    Ejecuta el benchmark de varios modelos sobre el mismo train/test split.

    Parameters
    ----------
    features_df : pd.DataFrame
        DataFrame final con features + columna target.
    target_col : str
        Nombre de la columna target.

    Returns
    -------
    results_df : pd.DataFrame
        DataFrame con métricas por modelo.
    """
    print("🏁 INICIANDO BENCHMARK DE MODELOS")
    print("============================================================")

    if target_col not in features_df.columns:
        raise ValueError(f"❌ La columna target '{target_col}' no está en el DataFrame.")

    # -----------------------------
    # Separar X e y
    # -----------------------------
    X = features_df.drop(columns=[target_col]).copy()
    y = features_df[target_col].copy()

    # 🔧 MUY IMPORTANTE:
    # Nos quedamos solo con columnas numéricas para evitar tipos Period/datetime
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X = X[numeric_cols].copy()

    print(f"📊 Features numéricas usadas en el benchmark: {list(X.columns)}")

    # Split aleatorio (el benchmark no necesita split temporal)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"📊 Train shape: {X_train.shape}")
    print(f"📊 Test  shape: {X_test.shape}")
    print("============================================================")

    results = []

    # ---------------------------------------------------------
    # 1) Modelo Naive (último valor observado)
    # ---------------------------------------------------------
    print("📌 MODELO 1: Naive (último valor)")
    last_value = y_train.iloc[-1]
    naive_pred = np.full_like(y_test, fill_value=last_value, dtype=float)

    naive_metrics = compute_metrics(y_test, naive_pred)
    results.append({"Modelo": "Naive (último valor)", **naive_metrics})

    # ---------------------------------------------------------
    # 2) RandomForest
    # ---------------------------------------------------------
    print("\n🌲 MODELO 2: RandomForestRegressor")
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)

    rf_metrics = compute_metrics(y_test, rf_pred)
    results.append({"Modelo": "RandomForest", **rf_metrics})

    # ---------------------------------------------------------
    # 3) XGBoost base
    # ---------------------------------------------------------
    print("\n💻 MODELO 3: XGBoost (base)")
    xgb_base = XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )
    xgb_base.fit(X_train, y_train)
    xgb_base_pred = xgb_base.predict(X_test)

    xgb_base_metrics = compute_metrics(y_test, xgb_base_pred)
    results.append({"Modelo": "XGBoost Base", **xgb_base_metrics})

    # ---------------------------------------------------------
    # 4) XGBoost tunado (hiperparámetros del EDA)
    # ---------------------------------------------------------
    print("\n🚀 MODELO 4: XGBoost Tunado (EDA)")
    xgb_tuned = XGBRegressor(
        n_estimators=294,
        max_depth=10,
        learning_rate=0.0348,
        min_child_weight=7,
        gamma=0.1788,
        subsample=0.6375,
        colsample_bytree=0.7976,
        colsample_bylevel=0.6072,
        reg_alpha=0.0882,
        reg_lambda=0.0257,
        random_state=42,
        n_jobs=-1,
    )
    xgb_tuned.fit(X_train, y_train)
    xgb_tuned_pred = xgb_tuned.predict(X_test)

    xgb_tuned_metrics = compute_metrics(y_test, xgb_tuned_pred)
    results.append({"Modelo": "XGBoost Tunado", **xgb_tuned_metrics})

    # ---------------------------------------------------------
    # 5) CatBoost Tunado (si está instalado)
    # ---------------------------------------------------------
    if HAS_CATBOOST:
        print("\n🐱 MODELO 5: CatBoost Tunado")
        cat = CatBoostRegressor(
            depth=8,
            learning_rate=0.05,
            n_estimators=500,
            loss_function="RMSE",
            verbose=False,
            random_seed=42,
        )
        cat.fit(X_train, y_train)
        cat_pred = cat.predict(X_test)

        cat_metrics = compute_metrics(y_test, cat_pred)
        results.append({"Modelo": "CatBoost Tunado", **cat_metrics})
    else:
        print("\n⚠️ CatBoost no está instalado. Se omite este modelo.")

    # ---------------------------------------------------------
    # Resultados finales
    # ---------------------------------------------------------
    results_df = pd.DataFrame(results)
    print("\n📊 RESULTADOS DEL BENCHMARK")
    print("============================================================")
    print(results_df.to_string(index=False))

    return results_df


if __name__ == "__main__":
    print("Este módulo se usa desde main.py con --benchmark.")
