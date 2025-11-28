"""
Evaluación y Benchmarking de Modelos – Sprint 3
Compara: Naive, RF, XGB, XGB Tunado, CatBoost Tunado
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
import json

def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / np.where(y_true==0,1,y_true))) * 100

def mase(y_true, y_pred, y_train):
    naive = np.abs(np.diff(y_train)).mean()
    return mean_absolute_error(y_true, y_pred) / (naive if naive != 0 else 1)

# -----------------------------
# 1. MODELO NAIVE
# -----------------------------
def naive_forecast(y_train, y_test):
    last_value = y_train.iloc[-1]
    y_pred = np.repeat(last_value, len(y_test))
    return y_pred

# -----------------------------
# 2. RANDOM FOREST
# -----------------------------
def train_rf(X_train, X_test, y_train):
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model.predict(X_test), model

# -----------------------------
# 3. XGBOOST BASELINE
# -----------------------------
def train_xgb(X_train, X_test, y_train):
    model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model.predict(X_test), model

# -----------------------------
# 4. XGBOOST TUNADO
# -----------------------------
def train_xgb_tuned(X_train, X_test, y_train):
    model = XGBRegressor(
        colsample_bylevel=0.6072,
        colsample_bytree=0.7976,
        gamma=0.1788,
        learning_rate=0.0348,
        max_depth=10,
        min_child_weight=7,
        n_estimators=294,
        reg_alpha=0.0882,
        reg_lambda=0.0257,
        subsample=0.6375,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model.predict(X_test), model

# -----------------------------
# 5. CATBOOST TUNADO
# -----------------------------
def train_catboost(X_train, X_test, y_train):
    model = CatBoostRegressor(
        depth=10,
        learning_rate=0.05,
        iterations=500,
        loss_function='MAE',
        random_seed=42,
        verbose=False
    )
    model.fit(X_train, y_train)
    return model.predict(X_test), model

# -----------------------------
# BENCHMARK MASTER
# -----------------------------
def evaluate_all_models(X_train, X_test, y_train, y_test, save_path="benchmark_results.csv"):
    results = []

    MODELS = {
        "Naive": lambda: naive_forecast(y_train, y_test),
        "RandomForest": lambda: train_rf(X_train, X_test, y_train),
        "XGBoost": lambda: train_xgb(X_train, X_test, y_train),
        "XGBoost_Tuned": lambda: train_xgb_tuned(X_train, X_test, y_train),
        "CatBoost_Tuned": lambda: train_catboost(X_train, X_test, y_train),
    }

    for name, func in MODELS.items():
        print(f"\n⚙️ Evaluando modelo: {name}")

        out = func()

        if isinstance(out, tuple):     # modelos que devuelven (predicciones y modelo)
            y_pred, model = out
        else:
            y_pred = out
            model = None

        metrics = {
            "Modelo": name,
            "MAE": mean_absolute_error(y_test, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
            "R²": r2_score(y_test, y_pred),
            "MAPE": mape(y_test, y_pred),
            "MASE": mase(y_test, y_pred, y_train)
        }

        results.append(metrics)

    df_results = pd.DataFrame(results)
    df_results.to_csv(save_path, index=False)

    print(f"\n📁 Benchmark guardado en: {save_path}")
    print(df_results)

    return df_results
