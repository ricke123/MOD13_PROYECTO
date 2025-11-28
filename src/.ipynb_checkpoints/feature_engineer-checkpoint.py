# src/feature_engineer.py
"""
Módulo de ingeniería de features para forecasting Olist.

1) A partir del dataset limpio a nivel ítem (salida de DataLoader):
   - Agrega a nivel mensual-categoría.
   - Construye una master table rica en variables de negocio.

2) Genera:
   - Features temporales (mes, año, dummies, codificación cíclica).
   - Features de serie temporal (lags, MA, EMA, crecimiento, momentum, estacionalidad).
   - Ratios de negocio (ventas por pedido, márgenes, logística, fidelidad, etc.).
   - Stats por categoría (promedios, z-scores, etc.).

3) Prepara el DataFrame final para el modelo:
   - Limpieza de NaN e infinitos.
   - Eliminación de columnas constantes.
   - Selección automática de hasta 99 features numéricas con XGBoost.
"""

import numpy as np
import pandas as pd
from xgboost import XGBRegressor


class FeatureEngineer:
    def __init__(self):
        self.best_features = None

    # ------------------------------------------------------------------
    # 1. CREACIÓN DE TARGET A NIVEL MENSUAL-CATEGORÍA (AGG BÁSICO)
    # ------------------------------------------------------------------
    def create_target_variable(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crea la tabla agregada mensual por categoría (agg_data),
        similar al notebook, con 'demand' como conteo de items.

        Siempre devuelve la columna 'purchase_year_month'.
        """
        df = df.copy()

        # --------------------------------------------------------------
        # Determinar columna base de mes / fecha y estandarizar
        # --------------------------------------------------------------
        if "purchase_year_month" in df.columns:
            # ya viene lista (string o Period)
            df["purchase_year_month"] = df["purchase_year_month"].astype(str)
        elif "order_year" in df.columns and "order_month" in df.columns:
            # combinación año-mes
            df["purchase_year_month"] = (
                df["order_year"].astype(str)
                + "-"
                + df["order_month"].astype(int).astype(str).str.zfill(2)
            )
        elif "order_month" in df.columns:
            df["purchase_year_month"] = df["order_month"].astype(str)
        elif "order_purchase_timestamp" in df.columns:
            df["purchase_year_month"] = (
                pd.to_datetime(df["order_purchase_timestamp"], errors="coerce")
                .dt.to_period("M")
                .astype(str)
            )
        else:
            raise ValueError(
                "❌ No se encontró ninguna columna de fecha mensual "
                "('purchase_year_month', 'order_year'+'order_month' o "
                "'order_purchase_timestamp')."
            )

        # --------------------------------------------------------------
        # Construir diccionario de agregación (solo columnas existentes)
        # --------------------------------------------------------------
        agg_dict: dict = {}

        # Volumen y clientes
        if "product_id" in df.columns:
            agg_dict["product_id"] = ["count", "nunique"]
        if "order_id" in df.columns:
            agg_dict["order_id"] = "nunique"
        if "customer_id" in df.columns:
            agg_dict["customer_id"] = "nunique"
        if "seller_id" in df.columns:
            agg_dict["seller_id"] = "nunique"

        # Precios y flete
        if "price" in df.columns:
            agg_dict["price"] = ["sum", "mean", "std", "min", "max"]
        if "freight_value" in df.columns:
            agg_dict["freight_value"] = ["sum", "mean"]

        # Pagos agregados
        if "payment_total" in df.columns:
            agg_dict["payment_total"] = ["sum", "mean"]
        if "payment_avg" in df.columns:
            agg_dict["payment_avg"] = "mean"
        if "payment_count" in df.columns:
            agg_dict["payment_count"] = "mean"
        if "installments_avg" in df.columns:
            agg_dict["installments_avg"] = "mean"

        # Reviews
        if "review_score_mean" in df.columns:
            agg_dict["review_score_mean"] = "mean"
        if "review_count" in df.columns:
            agg_dict["review_count"] = "sum"

        # Logística
        if "delivery_time_days" in df.columns:
            agg_dict["delivery_time_days"] = ["mean", "std"]
        if "delivery_delay" in df.columns:
            agg_dict["delivery_delay"] = "mean"
        if "is_delayed" in df.columns:
            agg_dict["is_delayed"] = "mean"

        # Porcentajes de tipo de pago
        for pt in ["credit_card", "boleto", "voucher", "debit_card"]:
            col_name = f"pct_{pt}"
            if col_name in df.columns:
                agg_dict[col_name] = "mean"

        # Porcentajes de review
        for score in [1, 2, 3, 4, 5]:
            col_name = f"review_pct_{score}"
            if col_name in df.columns:
                agg_dict[col_name] = "mean"

        if not agg_dict:
            raise ValueError(
                "❌ No se encontraron columnas esperadas para agregación en create_target_variable."
            )

        # --------------------------------------------------------------
        # Agregación mensual-categoría usando purchase_year_month
        # --------------------------------------------------------------
        agg_data = (
            df.groupby(["purchase_year_month", "product_category_name"])
            .agg(agg_dict)
            .reset_index()
        )

        # Renombrar columnas múltiples tipo ('col','sum') → 'col_sum'
        new_cols = []
        for col in agg_data.columns:
            if isinstance(col, tuple):
                base, func = col
                if func:
                    new_cols.append(f"{base}_{func}")
                else:
                    new_cols.append(base)
            else:
                new_cols.append(col)
        agg_data.columns = new_cols

        # Renombres clave
        rename_map = {}
        if "product_id_count" in agg_data.columns:
            rename_map["product_id_count"] = "demand"
        if "product_id_nunique" in agg_data.columns:
            rename_map["product_id_nunique"] = "unique_products"
        if "order_id_nunique" in agg_data.columns:
            rename_map["order_id_nunique"] = "unique_orders"
        if "customer_id_nunique" in agg_data.columns:
            rename_map["customer_id_nunique"] = "unique_customers"
        if "seller_id_nunique" in agg_data.columns:
            rename_map["seller_id_nunique"] = "unique_sellers"
        if "price_sum" in agg_data.columns:
            rename_map["price_sum"] = "total_sales"
        if "price_mean" in agg_data.columns:
            rename_map["price_mean"] = "avg_price"
        if "freight_value_sum" in agg_data.columns:
            rename_map["freight_value_sum"] = "total_freight"
        if "freight_value_mean" in agg_data.columns:
            rename_map["freight_value_mean"] = "avg_freight"
        if "payment_total_sum" in agg_data.columns:
            rename_map["payment_total_sum"] = "total_payments"
        if "payment_total_mean" in agg_data.columns:
            rename_map["payment_total_mean"] = "avg_payment"
        if "review_score_mean_mean" in agg_data.columns:
            rename_map["review_score_mean_mean"] = "avg_review_score"
        if "delivery_time_days_mean" in agg_data.columns:
            rename_map["delivery_time_days_mean"] = "avg_delivery_time_days"
        if "delivery_delay_mean" in agg_data.columns:
            rename_map["delivery_delay_mean"] = "avg_delivery_delay"
        if "is_delayed_mean" in agg_data.columns:
            rename_map["is_delayed_mean"] = "pct_delayed_orders"

        agg_data = agg_data.rename(columns=rename_map)

        print(f"Agg mensual-categoría (agg_data): {agg_data.shape}")
        print(f" monthly_demand: {agg_data.shape}")

        return agg_data

    # ------------------------------------------------------------------
    # 2. CREACIÓN DE FEATURES AVANZADOS (INGENIERÍA PESADA)
    # ------------------------------------------------------------------
    def create_advanced_features(self, agg_data: pd.DataFrame) -> pd.DataFrame:
        """
        Construye la MASTER table con target demand_next_month
        y features avanzados.
        """
        df = agg_data.copy()

        # --------------------------------------------------------------
        # FEATURES TEMPORALES BÁSICAS
        # --------------------------------------------------------------
        # Tolerante a distintos formatos / ubicaciones
        if "purchase_year_month" not in df.columns:
            # Puede venir como índice
            if isinstance(df.index, pd.MultiIndex) and "purchase_year_month" in df.index.names:
                df = df.reset_index()
            elif df.index.name == "purchase_year_month":
                df = df.reset_index()
            elif "order_year" in df.columns and "order_month" in df.columns:
                df["purchase_year_month"] = (
                    df["order_year"].astype(str)
                    + "-"
                    + df["order_month"].astype(int).astype(str).str.zfill(2)
                )
            elif "order_month" in df.columns:
                df["purchase_year_month"] = df["order_month"].astype(str)
                print("ℹ️ 'purchase_year_month' creado desde 'order_month'.")
            else:
                raise KeyError(
                    "❌ create_advanced_features espera la columna 'purchase_year_month' "
                    "o 'order_year'+'order_month' / 'order_month' para poder derivarla.\n"
                    f"Columnas disponibles: {list(df.columns)}"
                )

        df["purchase_year_month"] = df["purchase_year_month"].astype(str)

        df["date"] = pd.to_datetime(df["purchase_year_month"] + "-01", errors="coerce")
        df["year"] = df["date"].dt.year
        df["month_num"] = df["date"].dt.month
        df["quarter"] = df["date"].dt.quarter
        df["month_year"] = df["date"].dt.strftime("%Y-%m")

        # Codificación cíclica mes / trimestre
        df["month_sin"] = np.sin(2 * np.pi * df["month_num"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month_num"] / 12)
        df["quarter_sin"] = np.sin(2 * np.pi * df["quarter"] / 4)
        df["quarter_cos"] = np.cos(2 * np.pi * df["quarter"] / 4)

        # Dummies de temporada
        df["is_holiday_season"] = df["month_num"].isin([11, 12]).astype(int)
        df["is_beginning_year"] = df["month_num"].isin([1, 2]).astype(int)
        df["is_mid_year"] = df["month_num"].isin([6, 7]).astype(int)
        df["is_end_quarter"] = df["month_num"].isin([3, 6, 9, 12]).astype(int)

        # Dummies por mes
        important_months = {
            1: "january",
            2: "february",
            3: "march",
            4: "april",
            5: "may",
            6: "june",
            7: "july",
            8: "august",
            9: "september",
            10: "october",
            11: "november",
            12: "december",
        }
        for m_num, m_name in important_months.items():
            df[f"is_{m_name}"] = (df["month_num"] == m_num).astype(int)

        # Dummies por quarter
        for q in range(1, 5):
            df[f"is_quarter_{q}"] = (df["quarter"] == q).astype(int)

        print(f"Agg con features temporales básicas: {df.shape}")

        # --------------------------------------------------------------
        # FEATURES DE SERIE TEMPORAL (LAGS, MA, EMA, CRECIMIENTO)
        # --------------------------------------------------------------
        def add_temporal_features(group: pd.DataFrame) -> pd.DataFrame:
            group = group.sort_values("date").copy()

            if "demand" not in group.columns:
                raise KeyError(
                    "❌ La columna 'demand' no está en el DataFrame agregado. "
                    "Revisa create_target_variable."
                )

            # Target t+1
            group["demand_next_month"] = group["demand"].shift(-1)

            # LAGS COMPLETOS
            for lag in [1, 2, 3, 6, 12]:
                group[f"demand_lag_{lag}"] = group["demand"].shift(lag)
                if "total_sales" in group.columns:
                    group[f"sales_lag_{lag}"] = group["total_sales"].shift(lag)
                if "avg_price" in group.columns:
                    group[f"price_lag_{lag}"] = group["avg_price"].shift(lag)
                if "avg_review_score" in group.columns:
                    group[f"review_lag_{lag}"] = group["avg_review_score"].shift(lag)
                if "avg_delivery_time_days" in group.columns:
                    group[f"delivery_lag_{lag}"] = group["avg_delivery_time_days"].shift(
                        lag
                    )

            # MOVING AVERAGES
            for window in [2, 3, 6, 12]:
                group[f"ma_{window}"] = (
                    group["demand"].rolling(window, min_periods=1).mean().shift(1)
                )
                if "total_sales" in group.columns:
                    group[f"sales_ma_{window}"] = (
                        group["total_sales"]
                        .rolling(window, min_periods=1)
                        .mean()
                        .shift(1)
                    )
                if "avg_price" in group.columns:
                    group[f"price_ma_{window}"] = (
                        group["avg_price"]
                        .rolling(window, min_periods=1)
                        .mean()
                        .shift(1)
                    )
                if "avg_review_score" in group.columns:
                    group[f"review_ma_{window}"] = (
                        group["avg_review_score"]
                        .rolling(window, min_periods=1)
                        .mean()
                        .shift(1)
                    )

            # MOVING STATISTICS
            for window in [3, 6, 12]:
                group[f"demand_std_{window}"] = (
                    group["demand"].rolling(window, min_periods=1).std().shift(1)
                )
                group[f"demand_min_{window}"] = (
                    group["demand"].rolling(window, min_periods=1).min().shift(1)
                )
                group[f"demand_max_{window}"] = (
                    group["demand"].rolling(window, min_periods=1).max().shift(1)
                )
                if "avg_price" in group.columns:
                    group[f"price_std_{window}"] = (
                        group["avg_price"]
                        .rolling(window, min_periods=1)
                        .std()
                        .shift(1)
                    )

            # EXPONENTIAL MOVING AVERAGES
            for alpha in [0.3, 0.5, 0.7]:
                group[f"ema_{alpha}"] = (
                    group["demand"].ewm(alpha=alpha).mean().shift(1)
                )

            # GROWTH RATES
            group["demand_growth_1m"] = group["demand"].pct_change(1)
            group["demand_growth_3m"] = group["demand"].pct_change(3)
            group["demand_growth_12m"] = group["demand"].pct_change(12)
            if "total_sales" in group.columns:
                group["sales_growth_1m"] = group["total_sales"].pct_change(1)
            if "avg_price" in group.columns:
                group["price_growth_1m"] = group["avg_price"].pct_change(1)

            # MOMENTUM
            group["demand_momentum_3m"] = group["demand"] - group["demand"].shift(3)
            group["demand_momentum_12m"] = group["demand"] - group["demand"].shift(12)
            if "total_sales" in group.columns:
                group["sales_momentum_3m"] = (
                    group["total_sales"] - group["total_sales"].shift(3)
                )

            # SEASONALIDAD
            if len(group) >= 13:
                group["seasonal_ratio_12m"] = group["demand"] / group["demand"].shift(
                    12
                )
                group["seasonal_difference_12m"] = (
                    group["demand"] - group["demand"].shift(12)
                )

            # VOLATILIDAD
            group["demand_volatility_6m"] = group["demand"].rolling(6).std().shift(
                1
            ) / (group["demand"].rolling(6).mean().shift(1) + 1e-8)
            if "avg_price" in group.columns:
                group["price_volatility_6m"] = group["avg_price"].rolling(6).std().shift(
                    1
                ) / (group["avg_price"].rolling(6).mean().shift(1) + 1e-8)

            # ACELERACIÓN
            group["demand_acceleration"] = group["demand_growth_1m"].diff(1)

            # TENDENCIA
            if len(group) >= 3:
                group["demand_trend_3m"] = group["demand"].diff(3) / 3

            return group

        master = df.groupby("product_category_name", group_keys=False).apply(
            add_temporal_features
        )
        print(f"Master tras temporal features: {master.shape}")

        # --------------------------------------------------------------
        # FEATURES DE INTERACCIÓN Y RATIOS DE NEGOCIO
        # --------------------------------------------------------------
        def safe_ratio(num_col, den_col, name):
            if num_col in master.columns and den_col in master.columns:
                master[name] = master[num_col] / (master[den_col] + 1e-8)

        # Básicos de negocio
        if {"total_sales", "unique_orders"}.issubset(master.columns):
            master["sales_per_order"] = master["total_sales"] / (
                master["unique_orders"] + 1
            )
            master["avg_order_value"] = master["total_sales"] / (
                master["unique_orders"] + 1
            )

        if {"demand", "unique_orders"}.issubset(master.columns):
            master["items_per_order"] = master["demand"] / (
                master["unique_orders"] + 1
            )

        if {"unique_orders", "unique_customers"}.issubset(master.columns):
            master["conversion_rate"] = master["unique_orders"] / (
                master["unique_customers"] + 1
            )

        if {"avg_price", "avg_freight"}.issubset(master.columns):
            safe_ratio("avg_price", "avg_freight", "price_to_freight_ratio")
        if {"total_freight", "total_sales"}.issubset(master.columns):
            safe_ratio("total_freight", "total_sales", "freight_to_sales_ratio")
        if {"avg_price", "avg_freight"}.issubset(master.columns):
            master["profit_margin_estimate"] = (
                master["avg_price"] - master["avg_freight"]
            ) / (master["avg_price"] + 1e-8)

        if {"unique_orders", "unique_customers"}.issubset(master.columns):
            master["customer_loyalty_index"] = master["unique_orders"] / (
                master["unique_customers"] + 1
            )
        if {"unique_orders", "unique_sellers"}.issubset(master.columns):
            master["seller_concentration"] = master["unique_orders"] / (
                master["unique_sellers"] + 1
            )

        if {"unique_products", "demand"}.issubset(master.columns):
            master["product_diversity_index"] = master["unique_products"] / (
                master["demand"] + 1
            )
            master["avg_items_per_product"] = master["demand"] / (
                master["unique_products"] + 1
            )

        if "pct_delayed_orders" in master.columns:
            master["on_time_delivery_rate"] = 1 - master["pct_delayed_orders"]
        if {"avg_delivery_time_days", "avg_delivery_delay"}.issubset(master.columns):
            master["delivery_efficiency"] = master["avg_delivery_time_days"] / (
                master["avg_delivery_delay"].abs() + 1
            )

        # Pagos y reviews derivados
        if "installments_avg_mean" in master.columns:
            master["avg_installments_per_order"] = master["installments_avg_mean"]

        if (
            "pct_credit_card_mean" in master.columns
            and "pct_boleto_mean" in master.columns
        ):
            master["credit_card_usage_ratio"] = master["pct_credit_card_mean"] / (
                master["pct_boleto_mean"] + 0.01
            )

        if (
            "review_pct_5_mean" in master.columns
            and "review_pct_1_mean" in master.columns
        ):
            master["review_sentiment_score"] = (
                master["review_pct_5_mean"] - master["review_pct_1_mean"]
            )

        if (
            "review_count_sum" in master.columns
            and "unique_orders" in master.columns
        ):
            master["review_engagement_rate"] = master["review_count_sum"] / (
                master["unique_orders"] + 1
            )

        print(f"Master tras ratios de negocio: {master.shape}")

        # --------------------------------------------------------------
        # FEATURES DE AGREGACIÓN MULTINIVEL (categoría)
        # --------------------------------------------------------------
        cols_for_cat = {}
        if "demand" in master.columns:
            cols_for_cat["demand"] = ["mean", "std", "median", "max"]
        if "avg_price" in master.columns:
            cols_for_cat["avg_price"] = ["mean", "std"]
        if "avg_review_score" in master.columns:
            cols_for_cat["avg_review_score"] = "mean"

        if cols_for_cat:
            category_stats = (
                master.groupby("product_category_name")
                .agg(cols_for_cat)
                .reset_index()
            )

            # aplanar
            cat_cols = ["product_category_name"]
            for base, funcs in cols_for_cat.items():
                if isinstance(funcs, list):
                    for f in funcs:
                        cat_cols.append(f"category_{base}_{f}")
                else:
                    cat_cols.append(f"category_{base}_{funcs}")
            category_stats.columns = cat_cols

            master = master.merge(category_stats, on="product_category_name", how="left")

            # ratios y z-scores
            if {"demand", "category_demand_mean"}.issubset(master.columns):
                master["demand_vs_category_avg"] = master["demand"] / (
                    master["category_demand_mean"] + 1e-8
                )
            if {"avg_price", "category_avg_price_mean"}.issubset(master.columns):
                master["price_vs_category_avg"] = master["avg_price"] / (
                    master["category_avg_price_mean"] + 1e-8
                )
            if {
                "avg_review_score",
                "category_avg_review_score_mean",
            }.issubset(master.columns):
                master["review_vs_category_avg"] = master["avg_review_score"] / (
                    master["category_avg_review_score_mean"] + 1e-8
                )

        print(f"Master tras stats de categoría: {master.shape}")

        # --------------------------------------------------------------
        # LIMPIEZA FINAL DE MASTER
        # --------------------------------------------------------------
        master = master.dropna(subset=["demand_next_month"]).reset_index(drop=True)
        master = master.fillna(0)
        master = master.replace([np.inf, -np.inf], 0)

        numeric_cols_master = master.select_dtypes(include=[np.number]).columns
        constant_cols = [
            col for col in numeric_cols_master if master[col].nunique() <= 1
        ]
        master = master.drop(columns=constant_cols)

        print("\n=== MASTER FINAL ===")
        print("Shape:", master.shape)
        print("Columnas constantes eliminadas:", len(constant_cols))
        print(f" monthly_demand_with_features: {master.shape}")

        return master

    # ------------------------------------------------------------------
    # 3. PREPARACIÓN FINAL PARA EL MODELO + SELECCIÓN DE FEATURES
    # ------------------------------------------------------------------
    def prepare_model_features(
        self,
        df: pd.DataFrame,
        use_feature_selection: bool = True,
        target_col: str = "demand_next_month",
    ) -> pd.DataFrame:
        """
        Preparar features finales para el modelo.
        Mantiene 'purchase_year_month' para splits temporales.
        """
        print("📊 Preparando features para el modelo...")

        df = df.copy()

        cols_to_drop = ["product_category_name"]
        features_df = df.drop(columns=cols_to_drop, errors="ignore").copy()

        # Limpieza básica
        features_df = features_df.fillna(0)
        features_df = features_df.replace([np.inf, -np.inf], 0)

        print(
            f"📈 Features generados (incluyendo fecha y target): "
            f"{features_df.shape[1]} columnas"
        )

        if use_feature_selection:
            exclude_cols = [target_col, "purchase_year_month"]
            candidate_cols = [c for c in features_df.columns if c not in exclude_cols]

            numeric_candidate_cols = [
                c
                for c in candidate_cols
                if pd.api.types.is_numeric_dtype(features_df[c])
            ]

            if len(numeric_candidate_cols) > 10:
                print("🔍 Aplicando selección de features con XGBoost (top_k=99)...")

                selector_df_cols = [
                    c
                    for c in numeric_candidate_cols + [target_col]
                    if c in features_df.columns
                ]

                selected_features = self.select_best_features(
                    features_df[selector_df_cols],
                    target_col=target_col,
                    top_k=99,
                )

                base_cols = []
                if "purchase_year_month" in features_df.columns:
                    base_cols.append("purchase_year_month")
                if target_col in features_df.columns:
                    base_cols.append(target_col)

                final_cols = selected_features + base_cols
                final_cols = [c for c in final_cols if c in features_df.columns]

                features_df = features_df[final_cols].copy()
                print(f"✅ Features después de selección: {features_df.shape}")
            else:
                print(
                    "ℹ️ Muy pocas columnas para selección de features. Se mantienen todas."
                )

        return features_df

    # ------------------------------------------------------------------
    # 4. SELECCIÓN DE FEATURES CON XGBOOST IMPORTANCE (SOLO NUMÉRICAS)
    # ------------------------------------------------------------------
    def select_best_features(
        self,
        features_df: pd.DataFrame,
        target_col: str = "demand_next_month",
        top_k: int = 99,
    ):
        """
        Selecciona las top_k features según importancia de XGBoost.
        Solo usa columnas numéricas (int, float, bool).
        """
        print("🎯 Seleccionando mejores features con XGBoost...")

        if target_col not in features_df.columns:
            raise ValueError(
                f"❌ La columna target '{target_col}' no está en el DataFrame."
            )

        exclude_cols = [target_col, "purchase_year_month", "product_category_name"]

        all_feature_cols = [col for col in features_df.columns if col not in exclude_cols]

        numeric_feature_cols = features_df[all_feature_cols].select_dtypes(
            include=[np.number, "bool"]
        ).columns.tolist()

        if not numeric_feature_cols:
            raise ValueError("❌ No hay columnas numéricas disponibles para XGBoost.")

        X = features_df[numeric_feature_cols].copy()
        y = features_df[target_col].copy()

        X = X.dropna(axis=1, how="all").fillna(0)
        X = X.replace([np.inf, -np.inf], 0)

        model = XGBRegressor(
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
            importance_type="weight",
            n_jobs=-1,
        )

        model.fit(X, y)

        importances = model.feature_importances_
        fi = (
            pd.DataFrame({"feature": X.columns, "importance": importances})
            .sort_values("importance", ascending=False)
        )

        top_k = min(top_k, len(fi))
        selected_features = fi["feature"].head(top_k).tolist()

        print(f"📊 Features originales (numéricas): {len(X.columns)}")
        print(f"🎯 Features seleccionados (top_k={top_k}): {len(selected_features)}")

        print("\n🏆 TOP 20 FEATURES POR IMPORTANCIA:")
        for i, row in fi.head(20).iterrows():
            print(f"  {i+1:2d}. {row['feature']:35} → {row['importance']:.8f}")

        self.best_features = selected_features
        return selected_features
