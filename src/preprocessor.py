# src/preprocessor.py
import pandas as pd
import numpy as np

# Importamos config en vez de una clase Config inexistente
from config import TEMPORAL_CONFIG


class DataPreprocessor:
    """Clase para preprocesamiento de datos"""

    def __init__(self):
        self.feature_names = []

    # ----------------------------------------------------------------------
    # FEATURES TEMPORALES BÁSICAS (a partir de agg_data mensual)
    # ----------------------------------------------------------------------
    def create_temporal_features(self, agg_data: pd.DataFrame) -> pd.DataFrame:
        """Crea features temporales avanzadas a partir de 'order_month' (YYYY-MM)."""
        print("⏰ Creando features temporales básicas...")

        if 'order_month' not in agg_data.columns:
            raise ValueError("❌ 'order_month' no está en agg_data. Asegúrate de haber agregado por mes antes.")

        agg_data['date'] = pd.to_datetime(agg_data['order_month'] + '-01', errors='coerce')
        agg_data['year'] = agg_data['date'].dt.year
        agg_data['month_num'] = agg_data['date'].dt.month
        agg_data['quarter'] = agg_data['date'].dt.quarter

        # Features cíclicas
        agg_data['month_sin'] = np.sin(2 * np.pi * agg_data['month_num'] / 12)
        agg_data['month_cos'] = np.cos(2 * np.pi * agg_data['month_num'] / 12)
        agg_data['quarter_sin'] = np.sin(2 * np.pi * agg_data['quarter'] / 4)
        agg_data['quarter_cos'] = np.cos(2 * np.pi * agg_data['quarter'] / 4)

        # Estacionalidad (general)
        agg_data['is_holiday_season'] = agg_data['month_num'].isin([11, 12]).astype(int)
        agg_data['is_beginning_year'] = agg_data['month_num'].isin([1, 2]).astype(int)
        agg_data['is_mid_year'] = agg_data['month_num'].isin([6, 7]).astype(int)
        agg_data['is_end_quarter'] = agg_data['month_num'].isin([3, 6, 9, 12]).astype(int)

        # Dummies de meses
        important_months = {
            1: 'january', 2: 'february', 3: 'march', 4: 'april', 5: 'may', 6: 'june',
            7: 'july', 8: 'august', 9: 'september', 10: 'october', 11: 'november', 12: 'december'
        }

        for month_num, month_name in important_months.items():
            agg_data[f'is_{month_name}'] = (agg_data['month_num'] == month_num).astype(int)

        # Dummies de trimestres
        for qtr in range(1, 5):
            agg_data[f'is_quarter_{qtr}'] = (agg_data['quarter'] == qtr).astype(int)

        return agg_data

    # ----------------------------------------------------------------------
    # FEATURES DE SERIE TEMPORAL POR CATEGORÍA
    # ----------------------------------------------------------------------
    def create_temporal_series_features(self, group: pd.DataFrame) -> pd.DataFrame:
        """Crea features de series temporales (lags, moving averages, etc.) para un grupo."""
        group = group.sort_values('date').copy()

        # Target t+1
        group['demand_next_month'] = group['demand'].shift(-1)

        # LAGS
        for lag in TEMPORAL_CONFIG['lags']:
            group[f'demand_lag_{lag}'] = group['demand'].shift(lag)
            group[f'sales_lag_{lag}'] = group['total_sales'].shift(lag)
            group[f'price_lag_{lag}'] = group['avg_price'].shift(lag)
            group[f'review_lag_{lag}'] = group['avg_review_score'].shift(lag)
            if 'avg_delivery_time_days' in group.columns:
                group[f'delivery_lag_{lag}'] = group['avg_delivery_time_days'].shift(lag)

        # MOVING AVERAGES
        for window in TEMPORAL_CONFIG['moving_windows']:
            group[f'ma_{window}'] = group['demand'].rolling(window, min_periods=1).mean().shift(1)
            group[f'sales_ma_{window}'] = group['total_sales'].rolling(window, min_periods=1).mean().shift(1)
            group[f'price_ma_{window}'] = group['avg_price'].rolling(window, min_periods=1).mean().shift(1)
            group[f'review_ma_{window}'] = group['avg_review_score'].rolling(window, min_periods=1).mean().shift(1)

        # EXPONENTIAL MOVING AVERAGES
        for alpha in TEMPORAL_CONFIG['ema_alphas']:
            group[f'ema_{alpha}'] = group['demand'].ewm(alpha=alpha, adjust=False).mean().shift(1)

        # MOVING STATISTICS
        for window in [3, 6, 12]:
            group[f'demand_std_{window}'] = group['demand'].rolling(window, min_periods=1).std().shift(1)
            group[f'demand_min_{window}'] = group['demand'].rolling(window, min_periods=1).min().shift(1)
            group[f'demand_max_{window}'] = group['demand'].rolling(window, min_periods=1).max().shift(1)
            group[f'price_std_{window}'] = group['avg_price'].rolling(window, min_periods=1).std().shift(1)

        # GROWTH RATES
        group['demand_growth_1m'] = group['demand'].pct_change(1)
        group['demand_growth_3m'] = group['demand'].pct_change(3)
        group['demand_growth_12m'] = group['demand'].pct_change(12)
        group['sales_growth_1m'] = group['total_sales'].pct_change(1)
        group['price_growth_1m'] = group['avg_price'].pct_change(1)

        # MOMENTUM FEATURES
        group['demand_momentum_3m'] = group['demand'] - group['demand'].shift(3)
        group['demand_momentum_12m'] = group['demand'] - group['demand'].shift(12)
        group['sales_momentum_3m'] = group['total_sales'] - group['total_sales'].shift(3)

        # SEASONALITY FEATURES
        if len(group) >= 13:
            group['seasonal_ratio_12m'] = group['demand'] / group['demand'].shift(12)
            group['seasonal_difference_12m'] = group['demand'] - group['demand'].shift(12)

        # VOLATILITY FEATURES
        group['demand_volatility_6m'] = (
            group['demand'].rolling(6).std().shift(1) /
            (group['demand'].rolling(6).mean().shift(1) + 1e-8)
        )
        group['price_volatility_6m'] = (
            group['avg_price'].rolling(6).std().shift(1) /
            (group['avg_price'].rolling(6).mean().shift(1) + 1e-8)
        )

        # ACCELERATION
        group['demand_acceleration'] = group['demand_growth_1m'].diff(1)

        # TREND
        if len(group) >= 3:
            group['demand_trend_3m'] = group['demand'].diff(3) / 3

        return group

    def apply_temporal_features(self, agg_data: pd.DataFrame) -> pd.DataFrame:
        """Aplica features temporales a todo el dataset por categoría."""
        print("📈 Aplicando features de series temporales por categoría...")
        if 'product_category_name' not in agg_data.columns:
            raise ValueError("❌ Falta 'product_category_name' en agg_data para agrupar por categoría.")

        master = agg_data.groupby('product_category_name', group_keys=False).apply(
            self.create_temporal_series_features
        )
        return master

    # ----------------------------------------------------------------------
    # RATIOS DE NEGOCIO
    # ----------------------------------------------------------------------
    def create_business_ratios(self, master: pd.DataFrame) -> pd.DataFrame:
        """Crea ratios de negocio a partir del master mensual."""
        print("💰 Creando ratios de negocio...")

        # Business ratios
        master['sales_per_order'] = master['total_sales'] / (master['unique_orders'] + 1)
        master['items_per_order'] = master['demand'] / (master['unique_orders'] + 1)
        master['avg_order_value'] = master['total_sales'] / (master['unique_orders'] + 1)
        master['conversion_rate'] = master['unique_orders'] / (master['unique_customers'] + 1)

        # Price and cost ratios
        master['price_to_freight_ratio'] = master['avg_price'] / (master['avg_freight'] + 1)
        master['freight_to_sales_ratio'] = master['total_freight'] / (master['total_sales'] + 1)
        master['profit_margin_estimate'] = (
            master['avg_price'] - master['avg_freight']
        ) / (master['avg_price'] + 1)

        # Customer behavior ratios
        master['customer_loyalty_index'] = master['unique_orders'] / (master['unique_customers'] + 1)
        master['seller_concentration'] = master['unique_orders'] / (master['unique_sellers'] + 1)

        # Product diversity ratios
        if 'unique_products' in master.columns:
            master['product_diversity_index'] = master['unique_products'] / (master['demand'] + 1)
            master['avg_items_per_product'] = master['demand'] / (master['unique_products'] + 1)
        else:
            master['product_diversity_index'] = 0
            master['avg_items_per_product'] = 0

        # Delivery performance ratios
        if {'pct_delayed_orders', 'avg_delivery_time_days', 'avg_delivery_delay'}.issubset(master.columns):
            master['on_time_delivery_rate'] = 1 - master['pct_delayed_orders']
            master['delivery_efficiency'] = master['avg_delivery_time_days'] / (
                master['avg_delivery_delay'].abs() + 1
            )

        # Payment behavior ratios
        if 'installments_avg_mean' in master.columns:
            master['avg_installments_per_order'] = master['installments_avg_mean']
        else:
            master['avg_installments_per_order'] = 0

        if {'pct_credit_card_mean', 'pct_boleto_mean'}.issubset(master.columns):
            master['credit_card_usage_ratio'] = master['pct_credit_card_mean'] / (
                master['pct_boleto_mean'] + 0.01
            )
        else:
            master['credit_card_usage_ratio'] = 0

        # Review quality ratios
        if {'review_pct_5_mean', 'review_pct_1_mean'}.issubset(master.columns):
            master['review_sentiment_score'] = (
                master['review_pct_5_mean'] - master['review_pct_1_mean']
            )
        else:
            master['review_sentiment_score'] = 0

        if {'review_count_sum', 'unique_orders'}.issubset(master.columns):
            master['review_engagement_rate'] = master['review_count_sum'] / (master['unique_orders'] + 1)
        else:
            master['review_engagement_rate'] = 0

        return master

    # ----------------------------------------------------------------------
    # YOY / MOM / CUMULATIVOS
    # ----------------------------------------------------------------------
    def add_yoy_features(self, group: pd.DataFrame) -> pd.DataFrame:
        """Añade features Year-over-Year y acumulados dentro de un grupo."""
        group = group.sort_values('date').copy()

        # YoY comparisons
        if len(group) >= 13:
            group['yoy_demand_growth'] = group['demand'] / group['demand'].shift(12) - 1
            group['yoy_sales_growth'] = group['total_sales'] / group['total_sales'].shift(12) - 1
            group['yoy_price_growth'] = group['avg_price'] / group['avg_price'].shift(12) - 1

        # Month-over-Month
        group['mom_demand_growth'] = group['demand'].pct_change(1)
        group['mom_sales_growth'] = group['total_sales'].pct_change(1)

        # Cumulative yearly-like rolling (hasta 12 meses)
        group['cumulative_demand_year'] = group['demand'].rolling(12, min_periods=1).sum()
        group['cumulative_sales_year'] = group['total_sales'].rolling(12, min_periods=1).sum()

        return group

    def add_statistical_features(self, group: pd.DataFrame) -> pd.DataFrame:
        """Añade features estadísticos (skew, percentiles, CV)."""
        group = group.sort_values('date').copy()

        if len(group) >= 6:
            try:
                group['demand_skew_12m'] = group['demand'].rolling(12).skew().shift(1)
            except Exception:
                group['demand_skew_12m'] = 0

            group['demand_percentile_25_12m'] = group['demand'].rolling(12).quantile(0.25).shift(1)
            group['demand_percentile_75_12m'] = group['demand'].rolling(12).quantile(0.75).shift(1)
            group['demand_iqr_12m'] = (
                group['demand_percentile_75_12m'] - group['demand_percentile_25_12m']
            )

            group['demand_cv_12m'] = (
                group['demand'].rolling(12).std().shift(1) /
                (group['demand'].rolling(12).mean().shift(1) + 1e-8)
            )

        return group

    # ----------------------------------------------------------------------
    # FEATURES DE CATEGORÍA
    # ----------------------------------------------------------------------
    def add_category_features(self, master: pd.DataFrame) -> pd.DataFrame:
        """Añade features agregados a nivel categoría."""
        print("🏷️ Añadiendo features de categoría...")

        required_cols = {'product_category_name', 'demand', 'avg_price', 'avg_review_score'}
        if not required_cols.issubset(master.columns):
            raise ValueError(f"❌ Faltan columnas para category features: {required_cols - set(master.columns)}")

        category_stats = master.groupby('product_category_name').agg({
            'demand': ['mean', 'std', 'median', 'max'],
            'avg_price': ['mean', 'std'],
            'avg_review_score': ['mean']
        }).reset_index()

        category_stats.columns = [
            'product_category_name',
            'category_avg_demand', 'category_std_demand',
            'category_median_demand', 'category_max_demand',
            'category_avg_price', 'category_std_price',
            'category_avg_review'
        ]

        master = master.merge(category_stats, on='product_category_name', how='left')

        # Ratios relativos
        master['demand_vs_category_avg'] = master['demand'] / (master['category_avg_demand'] + 1e-8)
        master['price_vs_category_avg'] = master['avg_price'] / (master['category_avg_price'] + 1e-8)
        master['review_vs_category_avg'] = (
            master['avg_review_score'] / (master['category_avg_review'] + 1e-8)
        )

        # Z-scores
        master['demand_z_score'] = (
            master['demand'] - master['category_avg_demand']
        ) / (master['category_std_demand'] + 1e-8)
        master['price_z_score'] = (
            master['avg_price'] - master['category_avg_price']
        ) / (master['category_std_price'] + 1e-8)

        return master

    # ----------------------------------------------------------------------
    # LIMPIEZA FINAL
    # ----------------------------------------------------------------------
    def clean_final_dataset(self, master: pd.DataFrame) -> pd.DataFrame:
        """Limpieza final del dataset antes de entrenar el modelo."""
        print("🧼 Limpieza final del dataset...")

        # Quitar filas sin target
        if 'demand_next_month' not in master.columns:
            raise ValueError("❌ Falta la columna 'demand_next_month' antes de limpiar el dataset.")

        master = master.dropna(subset=['demand_next_month']).reset_index(drop=True)

        # Rellenar NaN
        master = master.fillna(0)

        # Reemplazar infinitos
        master = master.replace([np.inf, -np.inf], 0)

        # Quitar columnas constantes
        numeric_cols = master.select_dtypes(include=[np.number]).columns
        constant_cols = [col for col in numeric_cols if master[col].nunique() <= 1]

        if constant_cols:
            print(f"   🧽 Eliminando columnas constantes: {len(constant_cols)}")
            master = master.drop(columns=constant_cols)

        print(f"   ✅ Dataset final listo: {master.shape}")
        return master
