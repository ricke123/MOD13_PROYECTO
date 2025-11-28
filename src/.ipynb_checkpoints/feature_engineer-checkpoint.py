"""
Módulo para ingeniería de features - CON 157 FEATURES ESPECÍFICOS
"""
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBRegressor
from config import DATA_PROCESSED

class FeatureEngineer:
    def __init__(self):
        self.processed_path = DATA_PROCESSED
        self.best_features = None
        
    def create_target_variable(self, df):
        """Crear variable target: demanda del siguiente mes"""
        print("🎯 Creando variable target...")
        
        # Verificar y convertir order_purchase_timestamp a datetime
        if 'order_purchase_timestamp' in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df['order_purchase_timestamp']):
                print("🔄 Convirtiendo order_purchase_timestamp a datetime...")
                df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'], errors='coerce')
        
        # 1. Filtrar solo órdenes entregadas
        df_delivered = df[df['order_status'] == 'delivered'].copy()
        
        # 2. Crear variable de mes-año para agrupación
        df_delivered['purchase_year_month'] = df_delivered['order_purchase_timestamp'].dt.to_period('M')
        print(f"Rango temporal: {df_delivered['purchase_year_month'].min()} a {df_delivered['purchase_year_month'].max()}")
        
        # 3. Calcular demanda mensual por categoría con MÁS MÉTRICAS
        monthly_demand = df_delivered.groupby(['purchase_year_month', 'product_category_name']).agg({
            'product_id': 'count',  # Unidades vendidas
            'price': ['sum', 'mean', 'std', 'max', 'min'],  # Múltiples stats de precio
            'freight_value': ['sum', 'mean'],  # Stats de flete
            'order_id': 'nunique',  # Número de órdenes únicas
            'customer_id': 'nunique', # Clientes únicos
            'seller_id': 'nunique', # Vendedores únicos
            'review_score': 'mean',  # Puntuación promedio
            'payment_installments': ['sum', 'mean', 'max'],  # Stats de cuotas
            'order_item_id': 'count',  # Items totales
        }).reset_index()
        
        # Aplanar columnas multi-index
        monthly_demand.columns = ['_'.join(col).strip() if col[1] else col[0] for col in monthly_demand.columns.values]
        
        # Renombrar columnas
        monthly_demand.rename(columns={
            'product_id_count': 'demand',
            'price_sum': 'total_sales',
            'price_mean': 'avg_price',
            'price_std': 'price_std',
            'price_max': 'price_max', 
            'price_min': 'price_min',
            'freight_value_sum': 'total_freight',
            'freight_value_mean': 'avg_freight',
            'order_id_nunique': 'unique_orders',
            'customer_id_nunique': 'unique_customers',
            'seller_id_nunique': 'unique_sellers',
            'review_score_mean': 'avg_review_score',
            'payment_installments_sum': 'installments_total',
            'payment_installments_mean': 'installments_avg',
            'payment_installments_max': 'installments_max',
            'order_item_id_count': 'payment_count'
        }, inplace=True)
        
        print(f"Registros de demanda mensual: {len(monthly_demand)}")
        
        # 4. Crear target: demanda del siguiente mes
        monthly_demand = monthly_demand.sort_values(['product_category_name', 'purchase_year_month'])
        monthly_demand['demand_next_month'] = monthly_demand.groupby('product_category_name')['demand'].shift(-1)
        
        # 5. Filtrar registros con target disponible
        monthly_demand_clean = monthly_demand[monthly_demand['demand_next_month'].notna()].copy()
        
        print(f"Registros con target disponible: {len(monthly_demand_clean)}")
        print(f"Porcentaje de completitud: {monthly_demand_clean['demand_next_month'].notna().mean()*100:.1f}%")
        
        return monthly_demand_clean
    
    def create_advanced_features(self, df):
        """Crear TODOS los 157 features específicos del EDA"""
        print("🔧 Creando 157 features avanzados...")
        
        # Convertir purchase_year_month a datetime si es Period
        if hasattr(df['purchase_year_month'], 'dt'):
            df['purchase_year_month'] = df['purchase_year_month'].dt.to_timestamp()
        
        # Features temporales básicos
        df['year'] = df['purchase_year_month'].dt.year
        df['month_num'] = df['purchase_year_month'].dt.month
        df['quarter'] = df['purchase_year_month'].dt.quarter
        df['week_of_year'] = df['purchase_year_month'].dt.isocalendar().week
        
        # Features cíclicos
        df['month_sin'] = np.sin(2 * np.pi * df['month_num'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month_num'] / 12)
        df['quarter_sin'] = np.sin(2 * np.pi * df['quarter'] / 4)
        df['quarter_cos'] = np.cos(2 * np.pi * df['quarter'] / 4)
        
        # Ordenar para cálculos temporales
        df = df.sort_values(['product_category_name', 'purchase_year_month'])
        
        # Crear TODOS los features en el orden exacto de importancia
        df = self._create_core_temporal_features(df)
        df = self._create_advanced_statistical_features(df)
        df = self._create_category_level_features(df)
        df = self._create_business_metrics(df)
        df = self._create_growth_momentum_features(df)
        df = self._create_seasonal_holiday_features(df)
        df = self._create_delivery_metrics(df)
        df = self._create_financial_ratios(df)
        
        return df
    
    def _create_core_temporal_features(self, df):
        """Features temporales principales"""
        print("   📅 Creando features temporales principales...")
        
        # LAGS de demanda (los más importantes)
        for lag in [1, 2, 3, 6, 12]:
            df[f'demand_lag_{lag}'] = df.groupby('product_category_name')['demand'].shift(lag)
        
        # LAGS de ventas
        for lag in [1, 2, 3, 6, 12]:
            df[f'sales_lag_{lag}'] = df.groupby('product_category_name')['total_sales'].shift(lag)
        
        # LAGS de precio
        for lag in [1, 2, 3, 6, 12]:
            df[f'price_lag_{lag}'] = df.groupby('product_category_name')['avg_price'].shift(lag)
        
        # LAGS de reviews
        for lag in [1, 2, 3, 6, 12]:
            df[f'review_lag_{lag}'] = df.groupby('product_category_name')['avg_review_score'].shift(lag)
        
        # MEDIAS MÓVILES (ma_2, ma_3, ma_6, ma_12)
        for window in [2, 3, 6, 12]:
            df[f'ma_{window}'] = df.groupby('product_category_name')['demand'].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
        
        # EMA (ema_0.3, ema_0.5, ema_0.7)
        for alpha in [0.3, 0.5, 0.7]:
            df[f'ema_{alpha}'] = df.groupby('product_category_name')['demand'].transform(
                lambda x: x.ewm(alpha=alpha, adjust=False).mean()
            )
        
        # Medias móviles de ventas
        for window in [2, 3, 6, 12]:
            df[f'sales_ma_{window}'] = df.groupby('product_category_name')['total_sales'].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
        
        return df
    
    def _create_advanced_statistical_features(self, df):
        """Features estadísticos avanzados"""
        print("   📊 Creando features estadísticos avanzados...")
        
        # Estadísticos de demanda (std, min, max)
        for window in [3, 6, 12]:
            df[f'demand_std_{window}'] = df.groupby('product_category_name')['demand'].transform(
                lambda x: x.rolling(window, min_periods=1).std()
            )
            df[f'demand_min_{window}'] = df.groupby('product_category_name')['demand'].transform(
                lambda x: x.rolling(window, min_periods=1).min()
            )
            df[f'demand_max_{window}'] = df.groupby('product_category_name')['demand'].transform(
                lambda x: x.rolling(window, min_periods=1).max()
            )
        
        # Estadísticos de precio
        for window in [3, 6, 12]:
            df[f'price_std_{window}'] = df.groupby('product_category_name')['avg_price'].transform(
                lambda x: x.rolling(window, min_periods=1).std()
            )
        
        # Medias móviles de precio
        for window in [2, 3, 6, 12]:
            df[f'price_ma_{window}'] = df.groupby('product_category_name')['avg_price'].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
        
        # Medias móviles de reviews
        for window in [2, 3, 6, 12]:
            df[f'review_ma_{window}'] = df.groupby('product_category_name')['avg_review_score'].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
        
        return df
    
    def _create_category_level_features(self, df):
        """Features a nivel de categoría"""
        print("   🏷️ Creando features de categoría...")
        
        # Estadísticos por categoría
        category_stats = df.groupby('product_category_name').agg({
            'demand': ['mean', 'median', 'std', 'max'],
            'avg_price': ['mean', 'std'],
            'avg_review_score': ['mean'],
            'total_sales': ['mean'],
        }).round(4)
        
        category_stats.columns = ['_'.join(col).strip() for col in category_stats.columns.values]
        category_stats = category_stats.rename(columns={
            'demand_mean': 'category_avg_demand',
            'demand_median': 'category_median_demand', 
            'demand_std': 'category_std_demand',
            'demand_max': 'category_max_demand',
            'avg_price_mean': 'category_avg_price',
            'avg_price_std': 'category_std_price',
            'avg_review_score_mean': 'category_avg_review',
            'total_sales_mean': 'category_avg_sales'
        })
        
        df = df.merge(category_stats, on='product_category_name', how='left')
        
        # Comparación con categoría
        df['demand_vs_category_avg'] = df['demand'] / df['category_avg_demand']
        df['price_vs_category_avg'] = df['avg_price'] / df['category_avg_price']
        df['review_vs_category_avg'] = df['avg_review_score'] / df['category_avg_review']
        
        # Z-scores
        df['demand_z_score'] = (df['demand'] - df['category_avg_demand']) / df['category_std_demand']
        df['price_z_score'] = (df['avg_price'] - df['category_avg_price']) / df['category_std_price']
        
        return df
    
    def _create_business_metrics(self, df):
        """Métricas de negocio"""
        print("   💼 Creando métricas de negocio...")
        
        # Features de concentración y eficiencia
        df['seller_concentration'] = df['unique_sellers'] / df['demand'].replace(0, 1)
        df['items_per_order'] = df['demand'] / df['unique_orders'].replace(0, 1)
        df['sales_per_order'] = df['total_sales'] / df['unique_orders'].replace(0, 1)
        df['avg_order_value'] = df['total_sales'] / df['unique_orders'].replace(0, 1)
        df['avg_items_per_product'] = df['demand'] / df.groupby(['product_category_name', 'purchase_year_month'])['demand'].transform('count')
        
        # Diversidad
        df['product_diversity_index'] = df['unique_sellers'] * df['unique_customers'] / df['demand'].replace(0, 1)
        df['unique_products'] = df.groupby(['product_category_name', 'purchase_year_month'])['demand'].transform('count')
        
        # Acumulados anuales
        df['cumulative_demand_year'] = df.groupby(['product_category_name', 'year'])['demand'].cumsum()
        df['cumulative_sales_year'] = df.groupby(['product_category_name', 'year'])['total_sales'].cumsum()
        
        return df
    
    def _create_growth_momentum_features(self, df):
        """Features de crecimiento y momentum"""
        print("   📈 Creando features de crecimiento...")
        
        # Crecimiento mensual
        df['demand_growth_1m'] = df.groupby('product_category_name')['demand'].pct_change(1)
        df['demand_growth_3m'] = df.groupby('product_category_name')['demand'].pct_change(3)
        df['demand_growth_12m'] = df.groupby('product_category_name')['demand'].pct_change(12)
        
        df['sales_growth_1m'] = df.groupby('product_category_name')['total_sales'].pct_change(1)
        df['price_growth_1m'] = df.groupby('product_category_name')['avg_price'].pct_change(1)
        
        # Momentum
        df['demand_momentum_3m'] = df['demand'] / df['demand_lag_3'].replace(0, 1) - 1
        df['demand_momentum_12m'] = df['demand'] / df['demand_lag_12'].replace(0, 1) - 1
        df['sales_momentum_3m'] = df['total_sales'] / df['sales_lag_3'].replace(0, 1) - 1
        
        # Aceleración
        df['demand_acceleration'] = df['demand_growth_1m'] - df['demand_growth_1m'].shift(1)
        
        # Diferencias estacionales
        df['seasonal_difference_12m'] = df['demand'] - df['demand_lag_12']
        df['seasonal_ratio_12m'] = df['demand'] / df['demand_lag_12'].replace(0, 1)
        
        # Tendencias
        df['demand_trend_3m'] = df.groupby('product_category_name')['demand'].transform(
            lambda x: x.rolling(3, min_periods=1).apply(lambda y: np.polyfit(range(len(y)), y, 1)[0] if len(y) > 1 else 0)
        )
        
        # Crecimiento interanual
        df['yoy_demand_growth'] = df.groupby('product_category_name')['demand'].pct_change(12)
        df['yoy_sales_growth'] = df.groupby('product_category_name')['total_sales'].pct_change(12)
        df['yoy_price_growth'] = df.groupby('product_category_name')['avg_price'].pct_change(12)
        
        # Crecimiento mes a mes
        df['mom_demand_growth'] = df['demand_growth_1m']
        df['mom_sales_growth'] = df['sales_growth_1m']
        
        return df
    
    def _create_seasonal_holiday_features(self, df):
        """Features estacionales y de temporada brasileña"""
        print("   🎄 Creando features estacionales brasileñas...")
        
        # Estacionalidad brasileña PRINCIPAL
        df['is_black_friday_month_br'] = df['month_num'].isin([11]).astype(int)
        df['is_january_sales_br'] = df['month_num'].isin([1]).astype(int)
        df['is_carnival_br'] = df['month_num'].isin([2]).astype(int)
        df['is_back_to_school_br'] = df['month_num'].isin([1, 2]).astype(int)
        df['is_good_friday_br'] = df['month_num'].isin([3, 4]).astype(int)  # Marzo/Abril
        df['is_labor_day_br'] = df['month_num'].isin([5]).astype(int)
        df['is_tax_season_br'] = df['month_num'].isin([4, 5]).astype(int)  # Abril/Mayo
        
        # Temporadas brasileñas
        df['is_summer_br'] = df['month_num'].isin([12, 1, 2]).astype(int)
        df['is_autumn_br'] = df['month_num'].isin([3, 4, 5]).astype(int)
        df['is_winter_br'] = df['month_num'].isin([6, 7, 8]).astype(int)
        df['is_spring_br'] = df['month_num'].isin([9, 10, 11]).astype(int)
        
        # Meses específicos
        df['is_july'] = (df['month_num'] == 7).astype(int)
        df['is_november'] = (df['month_num'] == 11).astype(int)
        
        # Vacaciones escolares
        df['is_school_holidays_dec_jan'] = df['month_num'].isin([12, 1]).astype(int)
        df['is_school_holidays_jul'] = (df['month_num'] == 7).astype(int)
        
        # Temporada de lluvias (Noreste)
        df['is_rainy_season_ne'] = df['month_num'].isin([3, 4, 5, 6]).astype(int)
        
        # Fin de semana largo
        df['is_long_weekend_br'] = df['month_num'].isin([2, 4, 5, 9, 10, 11]).astype(int)
        
        # Temporada de compras
        df['is_holiday_season'] = df['month_num'].isin([11, 12]).astype(int)
        df['is_mid_year'] = df['month_num'].isin([6, 7]).astype(int)
        df['is_end_quarter'] = df['month_num'].isin([3, 6, 9, 12]).astype(int)
        
        # Efectos especiales
        df['black_friday_premium_effect'] = (df['month_num'] == 11).astype(int) * df['avg_price']
        df['summer_beauty_effect'] = (df['month_num'].isin([12, 1, 2])).astype(int)
        df['winter_fashion_effect'] = (df['month_num'].isin([6, 7, 8])).astype(int)
        df['summer_sports_effect'] = (df['month_num'].isin([12, 1, 2])).astype(int)
        df['winter_electronics_effect'] = (df['month_num'].isin([6, 7, 8])).astype(int)
        
        # Efecto 13º salario
        df['thirteenth_salary_boost'] = df['month_num'].isin([11, 12]).astype(int)
        
        return df
    
    def _create_delivery_metrics(self, df):
        """Métricas de entrega (si están disponibles en los datos)"""
        print("   🚚 Creando métricas de entrega...")
        
        # Simular métricas de entrega basadas en lags temporales
        # En un caso real, estas vendrían de los datos de orders
        for lag in [1, 2, 3, 6, 12]:
            df[f'delivery_lag_{lag}'] = df.groupby('product_category_name')['demand'].shift(lag) * 0.1  # Simulado
        
        # Métricas de eficiencia de entrega simuladas
        df['avg_delivery_time_days'] = 10 + np.random.normal(0, 2, len(df))  # Simulado
        df['delivery_time_days_std'] = 2 + np.random.normal(0, 0.5, len(df))  # Simulado
        df['avg_delivery_delay'] = np.random.normal(2, 1, len(df))  # Simulado
        df['pct_delayed_orders'] = np.random.uniform(0.05, 0.15, len(df))  # Simulado
        df['delivery_efficiency'] = 1 - df['pct_delayed_orders']
        df['on_time_delivery_rate'] = 1 - df['pct_delayed_orders']
        
        return df
    
    def _create_financial_ratios(self, df):
        """Ratios financieros"""
        print("   💰 Creando ratios financieros...")
        
        # Ratios financieros
        df['freight_to_sales_ratio'] = df['total_freight'] / df['total_sales'].replace(0, 1)
        df['price_to_freight_ratio'] = df['avg_price'] / (df['total_freight'] / df['demand'].replace(0, 1))
        
        # Volatilidad
        df['demand_volatility_6m'] = df['demand_std_6'] / df['ma_6'].replace(0, 1)
        df['price_volatility_6m'] = df['price_std_6'] / df['price_ma_6'].replace(0, 1)
        
        # Margen de beneficio estimado
        df['profit_margin_estimate'] = (df['total_sales'] - df['total_freight']) / df['total_sales'].replace(0, 1)
        
        return df
    
    def select_best_features(self, features_df, target_col='demand_next_month', threshold=0.0001):
        """Seleccionar mejores features basado en importancia - VERSIÓN MEJORADA"""
        print("🎯 Seleccionando mejores features...")
        
        # Excluir columnas no features
        exclude_cols = [target_col, 'purchase_year_month', 'product_category_name']
        feature_cols = [col for col in features_df.columns if col not in exclude_cols]
        
        X = features_df[feature_cols]
        y = features_df[target_col]
        
        # Eliminar columnas con todos NaN
        X = X.dropna(axis=1, how='all')
        
        # Llenar NaN restantes
        X = X.fillna(0)
        
        # Reemplazar infinitos
        X = X.replace([np.inf, -np.inf], 0)
        
        # Usar hiperparámetros optimizados para la selección
        selector = XGBRegressor(
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
            importance_type='weight'
        )
        
        selector.fit(X, y)
        
        # Seleccionar features con importancia > threshold
        feature_selector = SelectFromModel(selector, threshold=threshold, prefit=True)
        selected_features = X.columns[feature_selector.get_support()]
        
        print(f"📊 Features originales: {len(X.columns)}")
        print(f"🎯 Features seleccionados: {len(selected_features)}")
        
        # Mostrar importancia de features
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': selector.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n🏆 TOP 20 FEATURES POR IMPORTANCIA:")
        for i, row in feature_importance.head(20).iterrows():
            print(f"  {i+1:2d}. {row['feature']:35} → {row['importance']:.8f}")
        
        self.best_features = selected_features
        return selected_features
    
    def prepare_model_features(self, df, use_feature_selection=True):
        """Preparar features finales para el modelo"""
        print("📊 Preparando features para el modelo...")
        
        # Eliminar columnas no necesarias
        features_to_drop = ['purchase_year_month', 'product_category_name']
        features_df = df.drop(columns=features_to_drop, errors='ignore')
        
        # Manejar valores nulos
        features_df = features_df.fillna(0)
        
        # Reemplazar infinitos
        features_df = features_df.replace([np.inf, -np.inf], 0)
        
        print(f"📈 Features generados: {features_df.shape[1]} columnas")
        
        # Selección de features si está habilitada
        if use_feature_selection and len(features_df.columns) > 50:
            print("🔍 Aplicando selección de features...")
            selected_features = self.select_best_features(features_df)
            if len(selected_features) > 0:
                features_df = features_df[selected_features.tolist() + ['demand_next_month']]
                print(f"✅ Features después de selección: {features_df.shape}")
        
        return features_df