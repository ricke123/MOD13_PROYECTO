# src/feature_engineer.py
import pandas as pd
import numpy as np

# Cambiar import relativo
from config import Config


class FeatureEngineer:
    """Clase para ingeniería de features"""
    
    def __init__(self, data):
        self.data = data
        self.master_data = None
    
    def create_base_features(self):
        """Crea features base temporales y de negocio"""
        print("🔧 Creando features base...")
        
        orders = self.data['orders_clean'].copy()
        
        # Features temporales básicas
        orders['order_month'] = orders['order_purchase_timestamp'].dt.to_period('M').astype(str)
        orders['delivery_time_days'] = (
            orders['order_delivered_customer_date'] - orders['order_purchase_timestamp']
        ).dt.days
        
        # Features de delivery
        if 'order_estimated_delivery_date' in orders.columns:
            orders['delivery_delay'] = (
                orders['order_delivered_customer_date'] - orders['order_estimated_delivery_date']
            ).dt.days
            orders['is_delayed'] = (orders['delivery_delay'] > 0).astype(int)
        else:
            orders['delivery_delay'] = 0
            orders['is_delayed'] = 0
        
        self.data['orders_with_features'] = orders
        return orders
    
    def create_payment_features(self):
        """Crea features de pagos"""
        print("🔧 Creando features de pagos...")
        payments = self.data['payments_clean'].copy()
        
        # Agregaciones básicas
        payments_agg = payments.groupby('order_id').agg({
            'payment_value': ['sum', 'mean', 'count'],
            'payment_installments': ['mean', 'max', 'sum'],
        }).reset_index()
        
        payments_agg.columns = [
            'order_id', 'payment_total', 'payment_avg', 'payment_count',
            'installments_avg', 'installments_max', 'installments_total'
        ]
        
        # Distribución de tipos de pago
        try:
            payment_type_pivot = payments.groupby(['order_id', 'payment_type']).size().unstack(fill_value=0)
            payment_type_pivot = payment_type_pivot.div(payment_type_pivot.sum(axis=1), axis=0)
            payment_type_pivot.columns = [f'pct_{col}' for col in payment_type_pivot.columns]
            payment_type_pivot = payment_type_pivot.reset_index()
            payments_final = payments_agg.merge(payment_type_pivot, on='order_id', how='left')
        except:
            payments_final = payments_agg.copy()
            for pt in ['credit_card', 'boleto', 'voucher', 'debit_card']:
                payments_final[f'pct_{pt}'] = 0.0
        
        self.data['payments_features'] = payments_final
        return payments_final
    
    def create_review_features(self):
        """Crea features de reviews"""
        print("🔧 Creando features de reviews...")
        reviews = self.data['reviews_clean'].copy()
        
        reviews_agg = reviews.groupby('order_id').agg({
            'review_score': ['mean', 'count'],
        }).reset_index()
        reviews_agg.columns = ['order_id', 'review_score_mean', 'review_count']
        
        # Distribución de scores
        for score in [1, 2, 3, 4, 5]:
            try:
                score_pct = reviews[reviews['review_score'] == score].groupby('order_id').size() / reviews.groupby('order_id').size()
                score_pct.name = f'review_pct_{score}'
                reviews_agg = reviews_agg.merge(score_pct, on='order_id', how='left')
            except:
                reviews_agg[f'review_pct_{score}'] = 0.0
        
        reviews_agg = reviews_agg.fillna(0)
        self.data['reviews_features'] = reviews_agg
        return reviews_agg
    
    def create_dataset_principal(self):
        """Crea el dataset principal uniendo todas las tablas"""
        print("🔗 Creando dataset principal...")
        
        # Unir items con products
        items_prod = self.data['items_clean'].merge(
            self.data['products_clean'][['product_id', 'product_category_name']], 
            on='product_id', how='left'
        )
        
        # Unir todas las tablas
        df = (items_prod
              .merge(self.data['orders_with_features'], on='order_id', how='left')
              .merge(self.data['payments_features'], on='order_id', how='left')
              .merge(self.data['reviews_features'], on='order_id', how='left'))
        
        # Limpieza final
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)
        df = df[df['order_month'].notna() & df['product_category_name'].notna()].copy()
        
        self.data['dataset_principal'] = df
        print(f"   ✅ Dataset principal: {df.shape}")
        return df
    
    def aggregate_by_month_category(self):
        """Agrega datos por mes y categoría"""
        print("📊 Agregando por mes y categoría...")
        df = self.data['dataset_principal']
        
        agg_dict = {
            'product_id': ['count', 'nunique'],
            'order_id': 'nunique', 
            'customer_id': 'nunique',
            'seller_id': 'nunique',
            'price': ['sum', 'mean', 'std', 'min', 'max'],
            'freight_value': ['sum', 'mean'],
            'payment_total': ['sum', 'mean'],
            'payment_avg': 'mean',
            'payment_count': 'mean',
            'installments_avg': 'mean',
            'review_score_mean': 'mean',
            'review_count': 'sum',
            'delivery_time_days': ['mean', 'std'],
            'delivery_delay': 'mean',
            'is_delayed': 'mean',
        }
        
        # Agregar porcentajes de pago
        for pt in ['credit_card', 'boleto', 'voucher', 'debit_card']:
            col_name = f'pct_{pt}'
            if col_name in df.columns:
                agg_dict[col_name] = 'mean'
        
        # Agregar porcentajes de review
        for score in [1, 2, 3, 4, 5]:
            col_name = f'review_pct_{score}'
            if col_name in df.columns:
                agg_dict[col_name] = 'mean'
        
        agg_data = df.groupby(['order_month', 'product_category_name']).agg(agg_dict).reset_index()
        agg_data.columns = [f'{col[0]}_{col[1]}' if col[1] != '' else col[0] for col in agg_data.columns]
        
        # Renombrar columnas clave
        agg_data = agg_data.rename(columns={
            'product_id_count': 'demand',
            'product_id_nunique': 'unique_products',
            'order_id_nunique': 'unique_orders',
            'customer_id_nunique': 'unique_customers', 
            'seller_id_nunique': 'unique_sellers',
            'price_sum': 'total_sales',
            'price_mean': 'avg_price',
            'freight_value_sum': 'total_freight',
            'freight_value_mean': 'avg_freight',
            'payment_total_sum': 'total_payments',
            'payment_total_mean': 'avg_payment',
            'review_score_mean_mean': 'avg_review_score',
            'delivery_time_days_mean': 'avg_delivery_time_days',
            'delivery_delay_mean': 'avg_delivery_delay',
            'is_delayed_mean': 'pct_delayed_orders'
        })
        
        self.data['aggregated'] = agg_data
        print(f"   ✅ Datos agregados: {agg_data.shape}")
        return agg_data