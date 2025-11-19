# src/data_loader.py
import pandas as pd
import numpy as np
from .config import Config

class DataLoader:
    """Clase para carga y limpieza de datos"""
    
    def __init__(self):
        self.data = {}
        
    def load_all_data(self):
        """Carga todos los datasets"""
        print("📥 Cargando datasets...")
        
        for dataset in Config.FILES:
            file_path = Config.get_file_path(dataset)
            if file_path.exists():
                try:
                    # Cargar orders con parse_dates especial
                    if dataset == 'orders':
                        self.data[dataset] = pd.read_csv(
                            file_path,
                            parse_dates=[
                                'order_purchase_timestamp',
                                'order_delivered_carrier_date', 
                                'order_delivered_customer_date',
                                'order_estimated_delivery_date'
                            ]
                        )
                    else:
                        self.data[dataset] = pd.read_csv(file_path)
                    
                    print(f"   ✅ {dataset}: {self.data[dataset].shape}")
                    
                except Exception as e:
                    print(f"   ⚠️  Error cargando {dataset}: {e}")
                    self.data[dataset] = pd.DataFrame()
            else:
                print(f"   ❌ Archivo no encontrado: {file_path}")
                self.data[dataset] = pd.DataFrame()
        
        return self.data
    
    def clean_orders(self):
        """Limpieza específica de orders"""
        print("🧹 Limpiando orders...")
        orders = self.data['orders'].copy()
        
        # Seleccionar columnas relevantes
        available_cols = [col for col in Config.ORDERS_COLUMNS if col in orders.columns]
        orders = orders[available_cols]
        
        # Filtrar solo pedidos entregados
        orders = orders[orders['order_status'] == 'delivered'].copy()
        
        # Eliminar duplicados
        orders = orders.drop_duplicates(subset=['order_id'])
        
        # Eliminar nulos en fecha de compra
        orders = orders.dropna(subset=['order_purchase_timestamp'])
        
        self.data['orders_clean'] = orders
        print(f"   ✅ Orders limpias: {orders.shape}")
        return orders
    
    def clean_items(self):
        """Limpieza específica de items"""
        print("🧹 Limpiando items...")
        items = self.data['items'].copy()
        
        items = items.drop_duplicates()
        items = items[items['price'] > 0]
        items = items[items['freight_value'] >= 0]
        
        self.data['items_clean'] = items
        print(f"   ✅ Items limpios: {items.shape}")
        return items
    
    def clean_products(self):
        """Limpieza específica de products"""
        print("🧹 Limpiando products...")
        products = self.data['products'].copy()
        
        # Normalizar categorías
        products['product_category_name'] = (
            products['product_category_name']
            .astype(str)
            .str.lower()
            .str.replace(" ", "_")
        )
        
        # Convertir dimensiones a numéricas
        numeric_cols = ['product_weight_g', 'product_length_cm', 'product_height_cm', 'product_width_cm']
        for col in numeric_cols:
            products[col] = pd.to_numeric(products[col], errors='coerce')
        
        products = products.drop_duplicates(subset=['product_id'])
        
        self.data['products_clean'] = products
        print(f"   ✅ Products limpios: {products.shape}")
        return products
    
    def clean_payments(self):
        """Limpieza específica de payments"""
        print("🧹 Limpiando payments...")
        payments = self.data['payments'].copy()
        
        payments = payments.drop_duplicates()
        payments['payment_value'] = pd.to_numeric(payments['payment_value'], errors='coerce')
        payments = payments[payments['payment_value'] >= 0]
        
        self.data['payments_clean'] = payments
        print(f"   ✅ Payments limpios: {payments.shape}")
        return payments
    
    def clean_reviews(self):
        """Limpieza específica de reviews"""
        print("🧹 Limpiando reviews...")
        reviews = self.data['reviews'].copy()
        
        reviews = reviews.drop_duplicates(subset=['order_id'])
        reviews['review_score'] = pd.to_numeric(reviews['review_score'], errors='coerce')
        
        # Limpiar fechas
        date_cols = ['review_creation_date', 'review_answer_timestamp']
        for col in date_cols:
            reviews[col] = pd.to_datetime(reviews[col], errors='coerce')
        
        self.data['reviews_clean'] = reviews
        print(f"   ✅ Reviews limpias: {reviews.shape}")
        return reviews
    
    def clean_all_data(self):
        """Ejecuta toda la limpieza"""
        print("\n" + "="*50)
        print("INICIANDO LIMPIEZA COMPLETA DE DATOS")
        print("="*50)
        
        self.clean_orders()
        self.clean_items() 
        self.clean_products()
        self.clean_payments()
        self.clean_reviews()
        
        print("✅ Limpieza completada!")
        return self.data