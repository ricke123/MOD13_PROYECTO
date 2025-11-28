"""
Módulo para carga y limpieza de datos - VERSIÓN CORREGIDA
"""
import pandas as pd
from src.config import DATA_FILES, DATE_COLS, DATA_PROCESSED

class DataLoader:
    def __init__(self):
        self.data_files = DATA_FILES
        self.date_cols = DATE_COLS
        self.processed_path = DATA_PROCESSED
        
    def load_raw_data(self):
        """Cargar todos los datasets raw"""
        print("📥 Cargando datasets raw...")
        
        # Cargar orders con parseo de fechas
        orders = pd.read_csv(
            self.data_files['orders'],
            parse_dates=self.date_cols['orders']
        )
        
        # Cargar resto de datasets
        items = pd.read_csv(self.data_files['items'])
        products = pd.read_csv(self.data_files['products'])
        reviews = pd.read_csv(self.data_files['reviews'])
        payments = pd.read_csv(self.data_files['payments'])
        
        print("✅ Datasets cargados exitosamente")
        print(f"Orders:   {orders.shape}")
        print(f"Items:    {items.shape}")
        print(f"Products: {products.shape}")
        print(f"Reviews:  {reviews.shape}")
        print(f"Payments: {payments.shape}")
        
        return orders, items, products, reviews, payments
    
    def clean_data(self, orders, items, products, reviews, payments):
        """Realizar limpieza básica de datos"""
        print("🧹 Realizando limpieza de datos...")
        
        # Asegurar que las columnas de fecha sean datetime
        for date_col in self.date_cols['orders']:
            if date_col in orders.columns:
                orders[date_col] = pd.to_datetime(orders[date_col], errors='coerce')
                print(f"✅ Convertida {date_col} a datetime")
        
        # Filtrar solo órdenes entregadas
        df_delivered = orders[orders['order_status'] == 'delivered'].copy()
        print(f"Órdenes totales:     {len(orders)}")
        print(f"Órdenes entregadas:  {len(df_delivered)}")
        
        # Unir todos los datasets SOLO con órdenes entregadas
        df = df_delivered.merge(items, on='order_id', how='left')
        df = df.merge(products, on='product_id', how='left')
        df = df.merge(reviews, on='order_id', how='left')
        df = df.merge(payments, on='order_id', how='left')
        
        print(f"Dataset unificado (solo delivered): {df.shape}")
        
        return df
    
    def load_processed_data(self):
        """Cargar datos ya procesados si existen"""
        processed_file = self.processed_path / "processed_data.csv"
        if processed_file.exists():
            print("📥 Cargando datos procesados...")
            df = pd.read_csv(processed_file)
            
            # Convertir columnas de fecha al cargar
            date_columns = [
                'order_purchase_timestamp',
                'order_delivered_carrier_date',
                'order_delivered_customer_date',
                'order_estimated_delivery_date'
            ]
            
            for col in date_columns:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    print(f"✅ Convertida {col} a datetime")
            
            return df
        else:
            print("❌ No se encontraron datos procesados")
            return None
    
    def save_processed_data(self, df):
        """Guardar datos procesados"""
        processed_file = self.processed_path / "processed_data.csv"
        df.to_csv(processed_file, index=False)
        print(f"💾 Datos procesados guardados en: {processed_file}")
