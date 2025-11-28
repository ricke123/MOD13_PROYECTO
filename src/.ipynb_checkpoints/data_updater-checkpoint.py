
"""
Módulo para actualización mensual de datos - VERSIÓN CORREGIDA
"""
import pandas as pd
import glob
from pathlib import Path
from datetime import datetime

class DataUpdater:
    def __init__(self):
        self.raw_path = Path("data/raw/")
        self.processed_path = Path("data/processed/")
        
    def cargar_y_combinar_tablas(self):
        """Cargar y combinar todas las tablas - VERSIÓN CORREGIDA Y ROBUSTA"""
        print("🔄 Cargando y combinando todas las tablas...")
        
        # INICIALIZAR el diccionario aquí
        datos_combinados = {}
        
        # Mapeo de tipos de tablas a sus archivos base
        tipos_tablas = {
            'orders': 'olist_orders_dataset.csv',
            'items': 'olist_order_items_dataset.csv', 
            'products': 'olist_products_dataset.csv',
            'reviews': 'olist_order_reviews_dataset.csv',
            'payments': 'olist_order_payments_dataset.csv'
        }
        
        for tipo, archivo_base in tipos_tablas.items():
            print(f"📊 Procesando: {tipo}")
            
            # 1. Cargar datos BASE (archivos principales en raw/)
            archivo_base_path = self.raw_path / archivo_base
            if archivo_base_path.exists():
                try:
                    df_base = pd.read_csv(archivo_base_path)
                    print(f"   📁 Base: {archivo_base} - {len(df_base)} registros")
                except Exception as e:
                    print(f"   ❌ Error cargando base {archivo_base}: {e}")
                    df_base = pd.DataFrame()
            else:
                print(f"   ⚠️  No se encontró archivo base: {archivo_base}")
                df_base = pd.DataFrame()
            
            # 2. Cargar datos NUEVOS (archivos en nuevos/)
            archivos_nuevos = glob.glob(f"{self.raw_path}/nuevos/nuevos_*{tipo}*.csv")
            datos_nuevos = []
            
            for archivo in archivos_nuevos:
                try:
                    df_nuevo = pd.read_csv(archivo)
                    datos_nuevos.append(df_nuevo)
                    print(f"   🆕 Nuevo: {Path(archivo).name} - {len(df_nuevo)} registros")
                except Exception as e:
                    print(f"   ❌ Error cargando nuevo archivo {archivo}: {e}")
            
            # 3. Combinar todos los datos
            todos_datos = []
            if not df_base.empty:
                todos_datos.append(df_base)
            todos_datos.extend(datos_nuevos)
            
            if todos_datos:
                try:
                    datos_combinados[tipo] = pd.concat(todos_datos, ignore_index=True)
                    print(f"   ✅ {tipo}: {len(datos_combinados[tipo])} registros totales")
                except Exception as e:
                    print(f"   ❌ Error combinando datos para {tipo}: {e}")
                    # Si hay error, usar solo los datos base
                    if not df_base.empty:
                        datos_combinados[tipo] = df_base
                        print(f"   ⚠️  Usando solo datos base para {tipo}")
                    else:
                        print(f"   💥 No hay datos disponibles para {tipo}")
            else:
                print(f"   ❌ No se encontraron datos para {tipo}")
        
        return datos_combinados
    
    def procesar_datos_completos(self):
        """Ejecutar pipeline completo de procesamiento - VERSIÓN ROBUSTA"""
        print("🚀 Iniciando procesamiento completo de datos...")
        
        # 1. Cargar y combinar datos
        tablas = self.cargar_y_combinar_tablas()
        
        # Verificar que tenemos todas las tablas necesarias
        tablas_requeridas = ['orders', 'items', 'products', 'reviews', 'payments']
        tablas_faltantes = []
        
        for tabla in tablas_requeridas:
            if tabla not in tablas:
                tablas_faltantes.append(tabla)
                print(f"❌ Faltan datos para: {tabla}")
        
        if tablas_faltantes:
            print(f"💥 No se pueden procesar los datos. Tablas faltantes: {tablas_faltantes}")
            return None
        
        # 2. Usar tu DataLoader existente para limpieza
        try:
            from data_loader import DataLoader
            data_loader = DataLoader()
            
            # 3. Realizar limpieza (igual que en tu código actual)
            print("🧹 Realizando limpieza de datos...")
            df_limpio = data_loader.clean_data(
                tablas['orders'],
                tablas['items'], 
                tablas['products'],
                tablas['reviews'],
                tablas['payments']
            )
            
            # 4. Guardar datos limpios
            data_loader.save_processed_data(df_limpio)
            
            print(f"✅ Procesamiento completo terminado")
            print(f"📊 Dataset final: {df_limpio.shape}")
            
            return df_limpio
            
        except Exception as e:
            print(f"❌ Error en el procesamiento de datos: {e}")
            return None
    
    def verificar_nuevos_datos(self):
        """Verificar si hay nuevos datos disponibles - VERSIÓN MEJORADA"""
        archivos_nuevos = glob.glob(f"{self.raw_path}/nuevos/nuevos_*.csv")
        if archivos_nuevos:
            print(f"📥 Se encontraron {len(archivos_nuevos)} archivos nuevos:")
            for archivo in archivos_nuevos:
                try:
                    df = pd.read_csv(archivo, nrows=1)  # Solo leer primera fila para info
                    print(f"   📄 {Path(archivo).name} - Columnas: {len(df.columns)}")
                except Exception as e:
                    print(f"   ❌ {Path(archivo).name} - Error al leer: {e}")
            return True
        else:
            print("📭 No se encontraron archivos nuevos en data/raw/nuevos/")
            print("💡 Los archivos deben llamarse: nuevos_orders_YYYY_MM.csv, nuevos_items_YYYY_MM.csv, etc.")
            return False

    def obtener_info_datos_combinados(self):
        """Obtener información sobre los datos combinados (para debugging)"""
        tablas = self.cargar_y_combinar_tablas()
        
        print("\n📊 INFORMACIÓN DE DATOS COMBINADOS:")
        print("=" * 40)
        for tipo, df in tablas.items():
            print(f"   {tipo:10}: {len(df):6} registros, {len(df.columns):2} columnas")
        print("=" * 40)
        
        return tablas

def ejecutar_actualizacion_mensual():
    """Función principal para ejecutar la actualización mensual - VERSIÓN MEJORADA"""
    print("=" * 60)
    print("🔄 EJECUTANDO ACTUALIZACIÓN MENSUAL DE DATOS")
    print("=" * 60)
    
    updater = DataUpdater()
    
    # Verificar datos nuevos
    if not updater.verificar_nuevos_datos():
        print("❌ No hay datos nuevos para procesar")
        return False
    
    # Para debugging, mostrar info de datos combinados
    print("\n🔍 Verificando datos combinados...")
    updater.obtener_info_datos_combinados()
    
    # Ejecutar procesamiento completo
    try:
        df_actualizado = updater.procesar_datos_completos()
        
        if df_actualizado is not None:
            print("✅ ¡Actualización mensual completada exitosamente!")
            return True
        else:
            print("❌ Error en el procesamiento de datos")
            return False
            
    except Exception as e:
        print(f"❌ Error en la actualización: {e}")
        import traceback
        print(f"📝 Detalles del error:")
        traceback.print_exc()
        return False








