import glob
from pathlib import Path
import pandas as pd
import numpy as np

from src.config import DATA_RAW, DATA_PROCESSED, DATA_FILES
from src.data_loader import DataLoader


class DataUpdater:
    """
    Clase para actualización mensual de datos
    """

    def __init__(self) -> None:
        self.raw_path = DATA_RAW
        self.nuevos_path = self.raw_path / "nuevos"
        self.processed_path = DATA_PROCESSED

        self.tipos_tablas = {
            'orders': DATA_FILES['orders'],
            'items': DATA_FILES['items'],
            'products': DATA_FILES['products'],
            'reviews': DATA_FILES['reviews'],
            'payments': DATA_FILES['payments'],
        }

    def _listar_archivos_nuevos_por_tipo(self, tipo: str) -> list[Path]:
        """Listar archivos nuevos para un tipo dado"""
        patron = str(self.nuevos_path / f"nuevos_*{tipo}*.csv")
        return [Path(p) for p in glob.glob(patron)]

    def _limpiar_y_validar_datos_nuevos(self, df: pd.DataFrame, tipo: str) -> pd.DataFrame:
        """Limpiar y validar datos nuevos antes de combinarlos"""
        df_limpio = df.copy()
        
        try:
            # SOLO ORDERS se filtran por fecha
            if tipo == 'orders':
                # Convertir columnas de fecha
                date_columns = [col for col in df_limpio.columns if 'date' in col.lower() or 'timestamp' in col.lower()]
                for col in date_columns:
                    df_limpio[col] = pd.to_datetime(df_limpio[col], errors='coerce')
                
                # Filtrar solo órdenes de octubre 2018
                if 'order_purchase_timestamp' in df_limpio.columns:
                    mask_oct_2018 = (
                        (df_limpio['order_purchase_timestamp'].dt.year == 2018) & 
                        (df_limpio['order_purchase_timestamp'].dt.month == 10)
                    )
                    df_limpio = df_limpio[mask_oct_2018]
            
            # ITEMS - Solo convertir fechas, NO filtrar
            elif tipo == 'items':
                if 'shipping_limit_date' in df_limpio.columns:
                    df_limpio['shipping_limit_date'] = pd.to_datetime(df_limpio['shipping_limit_date'], errors='coerce')
            
            # REVIEWS - Solo convertir fechas, NO filtrar
            elif tipo == 'reviews':
                if 'review_creation_date' in df_limpio.columns:
                    df_limpio['review_creation_date'] = pd.to_datetime(df_limpio['review_creation_date'], errors='coerce')
            
            # Eliminar filas con fechas inválidas
            date_columns = [col for col in df_limpio.columns if pd.api.types.is_datetime64_any_dtype(df_limpio[col])]
            for col in date_columns:
                df_limpio = df_limpio[df_limpio[col].notna()]
            
            # Manejar tipos numéricos
            numeric_columns = [col for col in df_limpio.columns if 'price' in col.lower() or 'value' in col.lower() or 'payment' in col.lower()]
            for col in numeric_columns:
                if col in df_limpio.columns:
                    df_limpio[col] = pd.to_numeric(df_limpio[col], errors='coerce')
                    
        except Exception:
            return df.copy()
        
        return df_limpio

    def cargar_y_combinar_tablas(self) -> dict[str, pd.DataFrame]:
        """Cargar y combinar todas las tablas relevantes"""
        print("🔄 Cargando y combinando todas las tablas...")

        datos_combinados = {}

        for tipo, ruta_base in self.tipos_tablas.items():
            print(f"Procesando: {tipo}")

            # 1. Cargar datos BASE
            df_base = pd.DataFrame()
            if ruta_base.exists():
                try:
                    df_base = pd.read_csv(ruta_base)
                except Exception:
                    continue

            # 2. Cargar datos NUEVOS
            archivos_nuevos = self._listar_archivos_nuevos_por_tipo(tipo)
            datos_nuevos = []

            if archivos_nuevos:
                for archivo in archivos_nuevos:
                    try:
                        print(f"Cargando: {archivo.name}")
                        df_nuevo = pd.read_csv(archivo)
                        df_nuevo_limpio = self._limpiar_y_validar_datos_nuevos(df_nuevo, tipo)
                        if len(df_nuevo_limpio) > 0:
                            datos_nuevos.append(df_nuevo_limpio)
                    except Exception:
                        continue

            # 3. Combinar base + nuevos
            todos_datos = []
            if not df_base.empty:
                todos_datos.append(df_base)
            todos_datos.extend(datos_nuevos)

            if todos_datos:
                try:
                    df_final = pd.concat(todos_datos, ignore_index=True, sort=False)
                    
                    # Eliminar duplicados
                    if tipo == 'orders' and 'order_id' in df_final.columns:
                        df_final = df_final.drop_duplicates(subset=['order_id'])
                    elif tipo == 'items' and 'order_id' in df_final.columns and 'product_id' in df_final.columns:
                        df_final = df_final.drop_duplicates(subset=['order_id', 'product_id'])
                    
                    datos_combinados[tipo] = df_final
                    print(f"✅ {tipo}: {len(df_final)} registros")
                    
                except Exception:
                    if not df_base.empty:
                        datos_combinados[tipo] = df_base

        return datos_combinados

    def procesar_datos_completos(self) -> pd.DataFrame | None:
        """Ejecutar pipeline completo de procesamiento"""
        print("🚀 Iniciando procesamiento completo de datos...")

        tablas = self.cargar_y_combinar_tablas()

        tablas_requeridas = ['orders', 'items', 'products', 'reviews', 'payments']
        tablas_faltantes = [t for t in tablas_requeridas if t not in tablas]

        if tablas_faltantes:
            print(f"❌ Tablas faltantes: {tablas_faltantes}")
            return None

        try:
            data_loader = DataLoader()

            print("🧹 Realizando limpieza de datos...")
            df_limpio = data_loader.clean_data(
                tablas['orders'],
                tablas['items'],
                tablas['products'],
                tablas['reviews'],
                tablas['payments']
            )

            data_loader.save_processed_data(df_limpio)

            print("✅ Procesamiento completo terminado")
            print(f"📊 Dataset final: {df_limpio.shape}")

            return df_limpio

        except Exception as e:
            print(f"❌ Error en procesamiento: {e}")
            return None

    def verificar_nuevos_datos(self) -> bool:
        """Verificar si existen archivos nuevos"""
        patron = str(self.nuevos_path / "nuevos_*.csv")
        archivos_nuevos = [Path(p) for p in glob.glob(patron)]

        if archivos_nuevos:
            print(f"📥 Archivos nuevos encontrados: {len(archivos_nuevos)}")
            return True
        else:
            print("📭 No se encontraron archivos nuevos")
            return False

    def obtener_info_datos_combinados(self) -> dict[str, pd.DataFrame]:
        """Información de datos combinados"""
        tablas = self.cargar_y_combinar_tablas()

        print("\n📊 RESUMEN DATOS COMBINADOS")
        print("=" * 40)
        for tipo, df in tablas.items():
            print(f"   {tipo:10}: {len(df):6} registros")
        print("=" * 40)

        return tablas


def ejecutar_actualizacion_mensual() -> bool:
    """Ejecutar flujo completo de actualización mensual"""
    print("=" * 60)
    print("🔄 EJECUTANDO ACTUALIZACIÓN MENSUAL")
    print("=" * 60)

    updater = DataUpdater()

    if not updater.verificar_nuevos_datos():
        print("❌ No hay datos nuevos para procesar")
        return False

    print("\n🔍 Verificando datos combinados...")
    updater.obtener_info_datos_combinados()

    try:
        df_actualizado = updater.procesar_datos_completos()

        if df_actualizado is not None:
            print("✅ ¡Actualización completada exitosamente!")
            return True
        else:
            print("❌ Error en el procesamiento")
            return False

    except Exception as e:
        print(f"❌ Error en actualización: {e}")
        return False
