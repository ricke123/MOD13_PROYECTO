"""
Módulo para actualización mensual de datos - VERSIÓN MEJORADA
- Combina datos base + nuevos por tabla (orders, items, etc.)
- Reutiliza rutas y archivos desde config.py
- Usa DataLoader para aplicar la misma limpieza que el pipeline principal
"""

import glob
from pathlib import Path
import pandas as pd

from src.config import DATA_RAW, DATA_PROCESSED, DATA_FILES
from src.data_loader import DataLoader


class DataUpdater:
    """
    Clase encargada de:
    - Detectar archivos nuevos en data/raw/nuevos/
    - Combinar datos base + nuevos por tipo de tabla
    - Pasar todo por el mismo proceso de limpieza de DataLoader
    """

    def __init__(self) -> None:
        # Usamos las rutas globales definidas en config.py
        self.raw_path = DATA_RAW              # data/raw/
        self.nuevos_path = self.raw_path / "nuevos"
        self.processed_path = DATA_PROCESSED  # data/processed/

        # Mapeo de tipos de tablas a sus archivos base (rutas completas)
        self.tipos_tablas = {
            'orders':   DATA_FILES['orders'],
            'items':    DATA_FILES['items'],
            'products': DATA_FILES['products'],
            'reviews':  DATA_FILES['reviews'],
            'payments': DATA_FILES['payments'],
        }

    # ------------------------------------------------------------------
    # UTILIDAD: listar archivos nuevos por tipo
    # ------------------------------------------------------------------
    def _listar_archivos_nuevos_por_tipo(self, tipo: str) -> list[Path]:
        """
        Lista archivos nuevos para un tipo dado (orders, items, etc.)
        Convención: data/raw/nuevos/nuevos_<loquesea><tipo><loquesea>.csv
        Ejemplo: nuevos_orders_2025_08.csv
        """
        patron = str(self.nuevos_path / f"nuevos_*{tipo}*.csv")
        rutas = [Path(p) for p in glob.glob(patron)]
        return rutas

    # ------------------------------------------------------------------
    # Cargar y combinar tablas base + nuevas
    # ------------------------------------------------------------------
    def cargar_y_combinar_tablas(self) -> dict[str, pd.DataFrame]:
        """
        Cargar y combinar todas las tablas relevantes:
        - orders, items, products, reviews, payments
        Devuelve un diccionario {tipo: DataFrame}
        """
        print("🔄 Cargando y combinando todas las tablas...")

        datos_combinados: dict[str, pd.DataFrame] = {}

        for tipo, ruta_base in self.tipos_tablas.items():
            print(f"\n📊 Procesando: {tipo}")

            # 1. Cargar datos BASE
            df_base = pd.DataFrame()
            if ruta_base.exists():
                try:
                    df_base = pd.read_csv(ruta_base)
                    print(f"   📁 Base: {ruta_base.name} - {len(df_base)} registros")
                except Exception as e:
                    print(f"   ❌ Error cargando base {ruta_base.name}: {e}")
            else:
                print(f"   ⚠️ No se encontró archivo base: {ruta_base}")

            # 2. Cargar datos NUEVOS
            archivos_nuevos = self._listar_archivos_nuevos_por_tipo(tipo)
            datos_nuevos: list[pd.DataFrame] = []

            if archivos_nuevos:
                for archivo in archivos_nuevos:
                    try:
                        df_nuevo = pd.read_csv(archivo)
                        datos_nuevos.append(df_nuevo)
                        print(f"   🆕 Nuevo: {archivo.name} - {len(df_nuevo)} registros")
                    except Exception as e:
                        print(f"   ❌ Error cargando nuevo archivo {archivo.name}: {e}")
            else:
                print("   📭 No se encontraron archivos nuevos para este tipo")

            # 3. Combinar base + nuevos
            todos_datos: list[pd.DataFrame] = []
            if not df_base.empty:
                todos_datos.append(df_base)
            todos_datos.extend(datos_nuevos)

            if todos_datos:
                try:
                    df_final = pd.concat(todos_datos, ignore_index=True)
                    datos_combinados[tipo] = df_final
                    print(f"   ✅ {tipo}: {len(df_final)} registros totales")
                except Exception as e:
                    print(f"   ❌ Error combinando datos para {tipo}: {e}")
                    if not df_base.empty:
                        datos_combinados[tipo] = df_base
                        print(f"   ⚠️ Usando solo datos base para {tipo}")
                    else:
                        print(f"   💥 No hay datos disponibles para {tipo}")
            else:
                print(f"   ❌ No se encontraron datos para {tipo}")

        return datos_combinados

    # ------------------------------------------------------------------
    # Ejecutar procesamiento completo (limpieza con DataLoader)
    # ------------------------------------------------------------------
    def procesar_datos_completos(self) -> pd.DataFrame | None:
        """
        Ejecuta el pipeline de:
        - Cargar + combinar tablas base + nuevas
        - Limpiar con DataLoader (filtro delivered + merges)
        - Guardar processed_data.csv en data/processed/
        Devuelve el DataFrame limpio o None si algo falla.
        """
        print("\n🚀 Iniciando procesamiento completo de datos...")

        tablas = self.cargar_y_combinar_tablas()

        tablas_requeridas = ['orders', 'items', 'products', 'reviews', 'payments']
        tablas_faltantes = [t for t in tablas_requeridas if t not in tablas]

        if tablas_faltantes:
            print(f"💥 No se pueden procesar los datos. Tablas faltantes: {tablas_faltantes}")
            return None

        try:
            data_loader = DataLoader()

            print("🧹 Realizando limpieza de datos (DataLoader.clean_data)...")
            df_limpio = data_loader.clean_data(
                tablas['orders'],
                tablas['items'],
                tablas['products'],
                tablas['reviews'],
                tablas['payments']
            )

            data_loader.save_processed_data(df_limpio)

            print("✅ Procesamiento completo terminado")
            print(f"📊 Dataset limpio unificado: {df_limpio.shape}")

            return df_limpio

        except Exception as e:
            print(f"❌ Error en el procesamiento de datos: {e}")
            return None

    # ------------------------------------------------------------------
    # Verificar si hay nuevos datos en data/raw/nuevos
    # ------------------------------------------------------------------
    def verificar_nuevos_datos(self) -> bool:
        """
        Verifica si existen archivos nuevos en data/raw/nuevos/.
        Devuelve True si hay al menos un archivo nuevos_*.csv
        """
        patron = str(self.nuevos_path / "nuevos_*.csv")
        archivos_nuevos = [Path(p) for p in glob.glob(patron)]

        if archivos_nuevos:
            print(f"📥 Se encontraron {len(archivos_nuevos)} archivos nuevos:")
            for archivo in archivos_nuevos:
                try:
                    df = pd.read_csv(archivo, nrows=1)
                    print(f"   📄 {archivo.name} - Columnas: {len(df.columns)}")
                except Exception as e:
                    print(f"   ❌ {archivo.name} - Error al leer: {e}")
            return True
        else:
            print("📭 No se encontraron archivos nuevos en data/raw/nuevos/")
            print("💡 Convención sugerida: nuevos_orders_YYYY_MM.csv, nuevos_items_YYYY_MM.csv, etc.")
            return False

    # ------------------------------------------------------------------
    # Info rápida de datos combinados (debug)
    # ------------------------------------------------------------------
    def obtener_info_datos_combinados(self) -> dict[str, pd.DataFrame]:
        """
        Carga y combina tablas, e imprime un resumen rápido por tipo.
        Devuelve el diccionario {tipo: DataFrame}
        """
        tablas = self.cargar_y_combinar_tablas()

        print("\n📊 INFORMACIÓN DE DATOS COMBINADOS")
        print("=" * 50)
        for tipo, df in tablas.items():
            print(f"   {tipo:10}: {len(df):6} registros, {len(df.columns):3} columnas")
        print("=" * 50)

        return tablas


# ----------------------------------------------------------------------
# Función de conveniencia para lanzar la actualización mensual
# ----------------------------------------------------------------------
def ejecutar_actualizacion_mensual() -> bool:
    """
    Ejecuta el flujo completo de actualización mensual:
    - Verifica si hay nuevos datos
    - Muestra info combinada (debug)
    - Ejecuta el procesamiento completo con DataLoader
    Devuelve True si todo va bien, False si no.
    """
    print("=" * 60)
    print("🔄 EJECUTANDO ACTUALIZACIÓN MENSUAL DE DATOS")
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
            print("✅ ¡Actualización mensual completada exitosamente!")
            return True
        else:
            print("❌ Error en el procesamiento de datos")
            return False

    except Exception as e:
        print(f"❌ Error en la actualización: {e}")
        import traceback
        print("📝 Detalles del error:")
        traceback.print_exc()
        return False







