"""
Módulo de análisis de correlación y selección de features
- Calcula matriz de correlación de todas las features numéricas
- Selecciona hasta N=99 features poco colineales entre sí
  priorizando mayor |correlación| con la variable target
- Guarda resultados en data/model/
"""

import numpy as np
import pandas as pd
from pathlib import Path

from src.config import MODEL_DIR, MODEL_CONFIG


class FeatureCorrelationAnalyzer:
    def __init__(
        self,
        target_col: str = None,
        corr_threshold: float = 0.90,
        top_n: int = 99,
    ):
        """
        Parameters
        ----------
        target_col : str
            Nombre de la columna target. Por defecto usa MODEL_CONFIG['target_col'].
        corr_threshold : float
            Umbral de correlación absoluta entre features para considerar que
            dos columnas están "demasiado colineales" (se descarta una).
        top_n : int
            Número máximo de features a seleccionar.
        """
        self.target_col = target_col or MODEL_CONFIG.get(
            "target_col", "demand_next_month"
        )
        self.corr_threshold = corr_threshold
        self.top_n = top_n
        self.model_dir: Path = MODEL_DIR

    # ------------------------------------------------------------------ #
    # 1. Cálculo de matriz de correlación
    # ------------------------------------------------------------------ #
    def compute_correlation_matrix(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula la matriz de correlación de Pearson solo para columnas numéricas.

        Parameters
        ----------
        features_df : pd.DataFrame
            DataFrame con las features y la columna target.

        Returns
        -------
        corr_matrix : pd.DataFrame
            Matriz de correlación entre todas las columnas numéricas.
        """
        print("📈 Calculando matriz de correlación (Pearson) ...")

        # Solo columnas numéricas
        numeric_df = features_df.select_dtypes(include=[np.number]).copy()

        if self.target_col not in numeric_df.columns:
            raise ValueError(
                f"❌ La columna target '{self.target_col}' no está en el DataFrame numérico."
            )

        print(f"   • Columnas numéricas totales: {numeric_df.shape[1]}")
        corr_matrix = numeric_df.corr(method="pearson")

        return corr_matrix

    # ------------------------------------------------------------------ #
    # 2. Selección de features basada en correlación con la target
    #    y eliminación de colinealidad entre features
    # ------------------------------------------------------------------ #
    def select_top_features(
        self, features_df: pd.DataFrame, save_results: bool = True
    ):
        """
        Selecciona hasta top_n features:
        - Mayor |correlación| con la target
        - Eliminando columnas muy colineales entre sí (|corr| >= corr_threshold)

        Parameters
        ----------
        features_df : pd.DataFrame
            DataFrame con todas las features y la columna target.
        save_results : bool
            Si True, guarda la matriz de correlación y las features seleccionadas en disco.

        Returns
        -------
        result : dict
            {
                "selected_features": list[str],
                "selected_df": pd.DataFrame,
                "correlation_matrix": pd.DataFrame
            }
        """
        print("🔍 Iniciando selección de features por correlación...")
        print(f"   • Target: {self.target_col}")
        print(f"   • Umbral colinealidad: |corr| ≥ {self.corr_threshold}")
        print(f"   • Máximo de features a seleccionar: {self.top_n}")

        # 1) Matriz de correlación completa
        corr_matrix = self.compute_correlation_matrix(features_df)

        # 2) Correlación de cada feature con la target (excluyendo la propia target)
        target_corr = corr_matrix[self.target_col].drop(
            labels=[self.target_col], errors="ignore"
        )
        target_corr = target_corr.dropna()

        # Ordenar por correlación absoluta descendente
        target_corr = target_corr.reindex(
            target_corr.abs().sort_values(ascending=False).index
        )

        print(f"   • Features candidatas (numéricas): {len(target_corr)}")

        # 3) Selección greedy para evitar alta colinealidad entre features
        selected_features: list[str] = []

        for feature in target_corr.index:
            if len(selected_features) >= self.top_n:
                break

            # Chequear colinealidad con las ya seleccionadas
            keep = True
            for sel in selected_features:
                if abs(corr_matrix.loc[feature, sel]) >= self.corr_threshold:
                    keep = False
                    break

            if keep:
                selected_features.append(feature)

        print(f"✅ Features seleccionadas: {len(selected_features)}")

        # DataFrame con info de las features seleccionadas
        selected_df = pd.DataFrame(
            {
                "feature": selected_features,
                "corr_with_target": target_corr[selected_features].values,
            }
        )

        # ------------------------------------------------------------------
        # 4) Guardar resultados
        # ------------------------------------------------------------------
        if save_results:
            self.model_dir.mkdir(parents=True, exist_ok=True)

            # 4.1 Matriz de correlación completa (para análisis en notebook)
            corr_path = self.model_dir / "feature_correlation_matrix.csv"
            corr_matrix.to_csv(corr_path)
            print(f"💾 Matriz de correlación guardada en: {corr_path}")

            # 4.2 Lista de features seleccionadas
            selected_path = self.model_dir / "selected_features_corr_top_99.csv"
            selected_df.to_csv(selected_path, index=False)
            print(f"💾 Features seleccionadas guardadas en: {selected_path}")

        return {
            "selected_features": selected_features,
            "selected_df": selected_df,
            "correlation_matrix": corr_matrix,
        }

    # ------------------------------------------------------------------ #
    # 3. Construir dataset reducido para el modelo
    # ------------------------------------------------------------------ #
    def reduce_dataset(self, df: pd.DataFrame, selected_features: list) -> pd.DataFrame:
        """
        Devuelve un DataFrame reducido que contiene:
          - purchase_year_month (si existe)
          - target
          - features seleccionadas

        Esto permite mantener el split temporal en ModelTrainer.
        """
        print("📦 Construyendo DataFrame reducido para modelado...")

        cols = []

        # Mantener fecha si existe
        if "purchase_year_month" in df.columns:
            cols.append("purchase_year_month")

        # Target
        if self.target_col in df.columns:
            cols.append(self.target_col)

        # Features seleccionadas
        cols.extend([f for f in selected_features if f in df.columns])

        # Quitar duplicados preservando orden
        seen = set()
        final_cols = []
        for c in cols:
            if c not in seen:
                final_cols.append(c)
                seen.add(c)

        reduced = df[final_cols].copy()
        print(f"📊 DF reducido para modelo: {reduced.shape}")
        return reduced


# ----------------------------------------------------------------------
# Uso de ejemplo (para pruebas manuales)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("⚙️  Test rápido de FeatureCorrelationAnalyzer")
    print("💡 Normalmente se usa importándolo desde un notebook o main.py\n")

    print(
        """
Ejemplo de uso en notebook o script:

from feature_engineer import FeatureEngineer
from data_loader import DataLoader
from feature_correlation import FeatureCorrelationAnalyzer

# 1) Cargar datos procesados
loader = DataLoader()
df = loader.load_processed_data()

# 2) Crear target mensual y features avanzados
eng = FeatureEngineer()
monthly = eng.create_target_variable(df)
features = eng.create_advanced_features(monthly)
features_final = eng.prepare_model_features(features)

# 3) Analizar correlación y seleccionar 99 features
analyzer = FeatureCorrelationAnalyzer(corr_threshold=0.90, top_n=99)
result = analyzer.select_top_features(features_final, save_results=True)

selected = result['selected_features']
reduced_df = analyzer.reduce_dataset(features_final, selected)
print(reduced_df.shape)
"""
    )
