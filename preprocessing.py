"""
=============================================================================
MÓDULO DE PREPROCESAMIENTO DE DATOS
=============================================================================
Carga, limpia y prepara el dataset para entrenamiento.
=============================================================================
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle
import os
import config


def load_data() -> pd.DataFrame:
    """Carga el dataset desde CSV."""
    print(f"[INFO] Cargando datos desde: {config.DATA_PATH}")
    df = pd.read_csv(
        config.DATA_PATH,
        sep=config.DATA_SEPARATOR,
        encoding=config.DATA_ENCODING,
    )
    print(f"[INFO] Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
    print(f"[INFO] Columnas disponibles: {list(df.columns)}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Limpia el dataset eliminando outliers y valores nulos."""
    initial_rows = len(df)

    # Eliminar filas con nulos en columnas críticas
    cols_to_check = [c for c in config.DROP_NA_COLS if c in df.columns]
    df = df.dropna(subset=cols_to_check)
    print(f"[INFO] Después de eliminar nulos en {cols_to_check}: {len(df)} filas")

    # Filtrar rango de precios
    if config.TARGET_COL in df.columns:
        df = df[
            (df[config.TARGET_COL] >= config.PRICE_MIN)
            & (df[config.TARGET_COL] <= config.PRICE_MAX)
        ]
        print(f"[INFO] Después de filtrar precios [{config.PRICE_MIN:,} - {config.PRICE_MAX:,}]: {len(df)} filas")

    # Filtrar rango de año (si existe la columna)
    if "año" in df.columns:
        df = df[(df["año"] >= config.YEAR_MIN) & (df["año"] <= config.YEAR_MAX)]
        print(f"[INFO] Después de filtrar años [{config.YEAR_MIN} - {config.YEAR_MAX}]: {len(df)} filas")

    # Filtrar rango de kilometraje (si existe la columna)
    if "kilometraje" in df.columns:
        df = df[
            (df["kilometraje"] >= config.KM_MIN)
            & (df["kilometraje"] <= config.KM_MAX)
        ]
        print(f"[INFO] Después de filtrar km [{config.KM_MIN:,} - {config.KM_MAX:,}]: {len(df)} filas")

    removed = initial_rows - len(df)
    print(f"[INFO] Total de filas eliminadas en limpieza: {removed} ({removed/initial_rows*100:.1f}%)")

    return df.reset_index(drop=True)


def encode_categoricals(
    df: pd.DataFrame,
    fit: bool = True,
    encoders: dict = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Aplica Label Encoding a las features categóricas.

    Args:
        df: DataFrame con los datos.
        fit: Si True, ajusta nuevos encoders. Si False, usa los proporcionados.
        encoders: Diccionario de LabelEncoders ya ajustados (para inferencia).

    Returns:
        (df_encoded, encoders_dict)
    """
    if encoders is None:
        encoders = {}

    df_encoded = df.copy()
    cat_cols = [c for c in config.CATEGORICAL_FEATURES if c in df.columns]

    for col in cat_cols:
        # Convertir a string para evitar problemas
        df_encoded[col] = df_encoded[col].astype(str)

        if fit:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col])
            encoders[col] = le
            print(f"[INFO] Encoded '{col}': {len(le.classes_)} categorías únicas")
        else:
            le = encoders[col]
            # Manejar categorías no vistas durante entrenamiento
            known = set(le.classes_)
            df_encoded[col] = df_encoded[col].apply(
                lambda x: le.transform([x])[0] if x in known else -1
            )

    return df_encoded, encoders


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separa features (X) y target (y).

    Returns:
        (X, y)
    """
    available_features = [f for f in config.ALL_FEATURES if f in df.columns]
    missing = [f for f in config.ALL_FEATURES if f not in df.columns]

    if missing:
        print(f"[WARN] Features no encontradas en el dataset: {missing}")

    X = df[available_features]
    y = df[config.TARGET_COL]

    print(f"[INFO] Features seleccionadas ({len(available_features)}): {available_features}")
    print(f"[INFO] Target: {config.TARGET_COL}")
    print(f"[INFO] X shape: {X.shape}, y shape: {y.shape}")

    return X, y


def save_encoders(encoders: dict, feature_names: list):
    """Guarda los encoders y nombres de features para uso en la API."""
    os.makedirs(os.path.dirname(config.ENCODERS_OUTPUT_PATH), exist_ok=True)

    with open(config.ENCODERS_OUTPUT_PATH, "wb") as f:
        pickle.dump(encoders, f)
    print(f"[INFO] Encoders guardados en: {config.ENCODERS_OUTPUT_PATH}")

    with open(config.FEATURE_NAMES_PATH, "wb") as f:
        pickle.dump(feature_names, f)
    print(f"[INFO] Feature names guardados en: {config.FEATURE_NAMES_PATH}")


def load_encoders() -> tuple[dict, list]:
    """Carga encoders y feature names guardados."""
    with open(config.ENCODERS_OUTPUT_PATH, "rb") as f:
        encoders = pickle.load(f)
    with open(config.FEATURE_NAMES_PATH, "rb") as f:
        feature_names = pickle.load(f)
    return encoders, feature_names


def run_preprocessing_pipeline() -> tuple[pd.DataFrame, dict]:
    """
    Pipeline completo de preprocesamiento.

    Returns:
        (df_limpio_y_encoded, encoders)
    """
    df = load_data()

    print("\n" + "=" * 60)
    print("EXPLORACIÓN INICIAL")
    print("=" * 60)
    print(f"\nPrimeras filas:\n{df.head()}")
    print(f"\nTipos de datos:\n{df.dtypes}")
    print(f"\nNulos por columna:\n{df.isnull().sum()}")
    print(f"\nEstadísticas del target ({config.TARGET_COL}):")
    if config.TARGET_COL in df.columns:
        print(df[config.TARGET_COL].describe())

    print("\n" + "=" * 60)
    print("LIMPIEZA DE DATOS")
    print("=" * 60)
    df = clean_data(df)

    print("\n" + "=" * 60)
    print("ENCODING DE CATEGÓRICAS")
    print("=" * 60)
    df, encoders = encode_categoricals(df, fit=True)

    return df, encoders


if __name__ == "__main__":
    df, encoders = run_preprocessing_pipeline()
    print("\n[OK] Preprocesamiento completado exitosamente.")
    print(f"Dataset final: {df.shape}")
