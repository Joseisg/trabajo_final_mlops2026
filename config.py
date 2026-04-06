"""
=============================================================================
CONFIGURACIÓN DEL MODELO - PREDICTOR DE PRECIOS DE AUTOMÓVILES
=============================================================================
Archivo centralizado de configuración.
Modificar SOLO este archivo para adaptar el modelo a tu dataset.
=============================================================================
"""

# =============================================================================
# 1. DATASET
# =============================================================================
DATA_PATH = "ml_raw_detalle.csv"       # Ruta al archivo CSV
DATA_SEPARATOR = ","                    # Separador del CSV ("," o ";")
DATA_ENCODING = "utf-8"                # Encoding del CSV ("utf-8", "latin-1", etc.)

# =============================================================================
# 2. VARIABLE OBJETIVO (Target)
# =============================================================================
TARGET_COL = "precio"                   # Nombre de la columna de precio

# =============================================================================
# 3. FEATURES - Definir qué columnas usar y de qué tipo son
# =============================================================================
# Features numéricas (año, kilometraje, cilindrada, etc.)
NUMERIC_FEATURES = [
    "año",
    "kilometraje",
    "antiguedad",
    "km_por_ano",
    "log_km",
]

# Features categóricas (marca, modelo, combustible, transmisión, etc.)
CATEGORICAL_FEATURES = [
    "marca",
    "modelo",
    "tipo_de_combustible",
    "transmision",
    "tipo_de_carroceria",
]

# Lista completa de features (se genera automáticamente)
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# =============================================================================
# 4. LIMPIEZA DE DATOS
# =============================================================================
# Rango válido de precios (elimina outliers extremos)
PRICE_MIN = 500_000          # Precio mínimo en CLP (ej: 500.000)
PRICE_MAX = 150_000_000      # Precio máximo en CLP (ej: 150.000.000)

# Rango válido de año
YEAR_MIN = 1990
YEAR_MAX = 2026

# Rango válido de kilometraje
KM_MIN = 0
KM_MAX = 500_000

# Columnas que no deben tener valores nulos (se eliminan esas filas)
DROP_NA_COLS = ["precio", "marca", "modelo", "año", "kilometraje"]

# =============================================================================
# 5. MODELO - HIPERPARÁMETROS LIGHTGBM
# =============================================================================
LGBM_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "n_estimators": 1000,
    "learning_rate": 0.05,
    "max_depth": 8,
    "num_leaves": 63,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "random_state": 42,
    "verbose": -1,
    "n_jobs": -1,
    # Restricciones monótonas: año(+), km(-), antiguedad(-), km_por_ano(-), log_km(-), resto(0)
    # Orden: año, kilometraje, antiguedad, km_por_ano, log_km, marca, modelo, combustible, transmision, carroceria
    "monotone_constraints": [1, -1, -1, -1, -1, 0, 0, 0, 0, 0],
}

# Early stopping
EARLY_STOPPING_ROUNDS = 50

# =============================================================================
# 6. ENTRENAMIENTO
# =============================================================================
TEST_SIZE = 0.2                # Proporción de datos para test
RANDOM_STATE = 42              # Semilla para reproducibilidad

# =============================================================================
# 7. EXPORTACIÓN
# =============================================================================
MODEL_OUTPUT_PATH = "model/model.pkl"
ENCODERS_OUTPUT_PATH = "model/label_encoders.pkl"
FEATURE_NAMES_PATH = "model/feature_names.pkl"

# =============================================================================
# 8. API
# =============================================================================
API_TITLE = "Predictor de Precios de Automóviles"
API_DESCRIPTION = "API para estimar el precio de mercado de un automóvil usado"
API_VERSION = "1.0.0"
