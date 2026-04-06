"""
=============================================================================
API REST - PREDICTOR DE PRECIOS DE AUTOMÓVILES
=============================================================================
FastAPI + Uvicorn para servir el modelo de predicción.
=============================================================================
"""

import pickle
import os
import json
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import config

# =============================================================================
# Cargar configuración
# =============================================================================
# Paths relativos al directorio de la app
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "model.pkl")
ENCODERS_PATH = os.path.join(BASE_DIR, "model", "label_encoders.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "model", "feature_names.pkl")

# =============================================================================
# Cargar modelo y artefactos al iniciar la app
# =============================================================================
print("[INFO] Cargando modelo y artefactos...")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)
print(f"[OK] Modelo cargado desde {MODEL_PATH}")

with open(ENCODERS_PATH, "rb") as f:
    label_encoders = pickle.load(f)
print(f"[OK] Encoders cargados: {list(label_encoders.keys())}")

with open(FEATURES_PATH, "rb") as f:
    feature_names = pickle.load(f)
print(f"[OK] Features: {feature_names}")

# =============================================================================
# Definir la app FastAPI
# =============================================================================
app = FastAPI(
    title=config.API_TITLE,
    description=config.API_DESCRIPTION,
    version=config.API_VERSION,
)


# =============================================================================
# Esquemas de entrada y salida (Pydantic)
# =============================================================================
class CarFeatures(BaseModel):
    """Features del automóvil para predicción."""
    marca: str = Field(..., example="Toyota", description="Marca del vehículo")
    modelo: str = Field(..., example="Corolla", description="Modelo del vehículo")
    año: int = Field(..., example=2019, ge=1990, le=2026, description="Año del vehículo")
    kilometraje: int = Field(..., example=45000, ge=0, description="Kilometraje en km")
    tipo_de_combustible: str = Field(..., example="Bencina", description="Tipo de combustible")
    transmision: str = Field(..., example="Automática", description="Tipo de transmisión")
    tipo_de_carroceria: str = Field(..., example="Sedán", description="Tipo de carrocería")

    class Config:
        json_schema_extra = {
            "example": {
                "marca": "Toyota",
                "modelo": "Corolla",
                "año": 2019,
                "kilometraje": 45000,
                "tipo_de_combustible": "Bencina",
                "transmision": "Automática",
                "tipo_de_carroceria": "Sedán",
            }
        }


class PredictionResponse(BaseModel):
    """Respuesta de la predicción."""
    precio_estimado: int = Field(..., description="Precio estimado en CLP")
    rango_bajo: int = Field(..., description="Estimación conservadora (-10%)")
    rango_alto: int = Field(..., description="Estimación optimista (+10%)")
    moneda: str = "CLP"


class OpportunityRequest(BaseModel):
    """Request para evaluar si un auto es buena oportunidad."""
    marca: str = Field(..., example="Toyota")
    modelo: str = Field(..., example="Corolla")
    año: int = Field(..., example=2019)
    kilometraje: int = Field(..., example=45000)
    tipo_de_combustible: str = Field(..., example="Bencina")
    transmision: str = Field(..., example="Automática")
    tipo_de_carroceria: str = Field(..., example="Sedán")
    precio_publicado: int = Field(..., example=10500000, description="Precio al que está publicado")


class OpportunityResponse(BaseModel):
    """Respuesta del análisis de oportunidad."""
    precio_estimado: int
    precio_publicado: int
    diferencia: int
    diferencia_porcentual: float
    evaluacion: str
    moneda: str = "CLP"


# =============================================================================
# Función auxiliar de predicción
# =============================================================================
def predict_price(features: dict) -> float:
    """
    Recibe un dict con las features y retorna la predicción.
    Aplica el encoding de categóricas usando los encoders guardados.
    """
    # Crear DataFrame con una fila
    df = pd.DataFrame([features])

    # Calcular features derivadas
    df["antiguedad"] = 2026 - df["año"]
    df["km_por_ano"] = df["kilometraje"] / df["antiguedad"].clip(lower=1)
    df["log_km"] = np.log1p(df["kilometraje"])

    # Aplicar Label Encoding a las categóricas
    for col, encoder in label_encoders.items():
        if col in df.columns:
            val = str(df[col].iloc[0])
            known = set(encoder.classes_)
            if val in known:
                df[col] = encoder.transform([val])[0]
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Valor desconocido para '{col}': '{val}'. "
                           f"Valores válidos: {list(encoder.classes_[:20])}..."
                )

    # Asegurar que las columnas estén en el orden correcto
    df = df[feature_names]

    # Predecir (modelo entrenado en log-scale, reconvertir)
    prediction_log = model.predict(df)[0]
    prediction = np.expm1(prediction_log)
    return max(prediction, 0)  # No permitir precios negativos


# =============================================================================
# Endpoints
# =============================================================================
@app.get("/")
def serve_frontend():
    """Sirve el frontend del cotizador."""
    return FileResponse(os.path.join(BASE_DIR, "static", "index.html"))


@app.get("/api/marca-modelos")
def get_marca_modelos():
    """Retorna el mapeo marca → modelos para los dropdowns."""
    filepath = os.path.join(BASE_DIR, "marca_modelos.json")
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/api/defaults-modelos")
def get_defaults_modelos():
    """Retorna defaults de combustible/transmisión/carrocería por marca+modelo."""
    filepath = os.path.join(BASE_DIR, "defaults_modelos.json")
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/health")
def health_check():
    """Verifica que la API y el modelo estén funcionando."""
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "features_expected": feature_names,
        "encoders_loaded": list(label_encoders.keys()),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(car: CarFeatures):
    """
    Predice el precio de mercado de un automóvil.

    Envía las características del auto y recibe una estimación
    de precio con rango bajo y alto.
    """
    try:
        features = car.model_dump()
        precio = predict_price(features)
        precio_int = int(round(precio))

        return PredictionResponse(
            precio_estimado=precio_int,
            rango_bajo=int(round(precio * 0.90)),
            rango_alto=int(round(precio * 1.10)),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en predicción: {str(e)}")


@app.post("/opportunity", response_model=OpportunityResponse)
def evaluate_opportunity(request: OpportunityRequest):
    """
    Evalúa si un auto publicado es buena oportunidad de compra.

    Compara el precio publicado contra la estimación del modelo
    y clasifica como: BUENA OPORTUNIDAD, PRECIO JUSTO, o SOBREPRECIO.
    """
    try:
        # Extraer features para predicción (excluir precio_publicado)
        features = request.model_dump()
        precio_publicado = features.pop("precio_publicado")

        precio_estimado = predict_price(features)
        precio_est_int = int(round(precio_estimado))

        diferencia = precio_publicado - precio_est_int
        diferencia_pct = (diferencia / precio_est_int) * 100

        # Clasificar la oportunidad
        if diferencia_pct <= -15:
            evaluacion = "🔥 EXCELENTE OPORTUNIDAD - Precio muy por debajo del mercado"
        elif diferencia_pct <= -5:
            evaluacion = "✅ BUENA OPORTUNIDAD - Precio bajo el mercado"
        elif diferencia_pct <= 5:
            evaluacion = "➡️ PRECIO JUSTO - Acorde al mercado"
        elif diferencia_pct <= 15:
            evaluacion = "⚠️ LEVEMENTE SOBREPRECIADO"
        else:
            evaluacion = "❌ SOBREPRECIO - Precio muy por encima del mercado"

        return OpportunityResponse(
            precio_estimado=precio_est_int,
            precio_publicado=precio_publicado,
            diferencia=diferencia,
            diferencia_porcentual=round(diferencia_pct, 2),
            evaluacion=evaluacion,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en evaluación: {str(e)}")


# Montar archivos estáticos (debe ir AL FINAL, después de todas las rutas)
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
