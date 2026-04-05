# 🚗 Predictor de Precios de Automóviles Usados

API REST para estimar el precio de mercado de automóviles usados, permitiendo detectar oportunidades de compra y venta.

## Descripción del Problema

El mercado de autos usados presenta alta asimetría de información: compradores y vendedores no siempre conocen el precio justo de un vehículo. Este modelo de Machine Learning (LightGBM) estima el precio de mercado de un automóvil en base a sus características (marca, modelo, año, kilometraje, etc.), permitiendo:

- **Compradores**: Detectar si un auto publicado está a buen precio.
- **Vendedores**: Fijar un precio competitivo y justo.

## Stack Tecnológico

- **Modelo**: LightGBM (Regresión)
- **API**: FastAPI + Uvicorn
- **Deploy**: [COMPLETAR con plataforma usada]
- **Lenguaje**: Python 3.10

## Estructura del Proyecto

```
car_price_predictor/
├── app/
│   └── main.py              # API FastAPI
├── model/
│   ├── model.pkl             # Modelo entrenado
│   ├── label_encoders.pkl    # Encoders de variables categóricas
│   └── feature_names.pkl     # Nombres de features
├── data/                     # Dataset (no incluido en el repo)
├── config.py                 # Configuración centralizada
├── preprocessing.py          # Preprocesamiento de datos
├── train.py                  # Entrenamiento y evaluación
├── requirements.txt          # Dependencias
├── runtime.txt               # Versión de Python
├── Procfile                  # Comando de inicio para deploy
├── Dockerfile                # (Opcional) Para deploy con Docker
└── README.md
```

## Instrucciones para Correr Localmente

### 1. Clonar el repositorio
```bash
git clone https://github.com/[USUARIO]/car-price-predictor.git
cd car-price-predictor
```

### 2. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 3. Entrenar el modelo (si no existe model.pkl)
```bash
python train.py
```

### 4. Levantar la API
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

La API estará disponible en: `http://localhost:8000`

Documentación interactiva: `http://localhost:8000/docs`

## Endpoints

### `GET /health`
Verifica que la API esté funcionando.

```bash
curl http://localhost:8000/health
```

### `POST /predict`
Predice el precio estimado de un automóvil.

**Request:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "marca": "Toyota",
    "modelo": "Corolla",
    "año": 2019,
    "kilometraje": 45000
  }'
```

**Response:**
```json
{
  "precio_estimado": 12500000,
  "rango_bajo": 11250000,
  "rango_alto": 13750000,
  "moneda": "CLP"
}
```

### `POST /opportunity`
Evalúa si un auto publicado es buena oportunidad de compra.

**Request:**
```bash
curl -X POST http://localhost:8000/opportunity \
  -H "Content-Type: application/json" \
  -d '{
    "marca": "Toyota",
    "modelo": "Corolla",
    "año": 2019,
    "kilometraje": 45000,
    "precio_publicado": 10500000
  }'
```

**Response:**
```json
{
  "precio_estimado": 12500000,
  "precio_publicado": 10500000,
  "diferencia": -2000000,
  "diferencia_porcentual": -16.0,
  "evaluacion": "🔥 EXCELENTE OPORTUNIDAD - Precio muy por debajo del mercado",
  "moneda": "CLP"
}
```

## Plataforma Cloud

**Deploy en:** [COMPLETAR - Render / Railway / Fly.io / etc.]

**URL Pública:** [COMPLETAR]

## Métricas del Modelo

| Métrica | Valor |
|---------|-------|
| R²      | [COMPLETAR] |
| MAE     | [COMPLETAR] |
| RMSE    | [COMPLETAR] |
| MAPE    | [COMPLETAR]% |

## Autores

- [COMPLETAR nombres del grupo]
