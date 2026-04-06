# Predictor de Precios de Automoviles Usados

API REST para estimar el precio de mercado de automoviles usados en Chile, permitiendo detectar oportunidades de compra y venta.

## Descripcion del Problema

El mercado de autos usados presenta alta asimetria de informacion: compradores y vendedores no siempre conocen el precio justo de un vehiculo. Este modelo de Machine Learning (LightGBM) estima el precio de mercado de un automovil en base a sus caracteristicas (marca, modelo, ano, kilometraje, combustible, transmision, carroceria), permitiendo:

- **Compradores**: Detectar si un auto publicado esta a buen precio.
- **Vendedores**: Fijar un precio competitivo y justo.

## Stack Tecnologico

- **Modelo**: LightGBM (Regresion)
- **API**: FastAPI + Uvicorn
- **Deploy**: Render
- **Lenguaje**: Python 3.10

## Estructura del Proyecto

```
trabajo_final_mlops2026/
├── main.py                  # API FastAPI + endpoints
├── config.py                # Configuracion centralizada
├── preprocessing.py         # Preprocesamiento de datos
├── train.py                 # Entrenamiento y evaluacion
├── generate_defaults.py     # Genera defaults por marca+modelo
├── model/
│   ├── model.pkl            # Modelo entrenado (LightGBM)
│   ├── label_encoders.pkl   # Encoders de variables categoricas
│   └── feature_names.pkl    # Nombres de features
├── static/
│   └── index.html           # Frontend cotizador
├── marca_modelos.json       # Mapeo marca -> modelos
├── defaults_modelos.json    # Defaults combustible/transmision/carroceria
├── requirements.txt         # Dependencias
├── runtime.txt              # Version de Python
├── Procfile                 # Comando de inicio para deploy
├── Dockerfile               # Para deploy con Docker
└── README.md
```

## Instrucciones para Correr Localmente

### 1. Clonar el repositorio
```bash
git clone https://github.com/Joseisg/trabajo_final_mlops2026.git
cd trabajo_final_mlops2026
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
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

La API estara disponible en: `http://localhost:8000`

Documentacion interactiva: `http://localhost:8000/docs`

## Endpoints

### `GET /health`
Verifica que la API este funcionando.

```bash
curl http://localhost:8000/health
```

### `POST /predict`
Predice el precio estimado de un automovil.

**Request:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "marca": "Toyota",
    "modelo": "Corolla",
    "año": 2019,
    "kilometraje": 45000,
    "tipo_de_combustible": "Bencina",
    "transmision": "Automática",
    "tipo_de_carroceria": "Sedán"
  }'
```

**Response:**
```json
{
  "precio_estimado": 12390303,
  "rango_bajo": 11151273,
  "rango_alto": 13629333,
  "moneda": "CLP"
}
```

### `POST /opportunity`
Evalua si un auto publicado es buena oportunidad de compra.

**Request:**
```bash
curl -X POST http://localhost:8000/opportunity \
  -H "Content-Type: application/json" \
  -d '{
    "marca": "Toyota",
    "modelo": "Corolla",
    "año": 2019,
    "kilometraje": 45000,
    "tipo_de_combustible": "Bencina",
    "transmision": "Automática",
    "tipo_de_carroceria": "Sedán",
    "precio_publicado": 10500000
  }'
```

**Response:**
```json
{
  "precio_estimado": 12390303,
  "precio_publicado": 10500000,
  "diferencia": -1890303,
  "diferencia_porcentual": -15.26,
  "evaluacion": "EXCELENTE OPORTUNIDAD - Precio muy por debajo del mercado",
  "moneda": "CLP"
}
```

## Plataforma Cloud

**Deploy en:** Render

**URL Publica:** [Se actualizara despues del deploy]

## Metricas del Modelo

| Metrica | Valor |
|---------|-------|
| R²      | 0.8102 |
| MAE     | $2,592,938 CLP |
| RMSE    | $5,966,989 CLP |
| MAPE    | 15.55% |
| Median AE | $1,108,027 CLP |

### Justificacion del Modelo

Se eligio **LightGBM** por su excelente rendimiento en datos tabulares con variables categoricas, velocidad de entrenamiento y capacidad de manejar valores faltantes. El modelo utiliza 7 features: año, kilometraje, marca, modelo, tipo de combustible, transmision y tipo de carroceria. Se aplico early stopping (50 rondas) para evitar overfitting, con separacion train/test 80/20.

## Autores

- Jose Inostroza
- Sergio Meneses
- [Completar nombres del grupo]
