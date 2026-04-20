FROM python:3.10-slim

WORKDIR /app

# Dependencia de sistema requerida por LightGBM (libgomp)
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=8080
EXPOSE 8080

# Cloud Run inyecta $PORT dinámicamente; usar shell form para expandirlo
CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT}
