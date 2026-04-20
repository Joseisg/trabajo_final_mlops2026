#!/usr/bin/env bash
set -euo pipefail

# Despliegue directo a Cloud Run desde el código local.
# Evita Developer Connect / GitHub trigger (no requiere gitRepositoryLink).

PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project 2>/dev/null)}"
REGION="${REGION:-southamerica-west1}"
SERVICE="${SERVICE:-trabajo-final-mlops2026-git}"

if [[ -z "${PROJECT_ID}" || "${PROJECT_ID}" == "(unset)" ]]; then
  echo "ERROR: define PROJECT_ID o ejecuta 'gcloud config set project <TU_PROJECT_ID>'" >&2
  exit 1
fi

echo "==> Proyecto : ${PROJECT_ID}"
echo "==> Región   : ${REGION}"
echo "==> Servicio : ${SERVICE}"

gcloud run deploy "${SERVICE}" \
  --source . \
  --project "${PROJECT_ID}" \
  --region "${REGION}" \
  --platform managed \
  --allow-unauthenticated \
  --port 8080 \
  --memory 1Gi \
  --cpu 1 \
  --timeout 300

echo "==> Deploy OK"
gcloud run services describe "${SERVICE}" \
  --project "${PROJECT_ID}" \
  --region "${REGION}" \
  --format='value(status.url)'
