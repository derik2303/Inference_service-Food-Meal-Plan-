#!/usr/bin/env bash
set -euo pipefail

API_URL=${API_URL:-http://localhost:8000}

python - <<'PY'
from PIL import Image
import io
import sys
img = Image.new("RGB", (64, 64), color=(0, 128, 255))
buf = io.BytesIO()
img.save(buf, format="PNG")
open("/tmp/smoke.png", "wb").write(buf.getvalue())
PY

echo "Health check"
curl -sS "$API_URL/healthz" | jq .

echo "Sync predict"
curl -sS -X POST "$API_URL/predict" \
  -F "file=@/tmp/smoke.png" | jq .

if [[ "${ENABLE_ASYNC:-false}" == "true" ]]; then
  echo "Async predict"
  JOB_ID=$(curl -sS -X POST "$API_URL/predict-async" -F "file=@/tmp/smoke.png" | jq -r .job_id)
  curl -sS "$API_URL/jobs/$JOB_ID" | jq .
fi
