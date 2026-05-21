import base64
import logging
import time
from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import Response, HTMLResponse
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from celery.result import AsyncResult

from app.core.config import settings
from app.core.metrics import REQUEST_COUNT, REQUEST_LATENCY
from app.core.schemas import (
    PredictResponse,
    AsyncPredictResponse,
    JobStatusResponse,
    HealthResponse,
)
from app.core.logging import get_request_id
from app.core.utils import new_request_id, validate_upload, check_bytes_limit
from app.services.inference import inference_service
from app.storage.s3 import S3Client
from app.db.session import SessionLocal
from app.db.repository import create_inference_record
from app.worker.celery_app import celery_app

logger = logging.getLogger(__name__)

router = APIRouter()

_UI_HTML = """<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Inference UI</title>
    <style>
      :root {
        color-scheme: light;
        font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif;
      }
      body {
        margin: 24px;
        background: #f7f7f5;
        color: #1f1f1f;
      }
      .card {
        max-width: 760px;
        margin: 0 auto;
        background: #ffffff;
        border: 1px solid #e6e6e3;
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 6px 24px rgba(0, 0, 0, 0.06);
      }
      h1 { margin: 0 0 16px; font-size: 22px; }
      .camera-wrap {
        background: #111827;
        border-radius: 8px;
        overflow: hidden;
        aspect-ratio: 4 / 3;
        display: grid;
        place-items: center;
      }
      video,
      canvas {
        width: 100%;
        height: 100%;
        object-fit: cover;
      }
      canvas { display: none; }
      .controls {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-top: 12px;
      }
      button {
        background: #1f6feb;
        color: #fff;
        border: 0;
        padding: 10px 14px;
        border-radius: 6px;
        cursor: pointer;
        font-size: 14px;
      }
      button.secondary { background: #374151; }
      button:disabled {
        background: #9ca3af;
        cursor: not-allowed;
      }
      pre {
        background: #0f172a;
        color: #e2e8f0;
        padding: 12px;
        border-radius: 8px;
        overflow-x: auto;
      }
      .status { font-size: 13px; color: #555; margin-top: 8px; }
      h3 { margin-top: 22px; }
    </style>
  </head>
  <body>
    <div class="card">
      <h1>Image Inference</h1>
      <div class="camera-wrap">
        <video id="video" autoplay playsinline muted></video>
        <canvas id="canvas"></canvas>
      </div>
      <div class="controls">
        <button id="startBtn">Start camera</button>
        <button id="captureBtn" class="secondary" disabled>Capture</button>
        <button id="retakeBtn" class="secondary" disabled>Retake</button>
        <button id="predictBtn" disabled>Predict</button>
      </div>
      <div class="status" id="status"></div>
      <h3>Response</h3>
      <pre id="out">Waiting...</pre>
    </div>
    <script>
      const startBtn = document.getElementById("startBtn");
      const captureBtn = document.getElementById("captureBtn");
      const retakeBtn = document.getElementById("retakeBtn");
      const predictBtn = document.getElementById("predictBtn");
      const video = document.getElementById("video");
      const canvas = document.getElementById("canvas");
      const out = document.getElementById("out");
      const statusEl = document.getElementById("status");
      const ctx = canvas.getContext("2d");

      let stream = null;
      let capturedBlob = null;

      async function startCamera() {
        try {
          stream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: "environment" },
            audio: false
          });
          video.srcObject = stream;
          video.style.display = "block";
          canvas.style.display = "none";
          capturedBlob = null;
          captureBtn.disabled = false;
          retakeBtn.disabled = true;
          predictBtn.disabled = true;
          statusEl.textContent = "Camera ready.";
        } catch (err) {
          statusEl.textContent = "Cannot access camera. Please allow camera permission.";
          out.textContent = String(err);
        }
      }

      function captureFrame() {
        if (!video.videoWidth || !video.videoHeight) {
          statusEl.textContent = "Camera is not ready yet.";
          return;
        }

        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        canvas.toBlob((blob) => {
          if (!blob) {
            statusEl.textContent = "Could not capture image.";
            return;
          }
          capturedBlob = blob;
          video.style.display = "none";
          canvas.style.display = "block";
          captureBtn.disabled = true;
          retakeBtn.disabled = false;
          predictBtn.disabled = false;
          statusEl.textContent = "Image captured.";
        }, "image/jpeg", 0.92);
      }

      function retake() {
        capturedBlob = null;
        video.style.display = "block";
        canvas.style.display = "none";
        captureBtn.disabled = false;
        retakeBtn.disabled = true;
        predictBtn.disabled = true;
        statusEl.textContent = "Camera ready.";
      }

      async function predictCapturedImage() {
        if (!capturedBlob) {
          statusEl.textContent = "Please capture an image first.";
          return;
        }

        statusEl.textContent = "Uploading...";
        predictBtn.disabled = true;
        const form = new FormData();
        form.append("file", capturedBlob, "webcam-capture.jpg");

        try {
          const res = await fetch("/predict", { method: "POST", body: form });
          const data = await res.json();
          out.textContent = JSON.stringify(data, null, 2);
          statusEl.textContent = res.ok ? "Done." : "Error from API.";
        } catch (err) {
          statusEl.textContent = "Request failed.";
          out.textContent = String(err);
        } finally {
          predictBtn.disabled = false;
        }
      }

      startBtn.addEventListener("click", startCamera);
      captureBtn.addEventListener("click", captureFrame);
      retakeBtn.addEventListener("click", retake);
      predictBtn.addEventListener("click", predictCapturedImage);

      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        startBtn.disabled = true;
        statusEl.textContent = "Camera API is not supported in this browser.";
      } else {
        startCamera();
      }

      window.addEventListener("beforeunload", () => {
        if (!stream) return;
        stream.getTracks().forEach((track) => track.stop());
      });
    </script>
  </body>
</html>
"""


@router.get("/healthz", response_model=HealthResponse)
def healthz() -> HealthResponse:
    loaded = inference_service.is_loaded()
    return HealthResponse(status="ok", model_loaded=loaded, model_version=settings.model_version)


@router.get("/", response_class=HTMLResponse)
def ui() -> HTMLResponse:
    return HTMLResponse(_UI_HTML)


@router.get("/metrics")
def metrics() -> Response:
    if not settings.metrics_enabled:
        raise HTTPException(status_code=404, detail="Metrics disabled")
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@router.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)) -> PredictResponse:
    request_id = get_request_id()
    if request_id == "-":
        request_id = new_request_id()
    validate_upload(file, settings.max_upload_bytes, settings.allowed_content_types)

    start = time.time()
    raw = await file.read()
    check_bytes_limit(raw, settings.max_upload_bytes)

    input_path = None
    if settings.enable_s3:
        s3 = S3Client()
        input_path = s3.upload_bytes(raw, file.content_type) or None

    try:
        result = inference_service.predict(raw)
        status = "ok"
    except ValueError as exc:
        status = "error"
        logger.warning("Bad request: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc))

    latency_ms = int((time.time() - start) * 1000)

    preds = result.get("predictions", [])
    dish_name = result.get("dish_name")
    calories_kcal = result.get("calories_kcal")
    is_food = result.get("is_food")
    food_probability = result.get("food_probability")
    gate_threshold = result.get("gate_threshold")
    gate_decision = result.get("gate_decision")

    if settings.enable_db:
        db = SessionLocal()
        try:
            create_inference_record(
                db,
                request_id=request_id,
                model_version=settings.model_version,
                status=status,
                input_path=input_path,
                content_type=file.content_type,
                input_bytes=len(raw),
                latency_ms=latency_ms,
                predictions={
                    "predictions": preds,
                    "dish_name": dish_name,
                    "calories_kcal": calories_kcal,
                    "is_food": is_food,
                    "food_probability": food_probability,
                    "gate_threshold": gate_threshold,
                    "gate_decision": gate_decision,
                },
            )
        finally:
            db.close()

    REQUEST_COUNT.labels(endpoint="/predict", status=status).inc()
    REQUEST_LATENCY.labels(endpoint="/predict").observe(latency_ms / 1000.0)

    return PredictResponse(
        request_id=request_id,
        model_version=settings.model_version,
        predictions=preds,
        dish_name=dish_name,
        calories_kcal=calories_kcal,
        is_food=is_food,
        food_probability=food_probability,
        gate_threshold=gate_threshold,
        gate_decision=gate_decision,
        latency_ms=latency_ms,
    )


@router.post("/predict-async", response_model=AsyncPredictResponse)
async def predict_async(file: UploadFile = File(...)) -> AsyncPredictResponse:
    if not settings.enable_async:
        raise HTTPException(status_code=503, detail="Async mode disabled")

    request_id = get_request_id()
    if request_id == "-":
        request_id = new_request_id()
    validate_upload(file, settings.max_upload_bytes, settings.allowed_content_types)

    raw = await file.read()
    check_bytes_limit(raw, settings.max_upload_bytes)

    payload = {"request_id": request_id, "image_b64": base64.b64encode(raw).decode("utf-8")}
    job = celery_app.send_task("predict_image", args=[payload])

    REQUEST_COUNT.labels(endpoint="/predict-async", status="queued").inc()

    return AsyncPredictResponse(job_id=job.id, request_id=request_id)


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
def job_status(job_id: str) -> JobStatusResponse:
    if not settings.enable_async:
        raise HTTPException(status_code=503, detail="Async mode disabled")

    result = AsyncResult(job_id, app=celery_app)
    payload = result.result if result.successful() else None

    return JobStatusResponse(job_id=job_id, status=result.status, result=payload)
