import base64
import logging
from typing import Any

from app.core.config import settings
from app.services.inference import inference_service
from app.worker.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(name="predict_image")
def predict_image(payload: dict) -> dict:
    image_b64 = payload.get("image_b64")
    request_id = payload.get("request_id")
    if not image_b64:
        return {"request_id": request_id, "error": "missing image"}

    if inference_service.bundle is None:
        inference_service.load()

    image_bytes = base64.b64decode(image_b64)
    predictions = inference_service.predict(image_bytes)

    return {
        "request_id": request_id,
        "model_version": settings.model_version,
        "predictions": predictions,
    }
