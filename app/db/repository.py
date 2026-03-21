from typing import Optional

from sqlalchemy.orm import Session

from app.db.models import InferenceRequest


def create_inference_record(
    db: Session,
    request_id: str,
    model_version: str,
    status: str,
    input_path: Optional[str],
    content_type: Optional[str],
    input_bytes: Optional[int],
    latency_ms: Optional[int],
    predictions: Optional[dict],
) -> InferenceRequest:
    """Persist a single inference request record."""
    record = InferenceRequest(
        request_id=request_id,
        model_version=model_version,
        status=status,
        input_path=input_path,
        content_type=content_type,
        input_bytes=input_bytes,
        latency_ms=latency_ms,
        predictions=predictions,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record
