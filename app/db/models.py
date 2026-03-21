import uuid
from sqlalchemy import Column, String, Integer, DateTime, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func

from app.db.session import Base


class InferenceRequest(Base):
    __tablename__ = "inference_requests"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    request_id = Column(String, nullable=False, index=True)
    model_version = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    input_path = Column(String, nullable=True)
    content_type = Column(String, nullable=True)
    input_bytes = Column(Integer, nullable=True)
    latency_ms = Column(Integer, nullable=True)
    status = Column(String, nullable=False)
    predictions = Column(JSON, nullable=True)
