import logging
import uuid
from typing import Optional

import boto3

from app.core.config import settings

logger = logging.getLogger(__name__)


class S3Client:
    """Thin S3/MinIO wrapper for optional image uploads."""

    def __init__(self) -> None:
        self.client = boto3.client(
            "s3",
            endpoint_url=settings.s3_endpoint_url,
            aws_access_key_id=settings.s3_access_key,
            aws_secret_access_key=settings.s3_secret_key,
            region_name=settings.s3_region,
        )

    def upload_bytes(self, data: bytes, content_type: str) -> Optional[str]:
        """Upload bytes and return the object key on success."""
        key = f"uploads/{uuid.uuid4()}.bin"
        try:
            self.client.put_object(
                Bucket=settings.s3_bucket,
                Key=key,
                Body=data,
                ContentType=content_type,
            )
            return key
        except Exception as exc:
            logger.exception("Failed to upload to S3: %s", exc)
            return None
