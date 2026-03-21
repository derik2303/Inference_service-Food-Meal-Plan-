import os
from dataclasses import dataclass
from typing import List


def _get_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y"}


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _get_list(name: str, default: List[str]) -> List[str]:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return [item.strip() for item in raw.split(",") if item.strip()]


@dataclass(frozen=True)
class Settings:
    app_name: str
    env: str
    log_level: str
    host: str
    port: int
    workers: int
    request_timeout_s: int
    request_id_header: str
    max_upload_bytes: int
    allowed_content_types: List[str]

    model_path: str
    model_format: str
    model_version: str
    device: str
    class_labels: List[str]
    classes_path: str
    warmup: bool

    enable_db: bool
    database_url: str

    enable_async: bool
    redis_url: str

    enable_s3: bool
    s3_endpoint_url: str
    s3_access_key: str
    s3_secret_key: str
    s3_bucket: str
    s3_region: str

    metrics_enabled: bool

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            app_name=os.getenv("APP_NAME", "food-inference"),
            env=os.getenv("ENV", "local"),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            host=os.getenv("HOST", "0.0.0.0"),
            port=_get_int("PORT", 8000),
            workers=_get_int("WORKERS", 1),
            request_timeout_s=_get_int("REQUEST_TIMEOUT_S", 30),
            request_id_header=os.getenv("REQUEST_ID_HEADER", "X-Request-Id"),
            max_upload_bytes=_get_int("MAX_UPLOAD_BYTES", 5 * 1024 * 1024),
            allowed_content_types=_get_list("ALLOWED_CONTENT_TYPES", ["image/jpeg", "image/png"]),
            model_path=os.getenv("MODEL_PATH", "/models/model.pt"),
            model_format=os.getenv("MODEL_FORMAT", "pt").lower(),
            model_version=os.getenv("MODEL_VERSION", "v1"),
            device=os.getenv("DEVICE", "cpu"),
            class_labels=_get_list("CLASS_LABELS", ["classA"]),
            classes_path=os.getenv("CLASSES_PATH", "/models/classes.json"),
            warmup=_get_bool("WARMUP", True),
            enable_db=_get_bool("ENABLE_DB", False),
            database_url=os.getenv(
                "DATABASE_URL",
                "postgresql+psycopg2://postgres:postgres@localhost:5432/inference",
            ),
            enable_async=_get_bool("ENABLE_ASYNC", False),
            redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
            enable_s3=_get_bool("ENABLE_S3", False),
            s3_endpoint_url=os.getenv("S3_ENDPOINT_URL", "http://localhost:9000"),
            s3_access_key=os.getenv("S3_ACCESS_KEY", "minio"),
            s3_secret_key=os.getenv("S3_SECRET_KEY", "minio123"),
            s3_bucket=os.getenv("S3_BUCKET", "uploads"),
            s3_region=os.getenv("S3_REGION", "us-east-1"),
            metrics_enabled=_get_bool("METRICS_ENABLED", True),
        )


settings = Settings.from_env()
