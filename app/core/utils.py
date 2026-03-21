import uuid
from fastapi import HTTPException, UploadFile


def new_request_id() -> str:
    return str(uuid.uuid4())


def validate_upload(file: UploadFile, max_bytes: int, allowed_types: list[str]) -> None:
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=415, detail="Unsupported content type")
    size = getattr(file, "size", None)
    if size is not None and size > max_bytes:
        raise HTTPException(status_code=413, detail="File too large")


def check_bytes_limit(data: bytes, max_bytes: int) -> None:
    if len(data) > max_bytes:
        raise HTTPException(status_code=413, detail="File too large")
