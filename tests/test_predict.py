import io
import os
import importlib
from fastapi.testclient import TestClient
from PIL import Image
import app.core.config
import app.services.inference
import app.main

def get_app():
    os.environ["ENABLE_DB"] = "false"
    os.environ["ENABLE_S3"] = "false"
    os.environ["ENABLE_ASYNC"] = "false"
    os.environ["MODEL_PATH"] = "/tmp/missing.pt"
    os.environ["MODEL_FORMAT"] = "pt"

    

    importlib.reload(app.core.config)
    importlib.reload(app.services.inference)
    importlib.reload(app.main)

    return app.main.app


def make_image_bytes() -> bytes:
    img = Image.new("RGB", (64, 64), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_predict():
    app = get_app()
    client = TestClient(app)
    img_bytes = make_image_bytes()
    files = {"file": ("test.png", img_bytes, "image/png")}
    resp = client.post("/predict", files=files)
    assert resp.status_code == 200
    data = resp.json()
    assert "request_id" in data
    assert "predictions" in data
    assert isinstance(data["predictions"], list)
