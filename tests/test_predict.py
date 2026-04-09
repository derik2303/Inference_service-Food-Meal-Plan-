import io
import os
import importlib
from fastapi.testclient import TestClient
from PIL import Image
import app.core.config
import app.api.routes
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
    importlib.reload(app.api.routes)
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
    assert "is_food" in data
    assert "food_probability" in data
    assert "gate_threshold" in data
    assert "gate_decision" in data


def test_predict_blocks_non_food():
    fastapi_app = get_app()
    client = TestClient(fastapi_app)

    original_predict_gate = app.api.routes.inference_service._predict_gate
    try:
        app.api.routes.inference_service._predict_gate = lambda tensor: {
            "is_food": False,
            "food_probability": 0.03,
            "gate_threshold": 0.5,
            "gate_decision": "blocked_non_food",
        }

        img_bytes = make_image_bytes()
        files = {"file": ("test.png", img_bytes, "image/png")}
        resp = client.post("/predict", files=files)
        assert resp.status_code == 200

        data = resp.json()
        assert data["predictions"] == []
        assert data["dish_name"] is None
        assert data["calories_kcal"] is None
        assert data["is_food"] is False
        assert data["gate_decision"] == "blocked_non_food"
    finally:
        app.api.routes.inference_service._predict_gate = original_predict_gate
