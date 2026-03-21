import os
import importlib
from fastapi.testclient import TestClient
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


def test_healthz():
    app = get_app()
    client = TestClient(app)
    resp = client.get("/healthz")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "model_version" in data
