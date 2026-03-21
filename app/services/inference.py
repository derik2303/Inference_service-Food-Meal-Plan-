import io
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from PIL import Image
import torch
from torchvision import transforms

from app.core.config import settings
from app.core.metrics import MODEL_ERRORS
from app.core.model import ModelLoader

logger = logging.getLogger(__name__)


class InferenceService:
    """Loads the model once and serves image predictions."""

    def __init__(self) -> None:
        self.model_loader = ModelLoader(
            settings.model_path,
            settings.model_format,
            settings.device,
        )
        self.device = torch.device(settings.device)
        self.bundle = None
        self.classes = self._load_classes()
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        )

    def load(self) -> None:
        """Load the model into memory and optionally warm it up."""
        self.bundle = self.model_loader.load()
        if self.bundle.loaded and settings.warmup:
            self._warmup()

    def _warmup(self) -> None:
        """Run a single forward pass to reduce cold-start latency."""
        try:
            dummy = torch.zeros((1, 3, 224, 224), dtype=torch.float32, device=self.device)
            _ = self._run_model(dummy)
        except Exception as exc:
            logger.warning("Warmup failed: %s", exc)

    def _run_model(self, tensor: torch.Tensor) -> Any:
        """Execute the underlying model and return raw outputs."""
        if not self.bundle or not self.bundle.loaded:
            raise RuntimeError("Model not loaded")

        if self.bundle.format == "pt":
            with torch.no_grad():
                output = self.bundle.model(tensor.to(self.device))
            return output

        if self.bundle.format == "onnx":
            input_name = self.bundle.model.get_inputs()[0].name
            outputs = self.bundle.model.run(None, {input_name: tensor.numpy()})
            return outputs

        raise RuntimeError("Unsupported model format")

    def _dummy_predictions(self) -> List[dict]:
        """Fallback predictions when model artifact is missing."""
        label = settings.class_labels[0] if settings.class_labels else "classA"
        return [{"label": label, "score": 0.93}]

    def _load_classes(self) -> List[str]:
        path = Path(settings.classes_path)
        if not path.exists():
            return settings.class_labels
        try:
            data = json.loads(path.read_text())
            if isinstance(data, list):
                return [str(x) for x in data]
        except Exception as exc:
            logger.warning("Failed to load classes.json: %s", exc)
        return settings.class_labels

    def _resolve_labels(self, count: int) -> List[str]:
        labels = self.classes or settings.class_labels or []
        if len(labels) < count:
            labels = labels + [f"class{i}" for i in range(len(labels), count)]
        return labels

    def predict(self, image_bytes: bytes) -> Dict[str, Any]:
        """Perform inference on raw image bytes."""
        try:
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as exc:
            MODEL_ERRORS.labels(type="image_decode").inc()
            raise ValueError(f"Failed to decode image: {exc}")

        tensor = self.transform(img).unsqueeze(0)

        if not self.bundle or not self.bundle.loaded:
            return {"predictions": self._dummy_predictions(), "dish_name": None, "calories_kcal": None}

        try:
            output = self._run_model(tensor)
        except Exception as exc:
            MODEL_ERRORS.labels(type="inference").inc()
            logger.exception("Inference failed: %s", exc)
            return {"predictions": self._dummy_predictions(), "dish_name": None, "calories_kcal": None}

        # Multitask output: (kcal_pred, dish_id) or (kcal_pred, logits)
        if isinstance(output, (list, tuple)) and len(output) >= 2:
            kcal_raw = output[0]
            dish_raw = output[1]

            if torch.is_tensor(kcal_raw):
                kcal_pred = float(kcal_raw.detach().cpu().numpy().reshape(-1)[0])
            else:
                kcal_pred = float(np.array(kcal_raw).reshape(-1)[0])

            dish_conf = None
            if torch.is_tensor(dish_raw):
                dish_raw_np = dish_raw.detach().cpu().numpy()
            else:
                dish_raw_np = np.array(dish_raw)

            if dish_raw_np.ndim == 2 and dish_raw_np.shape[1] > 1:
                probs = torch.softmax(torch.tensor(dish_raw_np), dim=1).numpy()
                dish_id = int(np.argmax(probs, axis=1)[0])
                dish_conf = float(probs[0][dish_id])
            else:
                dish_id = int(dish_raw_np.reshape(-1)[0])

            labels = self._resolve_labels(dish_id + 1)
            dish_name = labels[dish_id] if dish_id < len(labels) else f"class{dish_id}"
            score = dish_conf if dish_conf is not None else 1.0
            return {
                "predictions": [{"label": dish_name, "score": float(score)}],
                "dish_name": dish_name,
                "calories_kcal": kcal_pred,
            }

        # Single-head fallback: map scores to labels if available.
        scores = np.array(output).flatten().tolist()
        labels = self._resolve_labels(len(scores))
        top = sorted(zip(labels, scores), key=lambda x: x[1], reverse=True)[:5]
        return {
            "predictions": [{"label": label, "score": float(score)} for label, score in top],
            "dish_name": None,
            "calories_kcal": None,
        }


inference_service = InferenceService()
