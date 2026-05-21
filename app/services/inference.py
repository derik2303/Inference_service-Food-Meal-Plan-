import io
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b0

from app.core.config import settings
from app.core.metrics import MODEL_ERRORS
from app.core.model import ModelLoader

logger = logging.getLogger(__name__)


class EfficientNetB0DishClassifier(nn.Module):
    """Classifier architecture for best_food101_efficientnet_dish.pth."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.backbone = efficientnet_b0(weights=None)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class EfficientNetB0Regressor(nn.Module):
    """Regressor architecture for best_regressor_efficientnetb0_fulltrain.pth."""

    def __init__(self, dropout: float = 0.2) -> None:
        super().__init__()
        self.backbone = efficientnet_b0(weights=None)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        self.reg_head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        return self.reg_head(feat)


class EfficientNetB0Gate(nn.Module):
    """Binary food/non-food gate architecture for best_gate_food_model.pth."""

    def __init__(self, dropout: float = 0.25) -> None:
        super().__init__()
        self.backbone = efficientnet_b0(weights=None)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        self.head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        return self.head(feat)


class InferenceService:
    """Load two checkpoints (dish classifier + calorie regressor) and serve predictions."""

    def __init__(self) -> None:
        self.model_loader = ModelLoader(
            settings.model_path,
            settings.model_format,
            settings.device,
        )
        self.device = torch.device(settings.device)

        # Legacy single-model fallback.
        self.bundle = None

        # Dual-model path (preferred).
        self.gate_model: Optional[nn.Module] = None
        self.gate_loaded: bool = False
        self.gate_threshold: float = settings.gate_threshold
        self.dish_model: Optional[nn.Module] = None
        self.reg_model: Optional[nn.Module] = None
        self.dual_loaded: bool = False

        self.classes = self._load_classes()
        self.kcal_mean: float = 0.0
        self.kcal_std: float = 1.0
        self.img_size: int = 224

        self.transform = self._build_transform(self.img_size)

    def _build_transform(self, img_size: int) -> transforms.Compose:
        return transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def load(self) -> None:
        """Load dual models first; fallback to legacy model loader."""
        self.gate_loaded = self._load_gate_model()
        self.dual_loaded = self._load_dual_models()
        if self.dual_loaded:
            if settings.warmup:
                self._warmup_dual()
            return

        logger.warning("Dual-model loading failed. Falling back to legacy single-model mode.")
        self.bundle = self.model_loader.load()
        if self.bundle.loaded and settings.warmup:
            self._warmup_legacy()

    def _gate_path_candidates(self) -> List[Path]:
        configured = Path(settings.gate_model_path)
        candidates = [configured]

        app_root = Path(__file__).resolve().parents[2]
        workspace_root = app_root.parent
        candidates.extend(
            [
                app_root / "models" / "best_gate_food_model.pth",
                workspace_root / "best_gate_food_model.pth",
                workspace_root / "models" / "best_gate_food_model.pth",
            ]
        )

        unique: List[Path] = []
        seen = set()
        for path in candidates:
            key = str(path)
            if key not in seen:
                unique.append(path)
                seen.add(key)
        return unique

    def _load_gate_model(self) -> bool:
        gate_path = next((path for path in self._gate_path_candidates() if path.exists()), None)
        if gate_path is None:
            logger.warning("Gate model not found. Checked: %s", [str(p) for p in self._gate_path_candidates()])
            self.gate_model = None
            return False

        try:
            gate_ckpt = torch.load(str(gate_path), map_location=self.device)
            if not isinstance(gate_ckpt, dict) or "model_state" not in gate_ckpt:
                raise RuntimeError("Gate checkpoint must be a dict with key 'model_state'")

            gate_cfg = gate_ckpt.get("cfg") or {}
            dropout = float(gate_cfg.get("DROPOUT", 0.25))
            self.gate_threshold = settings.gate_threshold

            self.gate_model = EfficientNetB0Gate(dropout=dropout).to(self.device)
            self.gate_model.load_state_dict(gate_ckpt["model_state"], strict=True)
            self.gate_model.eval()

            logger.info(
                "Loaded gate model successfully. path=%s, threshold=%.4f",
                gate_path,
                self.gate_threshold,
            )
            return True
        except Exception as exc:
            logger.exception("Failed to load gate model: %s", exc)
            self.gate_model = None
            return False

    def _load_dual_models(self) -> bool:
        dish_path = Path(settings.dish_model_path)
        reg_path = Path(settings.calorie_model_path)

        if not dish_path.exists() or not reg_path.exists():
            logger.warning(
                "Dual-model files missing. dish=%s exists=%s, reg=%s exists=%s",
                dish_path,
                dish_path.exists(),
                reg_path,
                reg_path.exists(),
            )
            return False

        try:
            dish_ckpt = torch.load(str(dish_path), map_location=self.device)
            if not isinstance(dish_ckpt, dict) or "model_state" not in dish_ckpt:
                raise RuntimeError("Dish checkpoint must be a dict with key 'model_state'")

            classes = dish_ckpt.get("classes")
            if isinstance(classes, list) and classes:
                self.classes = [str(x) for x in classes]
            elif not self.classes:
                raise RuntimeError("No class labels found in dish checkpoint/classes.json")

            self.dish_model = EfficientNetB0DishClassifier(num_classes=len(self.classes)).to(self.device)
            self.dish_model.load_state_dict(dish_ckpt["model_state"], strict=True)
            self.dish_model.eval()

            reg_ckpt = torch.load(str(reg_path), map_location=self.device)
            if not isinstance(reg_ckpt, dict) or "model_state" not in reg_ckpt:
                raise RuntimeError("Reg checkpoint must be a dict with key 'model_state'")

            self.kcal_mean = float(reg_ckpt.get("kcal_mean", 0.0))
            self.kcal_std = float(reg_ckpt.get("kcal_std", 1.0))
            if abs(self.kcal_std) < 1e-8:
                self.kcal_std = 1.0

            self.img_size = int(reg_ckpt.get("img_size", 224))
            self.transform = self._build_transform(self.img_size)

            self.reg_model = EfficientNetB0Regressor(dropout=0.2).to(self.device)
            self.reg_model.load_state_dict(reg_ckpt["model_state"], strict=True)
            self.reg_model.eval()

            logger.info(
                "Loaded dual models successfully. classes=%d, img_size=%d, kcal_mean=%.3f, kcal_std=%.3f",
                len(self.classes),
                self.img_size,
                self.kcal_mean,
                self.kcal_std,
            )
            return True
        except Exception as exc:
            logger.exception("Failed to load dual models: %s", exc)
            self.dish_model = None
            self.reg_model = None
            return False

    def is_loaded(self) -> bool:
        if self.dual_loaded and self.dish_model is not None and self.reg_model is not None:
            return True
        return bool(self.bundle and self.bundle.loaded)

    def _warmup_dual(self) -> None:
        try:
            dummy = torch.zeros((1, 3, self.img_size, self.img_size), dtype=torch.float32, device=self.device)
            with torch.no_grad():
                if self.gate_loaded and self.gate_model is not None:
                    _ = self.gate_model(dummy)
                _ = self.dish_model(dummy)  # type: ignore[misc]
                _ = self.reg_model(dummy)  # type: ignore[misc]
        except Exception as exc:
            logger.warning("Dual-model warmup failed: %s", exc)

    def _warmup_legacy(self) -> None:
        try:
            dummy = torch.zeros((1, 3, 224, 224), dtype=torch.float32, device=self.device)
            _ = self._run_legacy_model(dummy)
        except Exception as exc:
            logger.warning("Legacy warmup failed: %s", exc)

    def _run_legacy_model(self, tensor: torch.Tensor) -> Any:
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
        label = settings.class_labels[0] if settings.class_labels else "classA"
        return [{"label": label, "score": 0.93}]

    def _non_food_response(self, probability: float) -> Dict[str, Any]:
        return {
            "predictions": [],
            "dish_name": None,
            "calories_kcal": None,
            "is_food": False,
            "food_probability": float(probability),
            "gate_threshold": float(self.gate_threshold),
            "gate_decision": "blocked_non_food",
        }

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

    def _predict_dual(self, tensor: torch.Tensor) -> Dict[str, Any]:
        with torch.no_grad():
            dish_logits = self.dish_model(tensor.to(self.device))  # type: ignore[misc]
            reg_raw = self.reg_model(tensor.to(self.device))  # type: ignore[misc]

        probs = torch.softmax(dish_logits, dim=1).detach().cpu().numpy()[0]
        topk = min(5, probs.shape[0])
        top_idx = np.argsort(probs)[::-1][:topk]

        labels = self._resolve_labels(probs.shape[0])
        predictions = [
            {"label": labels[int(i)] if int(i) < len(labels) else f"class{int(i)}", "score": float(probs[int(i)])}
            for i in top_idx
        ]

        best_idx = int(top_idx[0]) if top_idx.size > 0 else 0
        dish_name = labels[best_idx] if best_idx < len(labels) else f"class{best_idx}"

        pred_std = float(reg_raw.detach().cpu().numpy().reshape(-1)[0])
        kcal = pred_std * self.kcal_std + self.kcal_mean
        kcal = float(np.clip(kcal, 0.0, 2000.0))

        return {
            "predictions": predictions,
            "dish_name": dish_name,
            "calories_kcal": kcal,
            "is_food": True,
            "food_probability": None,
            "gate_threshold": float(self.gate_threshold) if self.gate_loaded else None,
            "gate_decision": "passed_food_gate" if self.gate_loaded else "gate_unavailable",
        }

    def _predict_legacy(self, tensor: torch.Tensor) -> Dict[str, Any]:
        output = self._run_legacy_model(tensor)

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
                "is_food": True if self.gate_loaded else None,
                "food_probability": None,
                "gate_threshold": float(self.gate_threshold) if self.gate_loaded else None,
                "gate_decision": "passed_food_gate" if self.gate_loaded else "gate_unavailable",
            }

        scores = np.array(output).flatten().tolist()
        labels = self._resolve_labels(len(scores))
        top = sorted(zip(labels, scores), key=lambda x: x[1], reverse=True)[:5]
        return {
            "predictions": [{"label": label, "score": float(score)} for label, score in top],
            "dish_name": None,
            "calories_kcal": None,
            "is_food": True if self.gate_loaded else None,
            "food_probability": None,
            "gate_threshold": float(self.gate_threshold) if self.gate_loaded else None,
            "gate_decision": "passed_food_gate" if self.gate_loaded else "gate_unavailable",
        }

    def _predict_gate(self, tensor: torch.Tensor) -> Dict[str, Any]:
        if not self.gate_loaded or self.gate_model is None:
            return {
                "is_food": None,
                "food_probability": None,
                "gate_threshold": None,
                "gate_decision": "gate_unavailable",
            }

        with torch.no_grad():
            logits = self.gate_model(tensor.to(self.device))

        probability = float(torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)[0])
        is_food = probability >= self.gate_threshold
        return {
            "is_food": bool(is_food),
            "food_probability": probability,
            "gate_threshold": float(self.gate_threshold),
            "gate_decision": "passed_food_gate" if is_food else "blocked_non_food",
        }

    def predict(self, image_bytes: bytes) -> Dict[str, Any]:
        try:
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as exc:
            MODEL_ERRORS.labels(type="image_decode").inc()
            raise ValueError(f"Failed to decode image: {exc}")

        tensor = self.transform(img).unsqueeze(0)

        try:
            gate_result = self._predict_gate(tensor)
        except Exception as exc:
            MODEL_ERRORS.labels(type="gate_inference").inc()
            logger.exception("Gate inference failed: %s", exc)
            gate_result = {
                "is_food": None,
                "food_probability": None,
                "gate_threshold": None,
                "gate_decision": "gate_error",
            }

        if gate_result.get("is_food") is False:
            return self._non_food_response(float(gate_result["food_probability"]))

        if self.dual_loaded and self.dish_model is not None and self.reg_model is not None:
            try:
                result = self._predict_dual(tensor)
                if gate_result.get("food_probability") is not None:
                    result["food_probability"] = gate_result["food_probability"]
                if gate_result.get("gate_threshold") is not None:
                    result["gate_threshold"] = gate_result["gate_threshold"]
                if gate_result.get("gate_decision") is not None:
                    result["gate_decision"] = gate_result["gate_decision"]
                if gate_result.get("is_food") is not None:
                    result["is_food"] = gate_result["is_food"]
                return result
            except Exception as exc:
                MODEL_ERRORS.labels(type="inference").inc()
                logger.exception("Dual-model inference failed: %s", exc)
                return {
                    "predictions": self._dummy_predictions(),
                    "dish_name": None,
                    "calories_kcal": None,
                    "is_food": gate_result.get("is_food"),
                    "food_probability": gate_result.get("food_probability"),
                    "gate_threshold": gate_result.get("gate_threshold"),
                    "gate_decision": gate_result.get("gate_decision"),
                }

        if not self.bundle or not self.bundle.loaded:
            return {
                "predictions": self._dummy_predictions(),
                "dish_name": None,
                "calories_kcal": None,
                "is_food": gate_result.get("is_food"),
                "food_probability": gate_result.get("food_probability"),
                "gate_threshold": gate_result.get("gate_threshold"),
                "gate_decision": gate_result.get("gate_decision"),
            }

        try:
            result = self._predict_legacy(tensor)
            if gate_result.get("food_probability") is not None:
                result["food_probability"] = gate_result["food_probability"]
            if gate_result.get("gate_threshold") is not None:
                result["gate_threshold"] = gate_result["gate_threshold"]
            if gate_result.get("gate_decision") is not None:
                result["gate_decision"] = gate_result["gate_decision"]
            if gate_result.get("is_food") is not None:
                result["is_food"] = gate_result["is_food"]
            return result
        except Exception as exc:
            MODEL_ERRORS.labels(type="inference").inc()
            logger.exception("Legacy inference failed: %s", exc)
            return {
                "predictions": self._dummy_predictions(),
                "dish_name": None,
                "calories_kcal": None,
                "is_food": gate_result.get("is_food"),
                "food_probability": gate_result.get("food_probability"),
                "gate_threshold": gate_result.get("gate_threshold"),
                "gate_decision": gate_result.get("gate_decision"),
            }


inference_service = InferenceService()
