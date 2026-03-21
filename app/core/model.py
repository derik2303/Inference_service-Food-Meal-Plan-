import logging
from dataclasses import dataclass
from typing import Any, Optional

import torch

try:
    import onnxruntime as ort
except Exception:  # pragma: no cover - optional
    ort = None

logger = logging.getLogger(__name__)


@dataclass
class ModelBundle:
    model: Any
    format: str
    device: str
    loaded: bool


class ModelLoader:
    """Load a PyTorch TorchScript or ONNX model based on configuration."""

    def __init__(self, model_path: str, model_format: str, device: str) -> None:
        self.model_path = model_path
        self.model_format = model_format
        self.device = device
        self._bundle: Optional[ModelBundle] = None

    @property
    def bundle(self) -> Optional[ModelBundle]:
        return self._bundle

    def load(self) -> ModelBundle:
        """Load the model artifact and return a bundle with status."""
        if self.model_format == "onnx":
            if ort is None:
                logger.warning("onnxruntime is not installed; using dummy model")
                self._bundle = ModelBundle(model=None, format="onnx", device=self.device, loaded=False)
                return self._bundle
            try:
                session = ort.InferenceSession(self.model_path, providers=["CPUExecutionProvider"])
                self._bundle = ModelBundle(model=session, format="onnx", device=self.device, loaded=True)
                return self._bundle
            except Exception as exc:
                logger.exception("Failed to load ONNX model: %s", exc)
                self._bundle = ModelBundle(model=None, format="onnx", device=self.device, loaded=False)
                return self._bundle

        if self.model_format == "pt":
            try:
                model = torch.jit.load(self.model_path, map_location=self.device)
                model.eval()
                self._bundle = ModelBundle(model=model, format="pt", device=self.device, loaded=True)
                return self._bundle
            except Exception as exc:
                logger.exception("Failed to load PyTorch model: %s", exc)
                self._bundle = ModelBundle(model=None, format="pt", device=self.device, loaded=False)
                return self._bundle

        logger.warning("Unsupported model format: %s", self.model_format)
        self._bundle = ModelBundle(model=None, format=self.model_format, device=self.device, loaded=False)
        return self._bundle
