"""pytest fixtures for crop-disease-detection tests."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

# Ensure project root is on path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="session")
def client_unloaded():
    """TestClient with NO models loaded (inference_service = None)."""
    import api.main as main_module

    # Prevent background thread from trying to download models from GCS
    with patch.object(main_module, "_load_models_bg"):
        from api.main import app

        # Ensure inference_service is None (models still loading / not available)
        original_service = main_module.inference_service
        main_module.inference_service = None
        main_module._load_error = None

        with TestClient(app) as c:
            yield c

        # Restore
        main_module.inference_service = original_service


@pytest.fixture(scope="session")
def client_loaded():
    """TestClient with a mocked InferenceService (models "loaded")."""
    import api.main as main_module

    # Prevent background thread from trying to download models from GCS
    with patch.object(main_module, "_load_models_bg"):
        from api.main import app

        # Build a realistic mock
        mock_service = MagicMock()
        mock_service.models = {"resnet_50": {}, "mobilenet_v3": {}}
        mock_service.device = "cpu"
        mock_service.predict.return_value = {
            "prediction": "Apple_Apple_Scab",
            "confidence": 0.95,
            "top_k": [
                {"class": "Apple_Apple_Scab", "confidence": 0.95},
                {"class": "Apple_Black_Rot", "confidence": 0.03},
            ],
            "model_used": "resnet_50",
        }
        mock_service.get_gradcam.return_value = np.zeros((224, 224), dtype=np.float32)
        mock_service.get_available_models.return_value = [
            {
                "model_name": "resnet_50",
                "display_name": "ResNet-50",
                "type": "Transfer Learning",
                "params": "~25.6M",
                "speed": "medium",
                "epoch": 10,
                "val_f1_macro": 0.91,
                "val_accuracy": 0.93,
            },
            {
                "model_name": "mobilenet_v3",
                "display_name": "MobileNetV3-Large",
                "type": "Transfer Learning",
                "params": "~5.4M",
                "speed": "fast",
                "epoch": 10,
                "val_f1_macro": 0.88,
                "val_accuracy": 0.90,
            },
        ]

        original_service = main_module.inference_service
        main_module.inference_service = mock_service
        main_module._load_error = None

        with TestClient(app) as c:
            yield c

        # Restore
        main_module.inference_service = original_service
