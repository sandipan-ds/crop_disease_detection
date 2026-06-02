"""Tests for FastAPI endpoints."""

import io
from pathlib import Path

import pytest
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _make_image_bytes(format: str = "JPEG", size: tuple = (256, 256)) -> tuple:
    """Create a fake image in memory for upload testing."""
    img = Image.new("RGB", size, color=(120, 180, 90))
    buf = io.BytesIO()
    img.save(buf, format=format)
    buf.seek(0)
    content_type = f"image/{format.lower()}"
    return buf.getvalue(), content_type


# ─── /health ───

class TestHealth:
    def test_health_loading(self, client_unloaded):
        """When no models are loaded, /health should report status='loading'."""
        resp = client_unloaded.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "loading"
        assert data["models_loaded"] == 0
        assert data["models"] == []

    def test_health_healthy(self, client_loaded):
        """When models are loaded, /health should report status='healthy'."""
        resp = client_loaded.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["models_loaded"] == 2
        assert "resnet_50" in data["models"]


# ─── /models ───

class TestModels:
    def test_models_when_loading(self, client_unloaded):
        """If inference_service is None, /models should return 503."""
        resp = client_unloaded.get("/models")
        assert resp.status_code == 503
        assert "loading" in resp.json()["detail"].lower()

    def test_models_list(self, client_loaded):
        """When loaded, /models should return metadata for all loaded models."""
        resp = client_loaded.get("/models")
        assert resp.status_code == 200
        data = resp.json()
        assert "models" in data
        assert len(data["models"]) == 2
        names = {m["model_name"] for m in data["models"]}
        assert "resnet_50" in names
        assert "mobilenet_v3" in names


# ─── /predict ───

class TestPredict:
    def test_predict_service_unavailable(self, client_unloaded):
        """If models are still loading, predict should return 503 (not 500 crash)."""
        image_bytes, content_type = _make_image_bytes()
        resp = client_unloaded.post(
            "/predict",
            files={"image": ("leaf.jpg", image_bytes, content_type)},
            data={"model_name": "resnet_50"},
        )
        assert resp.status_code == 503
        assert "loading" in resp.json()["detail"].lower() or "downloading" in resp.json()["detail"].lower()

    def test_predict_invalid_file_type(self, client_loaded):
        """Uploading a .txt file should return 400, not crash."""
        resp = client_loaded.post(
            "/predict",
            files={"image": ("readme.txt", b"not an image", "text/plain")},
            data={"model_name": "resnet_50"},
        )
        assert resp.status_code == 400
        assert "invalid file type" in resp.json()["detail"].lower()

    def test_predict_unknown_model(self, client_loaded):
        """Requesting a model that does not exist should return 400."""
        image_bytes, content_type = _make_image_bytes()
        resp = client_loaded.post(
            "/predict",
            files={"image": ("leaf.jpg", image_bytes, content_type)},
            data={"model_name": "nonexistent_model"},
        )
        assert resp.status_code == 400
        assert "unknown model" in resp.json()["detail"].lower()

    def test_predict_success(self, client_loaded):
        """A valid image + known model should return 200 with prediction fields."""
        image_bytes, content_type = _make_image_bytes()
        resp = client_loaded.post(
            "/predict",
            files={"image": ("leaf.jpg", image_bytes, content_type)},
            data={"model_name": "resnet_50", "top_k": "3"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "prediction" in data
        assert "confidence" in data
        assert "top_k" in data
        assert "latency_ms" in data
        assert data["model_used"] == "resnet_50"
        assert len(data["top_k"]) <= 3

    def test_predict_large_image(self, client_loaded):
        """An image > 10MB should be rejected with 400."""
        # Create a large image (~15MB uncompressed)
        img = Image.new("RGB", (5000, 5000), color=(120, 180, 90))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=95)
        image_bytes = buf.getvalue()

        resp = client_loaded.post(
            "/predict",
            files={"image": ("huge.jpg", image_bytes, "image/jpeg")},
            data={"model_name": "resnet_50"},
        )
        # Should be rejected for size
        assert resp.status_code == 400
        assert "too large" in resp.json()["detail"].lower() or "max" in resp.json()["detail"].lower()


# ─── /explain ───

class TestExplain:
    def test_explain_service_unavailable(self, client_unloaded):
        """If models are still loading, explain should return 503."""
        image_bytes, content_type = _make_image_bytes()
        resp = client_unloaded.post(
            "/explain",
            files={"image": ("leaf.jpg", image_bytes, content_type)},
            data={"model_name": "resnet_50"},
        )
        assert resp.status_code == 503

    def test_explain_success(self, client_loaded):
        """A valid request should return prediction + heatmap_base64."""
        image_bytes, content_type = _make_image_bytes()
        resp = client_loaded.post(
            "/explain",
            files={"image": ("leaf.jpg", image_bytes, content_type)},
            data={"model_name": "resnet_50"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "prediction" in data
        assert "heatmap_base64" in data
        assert data["heatmap_base64"] is not None
        assert "latency_ms" in data
