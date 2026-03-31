"""ai-predicter/main.py uchun unit testlar."""

import io
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from main import app, CLASS_NAMES, ATTENTIVE_CLASSES, DISTRACTED_CLASSES


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def sample_image_bytes():
    """Test uchun 100x100 qizil rasm."""
    img = Image.new("RGB", (100, 100), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


class TestHealth:
    def test_health_returns_ok(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "model_loaded" in data


class TestPredict:
    def test_predict_without_model(self, client, sample_image_bytes):
        """Model yuklanmagan bo'lsa error qaytarishi kerak."""
        with patch("main.model", None):
            response = client.post(
                "/predict",
                files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
            )
            assert response.status_code == 200
            assert response.json()["error"] == "Model not loaded"

    def test_predict_with_mock_model(self, client, sample_image_bytes):
        """Mock model bilan predict ishlashi kerak."""
        mock_box = MagicMock()
        mock_box.cls = [MagicMock(__getitem__=lambda s, i: 0)]
        mock_box.cls[0] = 0
        mock_box.conf = [MagicMock(__getitem__=lambda s, i: 0.95)]
        mock_box.conf[0] = 0.95
        mock_box.xyxy = [MagicMock(tolist=lambda: [10.0, 20.0, 50.0, 60.0])]

        mock_result = MagicMock()
        mock_result.boxes = [mock_box]

        mock_model = MagicMock()
        mock_model.predict.return_value = [mock_result]

        with patch("main.model", mock_model):
            response = client.post(
                "/predict",
                files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
            )
            assert response.status_code == 200
            data = response.json()
            assert "detections" in data
            assert "summary" in data
            assert len(data["detections"]) == 1
            det = data["detections"][0]
            assert det["class_id"] == 0
            assert det["class_name"] == "hand-raising"
            assert det["group"] == "attentive"

    def test_predict_summary_calculation(self, client, sample_image_bytes):
        """Summary foizlari to'g'ri hisoblanishi kerak."""
        boxes = []
        for cls_id in [0, 1, 3]:  # 2 attentive, 1 distracted
            box = MagicMock()
            box.cls = [cls_id]
            box.conf = [0.9]
            box.xyxy = [MagicMock(tolist=lambda: [0, 0, 10, 10])]
            boxes.append(box)

        mock_result = MagicMock()
        mock_result.boxes = boxes
        mock_model = MagicMock()
        mock_model.predict.return_value = [mock_result]

        with patch("main.model", mock_model):
            response = client.post(
                "/predict",
                files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
            )
            data = response.json()
            assert data["summary"]["total"] == 3
            assert data["summary"]["attentive"] == 2
            assert data["summary"]["distracted"] == 1
            assert data["summary"]["attentive_percent"] == 66.7
            assert data["summary"]["distracted_percent"] == 33.3

    def test_predict_with_confidence_param(self, client, sample_image_bytes):
        """Confidence parametri AI modelga uzatilishi kerak."""
        mock_result = MagicMock()
        mock_result.boxes = []
        mock_model = MagicMock()
        mock_model.predict.return_value = [mock_result]

        with patch("main.model", mock_model):
            client.post(
                "/predict?confidence=0.8",
                files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
            )
            mock_model.predict.assert_called_once()
            call_kwargs = mock_model.predict.call_args
            assert call_kwargs.kwargs["conf"] == 0.8

    def test_predict_empty_detections(self, client, sample_image_bytes):
        """Hech narsa topilmasa bo'sh natija qaytarishi kerak."""
        mock_result = MagicMock()
        mock_result.boxes = []
        mock_model = MagicMock()
        mock_model.predict.return_value = [mock_result]

        with patch("main.model", mock_model):
            response = client.post(
                "/predict",
                files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
            )
            data = response.json()
            assert data["detections"] == []
            assert data["summary"]["total"] == 0
            assert data["summary"]["attentive_percent"] == 0


class TestConstants:
    def test_class_names_complete(self):
        assert len(CLASS_NAMES) == 6
        assert CLASS_NAMES[0] == "hand-raising"
        assert CLASS_NAMES[5] == "TurnHead"

    def test_attentive_classes(self):
        assert ATTENTIVE_CLASSES == {0, 1, 2}

    def test_distracted_classes(self):
        assert DISTRACTED_CLASSES == {3, 4, 5}

    def test_no_class_overlap(self):
        assert ATTENTIVE_CLASSES.isdisjoint(DISTRACTED_CLASSES)

    def test_all_classes_covered(self):
        assert ATTENTIVE_CLASSES | DISTRACTED_CLASSES == set(CLASS_NAMES.keys())
