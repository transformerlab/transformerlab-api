import base64
from fastapi.testclient import TestClient
import pytest
from unittest.mock import patch, MagicMock
from api import app


def test_diffusion_generate_success():
    # Patch get_pipeline to return a mock pipeline
    with patch("transformerlab.routers.diffusion.get_pipeline") as mock_get_pipeline:
        mock_pipe = MagicMock()
        # Mock the output of the pipeline call
        mock_image = MagicMock()
        mock_image.save = lambda buf, format: buf.write(b"fakepng")
        # Correctly mock the __call__ method
        mock_pipe.__call__ = MagicMock(return_value=MagicMock(images=[mock_image]))
        mock_get_pipeline.return_value = mock_pipe

        payload = {
            "model": "fake-model",
            "prompt": "a cat riding a bicycle",
            "negative_prompt": "blurry, low quality",
            "num_inference_steps": 5,
            "guidance_scale": 7.5,
            "seed": 42,
            "eta": 0.0,
            "clip_skip": 0,
            "guidance_rescale": 0.0,
        }
        with TestClient(app) as client:
            resp = client.post("/diffusion/generate", json=payload)
            assert resp.status_code == 200
            data = resp.json()
            assert data["prompt"] == payload["prompt"]
            assert data["error_code"] == 0

@pytest.mark.parametrize("missing_field", ["model"])
def test_diffusion_generate_missing_fields(missing_field):
    payload = {"model": "fake-model", "prompt": "a cat"}
    del payload[missing_field]
    with TestClient(app) as client:
        resp = client.post("/diffusion/generate", json=payload)
        # Accept both 400 and 422 as valid for missing fields
        assert resp.status_code in (400, 422)


def test_is_stable_diffusion_true_stable_diffusion_pipeline():
    """Test that StableDiffusionPipeline is correctly identified as SD model"""
    with patch("transformerlab.routers.diffusion.model_info") as mock_model_info:
        mock_info = MagicMock()
        mock_info.config = {"diffusers": {"_class_name": "StableDiffusionPipeline"}}
        mock_model_info.return_value = mock_info
        payload = {"model": "fake-model"}
        with TestClient(app) as client:
            resp = client.post("/diffusion/is_stable_diffusion", json=payload)
            assert resp.status_code == 200
            data = resp.json()
            assert data["is_stable_diffusion"] is True
            assert "Architecture matches allowed SD" in data["reason"]


def test_is_stable_diffusion_true_stable_diffusion_xl():
    """Test that StableDiffusionXLPipeline is correctly identified as SD model"""
    with patch("transformerlab.routers.diffusion.model_info") as mock_model_info:
        mock_info = MagicMock()
        mock_info.config = {"diffusers": {"_class_name": "StableDiffusionXLPipeline"}}
        mock_model_info.return_value = mock_info
        payload = {"model": "fake-model"}
        with TestClient(app) as client:
            resp = client.post("/diffusion/is_stable_diffusion", json=payload)
            assert resp.status_code == 200
            data = resp.json()
            assert data["is_stable_diffusion"] is True
            assert "Architecture matches allowed SD" in data["reason"]


def test_is_stable_diffusion_true_flux_pipeline():
    """Test that FluxPipeline is correctly identified as SD model"""
    with patch("transformerlab.routers.diffusion.model_info") as mock_model_info:
        mock_info = MagicMock()
        mock_info.config = {"diffusers": {"_class_name": "FluxPipeline"}}
        mock_model_info.return_value = mock_info
        payload = {"model": "fake-model"}
        with TestClient(app) as client:
            resp = client.post("/diffusion/is_stable_diffusion", json=payload)
            assert resp.status_code == 200
            data = resp.json()
            assert data["is_stable_diffusion"] is True
            assert "Architecture matches allowed SD" in data["reason"]


def test_is_stable_diffusion_true_list_architecture():
    """Test that a list of architectures containing SD pipeline is identified correctly"""
    with patch("transformerlab.routers.diffusion.model_info") as mock_model_info:
        mock_info = MagicMock()
        mock_info.config = {"diffusers": {"_class_name": ["SomeOtherPipeline", "StableDiffusionPipeline"]}}
        mock_model_info.return_value = mock_info
        payload = {"model": "fake-model"}
        with TestClient(app) as client:
            resp = client.post("/diffusion/is_stable_diffusion", json=payload)
            assert resp.status_code == 200
            data = resp.json()
            assert data["is_stable_diffusion"] is True
            assert "Architecture matches allowed SD" in data["reason"]


def test_is_stable_diffusion_false_no_diffusers_config():
    """Test that models without diffusers config are not identified as SD"""
    with patch("transformerlab.routers.diffusion.model_info") as mock_model_info:
        mock_info = MagicMock()
        mock_info.config = {}
        mock_model_info.return_value = mock_info
        payload = {"model": "fake-model"}
        with TestClient(app) as client:
            resp = client.post("/diffusion/is_stable_diffusion", json=payload)
            assert resp.status_code == 200
            data = resp.json()
            assert data["is_stable_diffusion"] is False
            assert "No SD indicators found" in data["reason"]


def test_is_stable_diffusion_false_unsupported_architecture():
    """Test that unsupported architectures are not identified as SD"""
    with patch("transformerlab.routers.diffusion.model_info") as mock_model_info:
        mock_info = MagicMock()
        mock_info.config = {"diffusers": {"_class_name": "SomeUnsupportedPipeline"}}
        mock_model_info.return_value = mock_info
        payload = {"model": "fake-model"}
        with TestClient(app) as client:
            resp = client.post("/diffusion/is_stable_diffusion", json=payload)
            assert resp.status_code == 200
            data = resp.json()
            assert data["is_stable_diffusion"] is False
            assert "No SD indicators found" in data["reason"]


def test_is_stable_diffusion_model_not_found():
    """Test that non-existent models return 404 error"""
    with patch("transformerlab.routers.diffusion.model_info") as mock_model_info:
        mock_model_info.side_effect = Exception("Model not found")
        payload = {"model": "non-existent-model"}
        with TestClient(app) as client:
            resp = client.post("/diffusion/is_stable_diffusion", json=payload)
            assert resp.status_code == 404
            assert "Model not found or error" in resp.json()["detail"]


def test_is_stable_diffusion_empty_class_name():
    """Test handling of empty _class_name in diffusers config"""
    with patch("transformerlab.routers.diffusion.model_info") as mock_model_info:
        mock_info = MagicMock()
        mock_info.config = {"diffusers": {"_class_name": ""}}
        mock_model_info.return_value = mock_info
        payload = {"model": "fake-model"}
        with TestClient(app) as client:
            resp = client.post("/diffusion/is_stable_diffusion", json=payload)
            assert resp.status_code == 200
            data = resp.json()
            assert data["is_stable_diffusion"] is False
            assert "No SD indicators found" in data["reason"]
