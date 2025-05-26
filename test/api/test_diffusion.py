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
            # Should be a base64 string
            assert isinstance(data["image_base64"], str)
            # Should decode to bytes
            base64.b64decode(data["image_base64"])


@pytest.mark.parametrize("missing_field", ["model"])
def test_diffusion_generate_missing_fields(missing_field):
    payload = {"model": "fake-model", "prompt": "a cat"}
    del payload[missing_field]
    with TestClient(app) as client:
        resp = client.post("/diffusion/generate", json=payload)
        # Accept both 400 and 422 as valid for missing fields
        assert resp.status_code in (400, 422)


def test_is_stable_diffusion_true_tag():
    with (
        patch("transformerlab.routers.diffusion.model_info") as mock_model_info,
        patch("transformerlab.routers.diffusion.list_repo_files") as mock_list_repo_files,
    ):
        mock_info = MagicMock()
        mock_info.tags = ["stable-diffusion"]
        mock_info.architectures = []
        mock_model_info.return_value = mock_info
        mock_list_repo_files.return_value = []
        payload = {"model": "fake-model"}
        with TestClient(app) as client:
            resp = client.post("/diffusion/is_stable_diffusion", json=payload)
            assert resp.status_code == 200
            data = resp.json()
            assert data["is_stable_diffusion"] is True
            assert "SD tag" in data["reason"]


def test_is_stable_diffusion_true_architecture():
    with (
        patch("transformerlab.routers.diffusion.model_info") as mock_model_info,
        patch("transformerlab.routers.diffusion.list_repo_files") as mock_list_repo_files,
    ):
        mock_info = MagicMock()
        mock_info.tags = []
        mock_info.architectures = ["StableDiffusionPipeline"]
        mock_model_info.return_value = mock_info
        mock_list_repo_files.return_value = []
        payload = {"model": "fake-model"}
        with TestClient(app) as client:
            resp = client.post("/diffusion/is_stable_diffusion", json=payload)
            assert resp.status_code == 200
            data = resp.json()
            assert data["is_stable_diffusion"] is True
            assert "Architecture" in data["reason"]


def test_is_stable_diffusion_true_model_index():
    with (
        patch("transformerlab.routers.diffusion.model_info") as mock_model_info,
        patch("transformerlab.routers.diffusion.list_repo_files") as mock_list_repo_files,
    ):
        mock_info = MagicMock()
        mock_info.tags = []
        mock_info.architectures = []
        mock_model_info.return_value = mock_info
        mock_list_repo_files.return_value = ["foo/model_index.json"]
        payload = {"model": "fake-model"}
        with TestClient(app) as client:
            resp = client.post("/diffusion/is_stable_diffusion", json=payload)
            assert resp.status_code == 200
            data = resp.json()
            assert data["is_stable_diffusion"] is True
            assert "model_index.json" in data["reason"]


def test_is_stable_diffusion_false():
    with (
        patch("transformerlab.routers.diffusion.model_info") as mock_model_info,
        patch("transformerlab.routers.diffusion.list_repo_files") as mock_list_repo_files,
    ):
        mock_info = MagicMock()
        mock_info.tags = []
        mock_info.architectures = []
        mock_model_info.return_value = mock_info
        mock_list_repo_files.return_value = ["foo/other_file.txt"]
        payload = {"model": "fake-model"}
        with TestClient(app) as client:
            resp = client.post("/diffusion/is_stable_diffusion", json=payload)
            assert resp.status_code == 200
            data = resp.json()
            assert data["is_stable_diffusion"] is False
            assert "No SD indicators" in data["reason"]
