from fastapi.testclient import TestClient
import pytest
from unittest.mock import patch, MagicMock, mock_open
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


def test_is_valid_diffusion_model_true_stable_diffusion_pipeline():
    """Test that StableDiffusionPipeline is correctly identified as SD model"""
    with patch("transformerlab.routers.diffusion.model_info") as mock_model_info:
        mock_info = MagicMock()
        mock_info.config = {"diffusers": {"_class_name": "StableDiffusionPipeline"}}
        mock_model_info.return_value = mock_info
        payload = {"model": "fake-model"}
        with TestClient(app) as client:
            resp = client.post("/diffusion/is_valid_diffusion_model", json=payload)
            assert resp.status_code == 200
            data = resp.json()
            assert data["is_valid_diffusion_model"] is True
            assert "Architecture matches allowed SD" in data["reason"]



def test_is_valid_diffusion_model_true_stable_diffusion_xl():
    """Test that StableDiffusionXLPipeline is correctly identified as SD model"""
    with patch("transformerlab.routers.diffusion.model_info") as mock_model_info:
        mock_info = MagicMock()
        mock_info.config = {"diffusers": {"_class_name": "StableDiffusionXLPipeline"}}
        mock_model_info.return_value = mock_info
        payload = {"model": "fake-model"}
        with TestClient(app) as client:
            resp = client.post("/diffusion/is_valid_diffusion_model", json=payload)
            assert resp.status_code == 200
            data = resp.json()
            assert data["is_valid_diffusion_model"] is True
            assert "Architecture matches allowed SD" in data["reason"]


def test_is_valid_diffusion_model_true_flux_pipeline():
    """Test that FluxPipeline is correctly identified as SD model"""
    with patch("transformerlab.routers.diffusion.model_info") as mock_model_info:
        mock_info = MagicMock()
        mock_info.config = {"diffusers": {"_class_name": "FluxPipeline"}}
        mock_model_info.return_value = mock_info
        payload = {"model": "fake-model"}
        with TestClient(app) as client:
            resp = client.post("/diffusion/is_valid_diffusion_model", json=payload)
            assert resp.status_code == 200
            data = resp.json()
            assert data["is_valid_diffusion_model"] is True
            assert "Architecture matches allowed SD" in data["reason"]


def test_is_valid_diffusion_model_true_list_architecture():
    """Test that a list of architectures containing SD pipeline is identified correctly"""
    with patch("transformerlab.routers.diffusion.model_info") as mock_model_info:
        mock_info = MagicMock()
        mock_info.config = {"diffusers": {"_class_name": ["SomeOtherPipeline", "StableDiffusionPipeline"]}}
        mock_model_info.return_value = mock_info
        payload = {"model": "fake-model"}
        with TestClient(app) as client:
            resp = client.post("/diffusion/is_valid_diffusion_model", json=payload)
            assert resp.status_code == 200
            data = resp.json()
            assert data["is_valid_diffusion_model"] is True
            assert "Architecture matches allowed SD" in data["reason"]


def test_is_valid_diffusion_model_false_no_diffusers_config():
    """Test that models without diffusers config are not identified as SD"""
    with patch("transformerlab.routers.diffusion.model_info") as mock_model_info:
        mock_info = MagicMock()
        mock_info.config = {}
        mock_model_info.return_value = mock_info
        payload = {"model": "fake-model"}
        with TestClient(app) as client:
            resp = client.post("/diffusion/is_valid_diffusion_model", json=payload)
            assert resp.status_code == 200
            data = resp.json()
            assert data["is_valid_diffusion_model"] is False
            assert "No SD indicators found" in data["reason"]


def test_is_valid_diffusion_model_false_unsupported_architecture():
    """Test that unsupported architectures are not identified as SD"""
    with patch("transformerlab.routers.diffusion.model_info") as mock_model_info:
        mock_info = MagicMock()
        mock_info.config = {"diffusers": {"_class_name": "SomeUnsupportedPipeline"}}
        mock_model_info.return_value = mock_info
        payload = {"model": "fake-model"}
        with TestClient(app) as client:
            resp = client.post("/diffusion/is_valid_diffusion_model", json=payload)
            assert resp.status_code == 200
            data = resp.json()
            assert data["is_valid_diffusion_model"] is False
            assert "No SD indicators found" in data["reason"]


def test_is_valid_diffusion_model_model_not_found():
    """Test that non-existent models return 404 error"""
    with patch("transformerlab.routers.diffusion.model_info") as mock_model_info:
        mock_model_info.side_effect = Exception("Model not found")
        payload = {"model": "non-existent-model"}
        with TestClient(app) as client:
            resp = client.post("/diffusion/is_valid_diffusion_model", json=payload)
            assert resp.status_code == 404
            assert "Model not found or error" in resp.json()["detail"]


def test_is_valid_diffusion_model_empty_class_name():
    """Test handling of empty _class_name in diffusers config"""
    with patch("transformerlab.routers.diffusion.model_info") as mock_model_info:
        mock_info = MagicMock()
        mock_info.config = {"diffusers": {"_class_name": ""}}
        mock_model_info.return_value = mock_info
        payload = {"model": "fake-model"}
        with TestClient(app) as client:
            resp = client.post("/diffusion/is_valid_diffusion_model", json=payload)
            assert resp.status_code == 200
            data = resp.json()
            assert data["is_valid_diffusion_model"] is False
            assert "No SD indicators found" in data["reason"]


def test_get_history_success():
    """Test getting diffusion history with default parameters"""
    with patch("transformerlab.routers.diffusion.load_history") as mock_load_history:
        mock_history = MagicMock()
        mock_history.images = []
        mock_history.total = 0
        mock_load_history.return_value = mock_history

        with TestClient(app) as client:
            resp = client.get("/diffusion/history")
            assert resp.status_code == 200
            mock_load_history.assert_called_once_with(limit=50, offset=0)


def test_get_history_with_pagination():
    """Test getting diffusion history with pagination parameters"""
    with patch("transformerlab.routers.diffusion.load_history") as mock_load_history:
        mock_history = MagicMock()
        mock_history.images = []
        mock_history.total = 0
        mock_load_history.return_value = mock_history

        with TestClient(app) as client:
            resp = client.get("/diffusion/history?limit=25&offset=10")
            assert resp.status_code == 200
            mock_load_history.assert_called_once_with(limit=25, offset=10)


def test_get_history_invalid_limit():
    """Test getting history with invalid limit parameter"""
    with TestClient(app) as client:
        resp = client.get("/diffusion/history?limit=0")
        assert resp.status_code == 400
        assert "Limit must be greater than 1" in resp.json()["detail"]


def test_get_history_invalid_offset():
    """Test getting history with invalid offset parameter"""
    with TestClient(app) as client:
        resp = client.get("/diffusion/history?offset=-5")
        assert resp.status_code == 400
        assert "Offset must be non-negative" in resp.json()["detail"]


def test_get_image_by_id_not_found():
    """Test getting a non-existent image by ID"""
    with patch("transformerlab.routers.diffusion.find_image_by_id") as mock_find_image:
        mock_find_image.return_value = None

        with TestClient(app) as client:
            resp = client.get("/diffusion/history/non-existent-id")
            assert resp.status_code == 404
            assert "Image with ID non-existent-id not found" in resp.json()["detail"]


def test_get_image_by_id_index_out_of_range():
    """Test getting image with index out of range"""
    with (
        patch("transformerlab.routers.diffusion.find_image_by_id") as mock_find_image,
        patch("os.path.exists", return_value=True),
        patch("os.path.isdir", return_value=True),
    ):
        # Create mock image with folder-format
        mock_image = MagicMock()
        mock_image.id = "test-folder-id"
        mock_image.image_path = "/fake/path/folder"
        mock_image.num_images = 2

        mock_find_image.return_value = mock_image

        with TestClient(app) as client:
            resp = client.get("/diffusion/history/test-folder-id?index=5")
            assert resp.status_code == 404
            assert "Image index 5 out of range" in resp.json()["detail"]


def test_get_image_info_by_id_success():
    """Test getting image metadata by ID"""
    with (
        patch("transformerlab.routers.diffusion.find_image_by_id") as mock_find_image,
        patch("os.path.exists", return_value=True),
        patch("os.path.isdir", return_value=True),
        patch("os.listdir", return_value=["0.png", "1.png", "2.png"]),
    ):
        # Create mock image
        mock_image = MagicMock()
        mock_image.id = "test-image-id"
        mock_image.image_path = "/fake/path/folder"
        mock_image.model_dump = MagicMock(return_value={"id": "test-image-id", "prompt": "test prompt"})

        mock_find_image.return_value = mock_image

        with TestClient(app) as client:
            resp = client.get("/diffusion/history/test-image-id/info")
            assert resp.status_code == 200
            data = resp.json()
            assert data["id"] == "test-image-id"
            assert data["metadata"]["num_images"] == 3


def test_get_image_count_success():
    """Test getting image count for an image set"""
    with (
        patch("transformerlab.routers.diffusion.find_image_by_id") as mock_find_image,
        patch("os.path.exists", return_value=True),
        patch("os.path.isdir", return_value=True),
        patch("os.listdir", return_value=["0.png", "1.png"]),
    ):
        # Create mock image
        mock_image = MagicMock()
        mock_image.id = "test-image-id"
        mock_image.image_path = "/fake/path/folder"

        mock_find_image.return_value = mock_image

        with TestClient(app) as client:
            resp = client.get("/diffusion/history/test-image-id/count")
            assert resp.status_code == 200
            data = resp.json()
            assert data["id"] == "test-image-id"
            assert data["num_images"] == 2


def test_delete_image_from_history_not_found():
    """Test deleting a non-existent image from history"""
    with (
        patch("transformerlab.routers.diffusion.get_history_file_path", return_value="/fake/history.json"),
        patch("os.path.exists", return_value=True),
        patch("builtins.open", mock_open(read_data='[{"id": "other-id", "image_path": "/fake/path.png"}]')),
    ):
        with TestClient(app) as client:
            resp = client.delete("/diffusion/history/non-existent-id")
            assert resp.status_code == 500
            assert "Image with ID non-existent-id not found" in resp.json()["detail"]


def test_create_dataset_from_history_success():
    """Test creating a dataset from history images"""
    with (
        patch("transformerlab.routers.diffusion.find_image_by_id") as mock_find_image,
        patch("transformerlab.db.get_dataset", return_value=None),
        patch("transformerlab.db.create_local_dataset") as mock_create_dataset,
        patch("transformerlab.shared.dirs.dataset_dir_by_id", return_value="/fake/dataset"),
        patch("os.makedirs"),
        patch("os.path.exists", return_value=True),
        patch("os.path.isdir", return_value=True),
        patch("os.listdir", return_value=["0.png", "1.png"]),
        patch("shutil.copy2"),
        patch("builtins.open", mock_open()),
    ):
        # Create mock image
        mock_image = MagicMock()
        mock_image.id = "test-image-id"
        mock_image.image_path = "/fake/path/folder"
        mock_image.prompt = "test prompt"
        mock_image.negative_prompt = "test negative"
        mock_image.model = "test-model"
        mock_image.adaptor = ""
        mock_image.adaptor_scale = 1.0
        mock_image.num_inference_steps = 20
        mock_image.guidance_scale = 7.5
        mock_image.seed = 42
        mock_image.upscaled = False
        mock_image.upscale_factor = 1
        mock_image.eta = 0.0
        mock_image.clip_skip = 0
        mock_image.guidance_rescale = 0.0
        mock_image.height = 512
        mock_image.width = 512
        mock_image.timestamp = "2023-01-01T00:00:00"

        mock_find_image.return_value = mock_image

        payload = {
            "dataset_name": "test-dataset",
            "image_ids": ["test-image-id"],
            "description": "Test dataset",
            "include_metadata": True,
        }

        with TestClient(app) as client:
            resp = client.post("/diffusion/dataset/create", json=payload)
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "success"
            assert "test-dataset" in data["message"]
            mock_create_dataset.assert_called_once()


def test_create_dataset_invalid_image_ids():
    """Test creating dataset with invalid image IDs"""
    payload = {
        "dataset_name": "test-dataset",
        "image_ids": [],
        "description": "Test dataset",
        "include_metadata": False,
    }

    with TestClient(app) as client:
        resp = client.post("/diffusion/dataset/create", json=payload)
        assert resp.status_code == 400
        assert "Invalid image IDs list" in resp.json()["detail"]


def test_create_dataset_existing_dataset():
    """Test creating dataset with name that already exists"""
    with patch("transformerlab.db.get_dataset", return_value={"id": "existing"}):
        payload = {
            "dataset_name": "existing-dataset",
            "image_ids": ["test-id"],
            "description": "Test dataset",
            "include_metadata": False,
        }

        with TestClient(app) as client:
            resp = client.post("/diffusion/dataset/create", json=payload)
            assert resp.status_code == 400
            assert "already exists" in resp.json()["detail"]


def test_create_dataset_no_images_found():
    """Test creating dataset when no images are found for given IDs"""
    with (
        patch("transformerlab.routers.diffusion.find_image_by_id") as mock_find_image,
        patch("transformerlab.db.get_dataset", return_value=None),
    ):
        mock_find_image.return_value = None

        payload = {
            "dataset_name": "test-dataset",
            "image_ids": ["non-existent-id"],
            "description": "Test dataset",
            "include_metadata": False,
        }

        with TestClient(app) as client:
            resp = client.post("/diffusion/dataset/create", json=payload)
            assert resp.status_code == 404
            assert "No images found for the given IDs" in resp.json()["detail"]


@pytest.mark.parametrize("img2img_flag", [True, False])
def test_is_valid_diffusion_model_img2img_detection(img2img_flag):
    """Test that is_valid_diffusion_model correctly handles img2img flag"""
    with patch("transformerlab.routers.diffusion.model_info") as mock_model_info:
        mock_info = MagicMock()
        # Use an architecture that's in both lists for testing
        mock_info.config = {"diffusers": {"_class_name": "StableDiffusionImg2ImgPipeline"}}
        mock_model_info.return_value = mock_info

        payload = {"model": "fake-model", "is_img2img": img2img_flag}
        with TestClient(app) as client:
            resp = client.post("/diffusion/is_valid_diffusion_model", json=payload)
            assert resp.status_code == 200
            data = resp.json()
            assert data["is_valid_diffusion_model"] is True
            if img2img_flag:
                assert "img2img" in data["reason"]
            else:
                assert "Architecture matches allowed SD" in data["reason"]
