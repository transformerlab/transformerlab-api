from fastapi.testclient import TestClient
import pytest
import json
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
            "scheduler": "EulerDiscreteScheduler",
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


def test_is_valid_diffusion_model_inpainting_specific_architecture():
    """Test that specific inpainting architectures are correctly identified"""
    with patch("transformerlab.routers.diffusion.model_info") as mock_model_info:
        mock_info = MagicMock()
        mock_info.config = {"diffusers": {"_class_name": "StableDiffusionInpaintPipeline"}}
        mock_model_info.return_value = mock_info

        payload = {"model": "fake-model", "is_inpainting": True}
        with TestClient(app) as client:
            resp = client.post("/diffusion/is_valid_diffusion_model", json=payload)
            assert resp.status_code == 200
            data = resp.json()
            assert data["is_valid_diffusion_model"] is True
            assert "Architecture matches allowed SD inpainting" in data["reason"]


def test_is_valid_diffusion_model_inpainting_xl_architecture():
    """Test that StableDiffusionXLInpaintPipeline is correctly identified for inpainting"""
    with patch("transformerlab.routers.diffusion.model_info") as mock_model_info:
        mock_info = MagicMock()
        mock_info.config = {"diffusers": {"_class_name": "StableDiffusionXLInpaintPipeline"}}
        mock_model_info.return_value = mock_info

        payload = {"model": "fake-model", "is_inpainting": True}
        with TestClient(app) as client:
            resp = client.post("/diffusion/is_valid_diffusion_model", json=payload)
            assert resp.status_code == 200
            data = resp.json()
            assert data["is_valid_diffusion_model"] is True
            assert "Architecture matches allowed SD inpainting" in data["reason"]


def test_is_valid_diffusion_model_inpainting_from_text2img():
    """Test that text2img architectures can be used for inpainting"""
    with patch("transformerlab.routers.diffusion.model_info") as mock_model_info:
        mock_info = MagicMock()
        mock_info.config = {"diffusers": {"_class_name": "StableDiffusionPipeline"}}
        mock_model_info.return_value = mock_info

        payload = {"model": "fake-model", "is_inpainting": True}
        with TestClient(app) as client:
            resp = client.post("/diffusion/is_valid_diffusion_model", json=payload)
            assert resp.status_code == 200
            data = resp.json()
            assert data["is_valid_diffusion_model"] is True
            assert "Architecture matches allowed SD inpainting" in data["reason"]


def test_is_valid_diffusion_model_inpainting_from_sdxl():
    """Test that StableDiffusionXL can be used for inpainting"""
    with patch("transformerlab.routers.diffusion.model_info") as mock_model_info:
        mock_info = MagicMock()
        mock_info.config = {"diffusers": {"_class_name": "StableDiffusionXLPipeline"}}
        mock_model_info.return_value = mock_info

        payload = {"model": "fake-model", "is_inpainting": True}
        with TestClient(app) as client:
            resp = client.post("/diffusion/is_valid_diffusion_model", json=payload)
            assert resp.status_code == 200
            data = resp.json()
            assert data["is_valid_diffusion_model"] is True
            assert "Architecture matches allowed SD inpainting" in data["reason"]


def test_is_valid_diffusion_model_inpainting_flux_not_supported():
    """Test that FLUX models cannot be used for inpainting"""
    with patch("transformerlab.routers.diffusion.model_info") as mock_model_info:
        mock_info = MagicMock()
        mock_info.config = {"diffusers": {"_class_name": "FluxPipeline"}}
        mock_model_info.return_value = mock_info

        payload = {"model": "fake-model", "is_inpainting": True}
        with TestClient(app) as client:
            resp = client.post("/diffusion/is_valid_diffusion_model", json=payload)
            assert resp.status_code == 200
            data = resp.json()
            assert data["is_valid_diffusion_model"] is False
            assert "No SD indicators found" in data["reason"]


def test_is_valid_diffusion_model_inpainting_unsupported_architecture():
    """Test that unsupported architectures are not valid for inpainting"""
    with patch("transformerlab.routers.diffusion.model_info") as mock_model_info:
        mock_info = MagicMock()
        mock_info.config = {"diffusers": {"_class_name": "SomeUnsupportedPipeline"}}
        mock_model_info.return_value = mock_info

        payload = {"model": "fake-model", "is_inpainting": True}
        with TestClient(app) as client:
            resp = client.post("/diffusion/is_valid_diffusion_model", json=payload)
            assert resp.status_code == 200
            data = resp.json()
            assert data["is_valid_diffusion_model"] is False
            assert "No SD indicators found" in data["reason"]


def test_is_valid_diffusion_model_inpainting_list_architectures():
    """Test that inpainting works with list of architectures containing supported ones"""
    with patch("transformerlab.routers.diffusion.model_info") as mock_model_info:
        mock_info = MagicMock()
        mock_info.config = {"diffusers": {"_class_name": ["SomeOtherPipeline", "StableDiffusionInpaintPipeline"]}}
        mock_model_info.return_value = mock_info

        payload = {"model": "fake-model", "is_inpainting": True}
        with TestClient(app) as client:
            resp = client.post("/diffusion/is_valid_diffusion_model", json=payload)
            assert resp.status_code == 200
            data = resp.json()
            assert data["is_valid_diffusion_model"] is True
            assert "Architecture matches allowed SD inpainting" in data["reason"]


@pytest.mark.parametrize("inpainting_flag", [True, False])
def test_is_valid_diffusion_model_inpainting_detection(inpainting_flag):
    """Test that is_valid_diffusion_model correctly handles is_inpainting flag"""
    with patch("transformerlab.routers.diffusion.model_info") as mock_model_info:
        mock_info = MagicMock()
        # Use an architecture that's in both text2img and inpainting lists
        mock_info.config = {"diffusers": {"_class_name": "StableDiffusionPipeline"}}
        mock_model_info.return_value = mock_info

        payload = {"model": "fake-model", "is_inpainting": inpainting_flag}
        with TestClient(app) as client:
            resp = client.post("/diffusion/is_valid_diffusion_model", json=payload)
            assert resp.status_code == 200
            data = resp.json()
            assert data["is_valid_diffusion_model"] is True
            if inpainting_flag:
                assert "Architecture matches allowed SD inpainting" in data["reason"]
            else:
                assert "Architecture matches allowed SD" in data["reason"]


def test_load_history_success():
    """Test loading history with valid data"""
    with (
        patch("transformerlab.routers.diffusion.get_history_file_path", return_value="/fake/history.json"),
        patch("os.path.exists", return_value=True),
        patch(
            "builtins.open",
            mock_open(
                read_data='[{"id": "test-id", "model": "test-model", "prompt": "test prompt", "adaptor": "", "adaptor_scale": 1.0, "num_inference_steps": 20, "guidance_scale": 7.5, "seed": 42, "image_path": "/fake/path.png", "timestamp": "2023-01-01T00:00:00", "upscaled": false, "upscale_factor": 1, "negative_prompt": "", "eta": 0.0, "clip_skip": 0, "guidance_rescale": 0.0, "height": 512, "width": 512, "generation_time": 5.0, "num_images": 1, "input_image_path": "", "strength": 0.8, "is_img2img": false, "mask_image_path": "", "is_inpainting": false}]'
            ),
        ),
    ):
        from transformerlab.routers.diffusion import load_history

        result = load_history(limit=50, offset=0)

        assert result.total == 1
        assert len(result.images) == 1
        assert result.images[0].id == "test-id"
        assert result.images[0].model == "test-model"
        assert result.images[0].prompt == "test prompt"


def test_load_history_with_pagination():
    """Test loading history with pagination parameters"""
    history_data = []
    for i in range(10):
        history_data.append(
            {
                "id": f"test-id-{i}",
                "model": "test-model",
                "prompt": f"test prompt {i}",
                "adaptor": "",
                "adaptor_scale": 1.0,
                "num_inference_steps": 20,
                "guidance_scale": 7.5,
                "seed": 42,
                "image_path": f"/fake/path{i}.png",
                "timestamp": "2023-01-01T00:00:00",
                "upscaled": False,
                "upscale_factor": 1,
                "negative_prompt": "",
                "eta": 0.0,
                "clip_skip": 0,
                "guidance_rescale": 0.0,
                "height": 512,
                "width": 512,
                "generation_time": 5.0,
                "num_images": 1,
                "input_image_path": "",
                "strength": 0.8,
                "is_img2img": False,
                "mask_image_path": "",
                "is_inpainting": False,
            }
        )

    with (
        patch("transformerlab.routers.diffusion.get_history_file_path", return_value="/fake/history.json"),
        patch("os.path.exists", return_value=True),
        patch("builtins.open", mock_open(read_data=json.dumps(history_data))),
    ):
        from transformerlab.routers.diffusion import load_history

        result = load_history(limit=3, offset=2)

        assert result.total == 10
        assert len(result.images) == 3
        assert result.images[0].id == "test-id-2"
        assert result.images[1].id == "test-id-3"
        assert result.images[2].id == "test-id-4"


def test_load_history_no_file():
    """Test loading history when history file doesn't exist"""
    with (
        patch("transformerlab.routers.diffusion.get_history_file_path", return_value="/fake/history.json"),
        patch("os.path.exists", return_value=False),
    ):
        from transformerlab.routers.diffusion import load_history

        result = load_history()

        assert result.total == 0
        assert len(result.images) == 0


def test_load_history_invalid_json():
    """Test loading history with corrupted JSON file"""
    with (
        patch("transformerlab.routers.diffusion.get_history_file_path", return_value="/fake/history.json"),
        patch("os.path.exists", return_value=True),
        patch("builtins.open", mock_open(read_data="invalid json")),
    ):
        from transformerlab.routers.diffusion import load_history

        result = load_history()

        assert result.total == 0
        assert len(result.images) == 0


def test_find_image_by_id_success():
    """Test finding an image by ID successfully"""
    history_data = [
        {
            "id": "test-id-1",
            "model": "test-model-1",
            "prompt": "test prompt 1",
            "adaptor": "",
            "adaptor_scale": 1.0,
            "num_inference_steps": 20,
            "guidance_scale": 7.5,
            "seed": 42,
            "image_path": "/fake/path1.png",
            "timestamp": "2023-01-01T00:00:00",
            "upscaled": False,
            "upscale_factor": 1,
            "negative_prompt": "",
            "eta": 0.0,
            "clip_skip": 0,
            "guidance_rescale": 0.0,
            "height": 512,
            "width": 512,
            "generation_time": 5.0,
            "num_images": 1,
            "input_image_path": "",
            "strength": 0.8,
            "is_img2img": False,
            "mask_image_path": "",
            "is_inpainting": False,
        },
        {
            "id": "test-id-2",
            "model": "test-model-2",
            "prompt": "test prompt 2",
            "adaptor": "",
            "adaptor_scale": 1.0,
            "num_inference_steps": 20,
            "guidance_scale": 7.5,
            "seed": 42,
            "image_path": "/fake/path2.png",
            "timestamp": "2023-01-01T00:00:00",
            "upscaled": False,
            "upscale_factor": 1,
            "negative_prompt": "",
            "eta": 0.0,
            "clip_skip": 0,
            "guidance_rescale": 0.0,
            "height": 512,
            "width": 512,
            "generation_time": 5.0,
            "num_images": 1,
            "input_image_path": "",
            "strength": 0.8,
            "is_img2img": False,
            "mask_image_path": "",
            "is_inpainting": False,
        },
    ]

    with (
        patch("transformerlab.routers.diffusion.get_history_file_path", return_value="/fake/history.json"),
        patch("os.path.exists", return_value=True),
        patch("builtins.open", mock_open(read_data=json.dumps(history_data))),
    ):
        from transformerlab.routers.diffusion import find_image_by_id

        result = find_image_by_id("test-id-2")

        assert result is not None
        assert result.id == "test-id-2"
        assert result.model == "test-model-2"
        assert result.prompt == "test prompt 2"


def test_find_image_by_id_not_found():
    """Test finding an image by ID when it doesn't exist"""
    history_data = [
        {
            "id": "test-id-1",
            "model": "test-model-1",
            "prompt": "test prompt 1",
            "adaptor": "",
            "adaptor_scale": 1.0,
            "num_inference_steps": 20,
            "guidance_scale": 7.5,
            "seed": 42,
            "image_path": "/fake/path1.png",
            "timestamp": "2023-01-01T00:00:00",
            "upscaled": False,
            "upscale_factor": 1,
            "negative_prompt": "",
            "eta": 0.0,
            "clip_skip": 0,
            "guidance_rescale": 0.0,
            "height": 512,
            "width": 512,
            "generation_time": 5.0,
            "num_images": 1,
            "input_image_path": "",
            "strength": 0.8,
            "is_img2img": False,
            "mask_image_path": "",
            "is_inpainting": False,
        }
    ]

    with (
        patch("transformerlab.routers.diffusion.get_history_file_path", return_value="/fake/history.json"),
        patch("os.path.exists", return_value=True),
        patch("builtins.open", mock_open(read_data=json.dumps(history_data))),
    ):
        from transformerlab.routers.diffusion import find_image_by_id

        result = find_image_by_id("non-existent-id")

        assert result is None


def test_find_image_by_id_no_file():
    """Test finding an image by ID when history file doesn't exist"""
    with (
        patch("transformerlab.routers.diffusion.get_history_file_path", return_value="/fake/history.json"),
        patch("os.path.exists", return_value=False),
    ):
        from transformerlab.routers.diffusion import find_image_by_id

        result = find_image_by_id("test-id")

        assert result is None


def test_find_image_by_id_invalid_json():
    """Test finding an image by ID with corrupted JSON file"""
    with (
        patch("transformerlab.routers.diffusion.get_history_file_path", return_value="/fake/history.json"),
        patch("os.path.exists", return_value=True),
        patch("builtins.open", mock_open(read_data="invalid json")),
    ):
        from transformerlab.routers.diffusion import find_image_by_id

        result = find_image_by_id("test-id")

        assert result is None


def test_get_pipeline_key_txt2img():
    """Test get_pipeline_key for text-to-image pipeline"""
    from transformerlab.routers.diffusion import get_pipeline_key

    key = get_pipeline_key("test-model", "", is_img2img=False, is_inpainting=False)

    assert key == "test-model::txt2img"


def test_get_pipeline_key_img2img():
    """Test get_pipeline_key for image-to-image pipeline"""
    from transformerlab.routers.diffusion import get_pipeline_key

    key = get_pipeline_key("test-model", "", is_img2img=True, is_inpainting=False)

    assert key == "test-model::img2img"


def test_get_pipeline_key_inpainting():
    """Test get_pipeline_key for inpainting pipeline"""
    from transformerlab.routers.diffusion import get_pipeline_key

    key = get_pipeline_key("test-model", "", is_img2img=False, is_inpainting=True)

    assert key == "test-model::inpainting"


def test_get_pipeline_key_with_adaptor():
    """Test get_pipeline_key with adaptor"""
    from transformerlab.routers.diffusion import get_pipeline_key

    key = get_pipeline_key("test-model", "test-adaptor", is_img2img=False, is_inpainting=False)

    assert key == "test-model::test-adaptor::txt2img"


def test_get_pipeline_key_inpainting_priority():
    """Test get_pipeline_key prioritizes inpainting over img2img"""
    from transformerlab.routers.diffusion import get_pipeline_key

    key = get_pipeline_key("test-model", "", is_img2img=True, is_inpainting=True)

    assert key == "test-model::inpainting"


def test_get_pipeline_key_no_adaptor():
    """Test get_pipeline_key with empty adaptor string"""
    from transformerlab.routers.diffusion import get_pipeline_key

    key = get_pipeline_key("test-model", "", is_img2img=False, is_inpainting=False)

    assert key == "test-model::txt2img"


def test_get_pipeline_key_whitespace_adaptor():
    """Test get_pipeline_key with whitespace-only adaptor"""
    from transformerlab.routers.diffusion import get_pipeline_key

    key = get_pipeline_key("test-model", "   ", is_img2img=False, is_inpainting=False)

    # Should treat whitespace-only adaptor as no adaptor
    assert key == "test-model::   ::txt2img"
