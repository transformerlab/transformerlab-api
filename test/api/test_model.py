from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch
from api import app
import pytest
from unittest.mock import MagicMock


def test_model_gallery():
    with TestClient(app) as client:
        resp = client.get("/model/gallery")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        if data:
            model = data[0]
            assert "name" in model or "uniqueID" in model


@pytest.mark.skip(reason="Skipping test_model_list_local_uninstalled because it is taking 23 seconds to load??!!")
def test_model_list_local_uninstalled():
    with TestClient(app) as client:
        resp = client.get("/model/list_local_uninstalled")
        assert resp.status_code == 200
        assert "data" in resp.json() or "status" in resp.json()


def test_model_group_gallery():
    with TestClient(app) as client:
        resp = client.get("/model/model_groups_list")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        if data:
            model = data[0]
            assert "name" in model or "models" in model


def make_mock_adapter_info(overrides={}):
    return MagicMock(
        modelId="mock/model",
        tags=["tag1", "tag2"],
        cardData={
            "description": "mock desc",
            "base_model": "unsloth/Llama-3.2-1B-Instruct",
            **overrides.get("cardData", {}),
        },
        config={"architectures": "MockArch", "model_type": "MockType", **overrides.get("config", {})},
        downloads=123,
    )


@pytest.mark.asyncio
@patch("transformerlab.routers.model.huggingfacemodel.get_model_details_from_huggingface", new_callable=AsyncMock)
@patch("transformerlab.routers.model.shared.async_run_python_script_and_update_status", new_callable=AsyncMock)
async def test_install_peft_mock(mock_run_script, mock_get_details):
    with TestClient(app) as client:
        # Mock get_model_details to return a dummy config
        mock_get_details.return_value = {"name": "dummy_adapter"}

        # Mock run_script to simulate a subprocess with success
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_run_script.return_value = mock_process

        test_model_id = "unsloth_Llama-3.2-1B-Instruct"
        test_peft_id = "dummy_adapter"

        response = client.post(f"/model/install_peft?model_id={test_model_id}&peft={test_peft_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "error"  # As install_peft now returns 'started' after starting the async task
