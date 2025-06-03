from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch
from api import app
import pytest


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


def test_peft_adapter_search_install_delete():
    adapter_id = "tcotter/Llama-3.2-1B-Instruct-Mojo-Adapter"
    model_id = "unsloth/Llama-3.2-1B-Instruct"

    with TestClient(app) as client:
        # 1. Search PEFT adapter
        resp = client.get(f"/model/search_peft?peft={adapter_id}&model_id={model_id}&device=cpu")
        assert resp.status_code == 200
        results = resp.json()
        assert isinstance(results, list)
        assert results[0]["adapter_id"] == adapter_id

        # 2. Install PEFT adapter
        resp = client.post(f"/model/install_peft?peft={adapter_id}&model_id={model_id}")
        assert resp.status_code == 200
        result = resp.json()
        assert result["status"] == "started"
        assert "job_id" in result

        # 3. Delete PEFT adapter (clean-up)
        secure_model_id = model_id.replace("/", "_")
        secure_peft = adapter_id.replace("/", "_")
        resp = client.get(f"/model/delete_peft?model_id={secure_model_id}&peft={secure_peft}")
        assert resp.status_code == 200
        assert resp.json()["message"] == "success"
