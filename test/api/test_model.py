from fastapi.testclient import TestClient
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