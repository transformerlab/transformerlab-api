from fastapi.testclient import TestClient
from api import app


def test_data_gallery():
    with TestClient(app) as client:
        resp = client.get("/data/gallery")
        assert resp.status_code == 200
        assert "data" in resp.json() or "status" in resp.json()


def test_data_list():
    with TestClient(app) as client:
        resp = client.get("/data/list")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list) or isinstance(resp.json(), dict)


def test_data_info():
    with TestClient(app) as client:
        resp = client.get("/data/info?dataset_id=dummy_dataset")
        assert resp.status_code in (200, 400, 404)


def test_data_preview():
    with TestClient(app) as client:
        resp = client.get("/data/preview?dataset_id=dummy_dataset")
        assert resp.status_code in (200, 400, 404)


def test_data_preview_trelis_touch_rugby_rules():
    with TestClient(app) as client:
        resp = client.get("/data/preview", params={"dataset_id": "Trelis/touch-rugby-rules", "limit": 2})
        assert resp.status_code in (200, 400, 404)
        if resp.status_code == 200 and resp.json().get("status") == "success":
            data = resp.json()["data"]
            assert "len" in data
            # Should have either columns or rows
            assert "columns" in data or "rows" in data
