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


def test_save_metadata():
    from io import BytesIO
    import json

    test_data = [
        {"__index__": 0, "image": "dummy.jpg", "text": "caption A"},
        {"__index__": 1, "image": "dummy2.jpg", "text": "caption B"},
    ]

    with TestClient(app) as client:
        response = client.post(
            "/data/save_metadata",
            data={"dataset_id": "dummy_dataset"},
            files={"file": ("patch.json", BytesIO(json.dumps(test_data).encode("utf-8")), "application/json")},
        )
        assert response.status_code in (200, 400, 404)
