import requests


def test_model_gallery(live_server):
    resp = requests.get(f"{live_server}/model/gallery")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    if data:
        model = data[0]
        assert "name" in model or "uniqueID" in model


def test_model_list_local_uninstalled(live_server):
    resp = requests.get(f"{live_server}/model/list_local_uninstalled")
    assert resp.status_code == 200
    assert "data" in resp.json() or "status" in resp.json()
