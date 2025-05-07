import requests


def test_plugins_gallery(live_server):
    resp = requests.get(f"{live_server}/plugins/gallery")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    if data:
        plugin = data[0]
        assert "name" in plugin or "description" in plugin


def test_plugins_list(live_server):
    resp = requests.get(f"{live_server}/plugins/list")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    if data:
        plugin = data[0]
        assert "name" in plugin or "description" in plugin


def test_plugins_install(live_server):
    resp = requests.get(f"{live_server}/plugins/gallery/fastchat_server/install")
    assert resp.status_code in (200, 404)
    if resp.status_code == 200:
        data = resp.json()
        assert "message" in data or "status" in data
