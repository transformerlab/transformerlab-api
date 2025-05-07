import requests


def test_plugins_gallery(live_server):
    resp = requests.get(f"{live_server}/plugins/gallery")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list) or isinstance(resp.json(), dict)


def test_plugins_list(live_server):
    resp = requests.get(f"{live_server}/plugins/list")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list) or isinstance(resp.json(), dict)
