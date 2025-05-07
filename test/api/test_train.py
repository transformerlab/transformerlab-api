import requests


def test_train_templates(live_server):
    resp = requests.get(f"{live_server}/train/templates")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list) or isinstance(resp.json(), dict)


def test_train_recipe_gallery(live_server):
    resp = requests.get(f"{live_server}/train/template/gallery")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list) or isinstance(resp.json(), dict)
