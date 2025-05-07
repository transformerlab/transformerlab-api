import requests


def test_train_templates(live_server):
    resp = requests.get(f"{live_server}/train/templates")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list) or isinstance(resp.json(), dict)


def test_train_recipe_gallery(live_server):
    resp = requests.get(f"{live_server}/train/template/gallery")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list) or isinstance(resp.json(), dict)


def test_train_export_recipe(live_server):
    resp = requests.get(f"{live_server}/train/template/1/export")
    assert resp.status_code in (200, 404)


def test_train_create_template(live_server):
    data = {"name": "test_template", "description": "desc", "type": "test", "config": "{}"}
    resp = requests.post(f"{live_server}/train/template/create", data=data)
    assert resp.status_code in (200, 422, 400)
