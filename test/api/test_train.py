def test_train_templates(client):
    resp = client.get("/train/templates")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list) or isinstance(resp.json(), dict)


def test_train_recipe_gallery(client):
    resp = client.get("/train/template/gallery")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list) or isinstance(resp.json(), dict)


def test_train_export_recipe(client):
    resp = client.get("/train/template/1/export")
    assert resp.status_code in (200, 404)


def test_train_create_template(client):
    data = {"name": "test_template", "description": "desc", "type": "test", "config": "{}"}
    resp = client.post("/train/template/create", data=data)
    assert resp.status_code in (200, 422, 400)

def test_train_sweep_results(client):
    resp = client.get("/train/job/1/sweep_results")
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data
    assert data["status"] in ("success", "error")
    if data["status"] == "success":
        assert "data" in data
    else:
        assert "message" in data