from fastapi.testclient import TestClient
from api import app


def test_plugins_gallery():
    with TestClient(app) as client:
        resp = client.get("/plugins/gallery")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        if data:
            plugin = data[0]
            assert "name" in plugin or "description" in plugin


def test_plugins_list():
    with TestClient(app) as client:
        resp = client.get("/plugins/list")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        if data:
            plugin = data[0]
            assert "name" in plugin or "description" in plugin


def test_plugins_install():
    with TestClient(app) as client:
        resp = client.get("/plugins/gallery/fastchat_server/install")
        assert resp.status_code in (200, 404)
        if resp.status_code == 200:
            data = resp.json()
            assert "message" in data or "status" in data


def test_run_installer_script():
    with TestClient(app) as client:
        resp = client.get("/plugins/fastchat_server/run_installer_script")
        # Installer may not exist, so allow 200 or 404
        assert resp.status_code in (200, 404)
        if resp.status_code == 200:
            data = resp.json()
            assert "message" in data or "status" in data


def test_list_missing_plugins_for_current_platform():
    with TestClient(app) as client:
        resp = client.get("/plugins/list_missing_plugins_for_current_platform")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)


def test_install_missing_plugins_for_current_platform():
    with TestClient(app) as client:
        resp = client.get("/plugins/install_missing_plugins_for_current_platform")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)


def test_autoupdate_all_plugins():
    with TestClient(app) as client:
        resp = client.get("/plugins/autoupdate_all_plugins")
        assert resp.status_code in (200, 404)
        if resp.status_code == 200:
            data = resp.json()
            assert "message" in data or "status" in data
