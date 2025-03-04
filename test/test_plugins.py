import pytest
from fastapi.testclient import TestClient
import os

os.environ["TFL_HOME_DIR"] = "./test/tmp"
os.environ["TFL_WORKSPACE_DIR"] = "./test/tmp"

from api import app

client = TestClient(app)


@pytest.fixture(scope="module", autouse=True)
def setup_module(tmp_path):
    print(tmp_path)
    # Setup code to create necessary directories and files for testing
    # os.makedirs(dirs.PLUGIN_PRELOADED_GALLERY, exist_ok=True)
    # os.makedirs(dirs.PLUGIN_DIR, exist_ok=True)
    yield
    # Teardown code to clean up after tests
    # os.rmdir(dirs.PLUGIN_PRELOADED_GALLERY)
    # os.rmdir(dirs.PLUGIN_DIR)


def test_plugin_gallery():
    response = client.get("/plugins/gallery")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


def test_install_plugin():
    plugin_id = "sample_plugin"
    response = client.get(f"/plugins/gallery/{plugin_id}/install")
    assert response.status_code == 200
    data = response.json()
    assert "error" not in data


def test_list_plugins():
    response = client.get("/plugins/list")
    print(response.json())
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


def test_list_missing_plugins_for_current_platform():
    response = client.get("/plugins/list_missing_plugins_for_current_platform")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


# def test_install_missing_plugins_for_current_platform():
#     response = client.get("/plugins/install_missing_plugins_for_current_platform")
#     assert response.status_code == 200
#     data = response.json()
#     assert isinstance(data, list)
