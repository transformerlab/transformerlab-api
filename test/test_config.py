import os
import shutil
import pytest
from fastapi.testclient import TestClient

os.environ["TFL_HOME_DIR"] = "./test/tmp/"
os.environ["TFL_WORKSPACE_DIR"] = "./test/tmp"

from api import app
import transformerlab.db as db
from transformerlab.shared import dirs

client = TestClient(app)


@pytest.fixture(scope="module", autouse=True)
async def setup_db():
    await db.init()
    yield
    await db.close()
    # delete the contents of the temporary workspace directory
    # but do not delete the directory itself
    for filename in os.listdir(dirs.WORKSPACE_DIR):
        if filename.startswith("."):  # ignore hidden files like .gitignore
            continue
        file_path = os.path.join(dirs.WORKSPACE_DIR, filename)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)


def test_config_get():
    # First, set a config value
    response = client.get("/config/set?k=test_key&v=test_value")
    assert response.status_code == 200
    assert response.json() == {"key": "test_key", "value": "test_value"}

    # Then, get the config value
    response = client.get("/config/get/test_key")
    assert response.status_code == 200
    assert response.json() == "test_value"
