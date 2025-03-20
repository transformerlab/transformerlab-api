import pytest
from fastapi.testclient import TestClient
import os

os.environ["TFL_HOME_DIR"] = "./test/tmp"
os.environ["TFL_WORKSPACE_DIR"] = "./test/tmp"

from api import app
from transformerlab.shared import dirs


client = TestClient(app)

print("dirs.GLOBAL_LOG_PATH", dirs.GLOBAL_LOG_PATH)


@pytest.fixture(scope="module", autouse=True)
def setup_module():
    # confirm that the global log path is set to ./test/tmp/transformerlab.log
    assert dirs.GLOBAL_LOG_PATH == "./test/tmp/transformerlab.log"
    # if the file doesn't exist, create it:
    if not os.path.exists(dirs.GLOBAL_LOG_PATH):
        with open(dirs.GLOBAL_LOG_PATH, "w") as f:
            f.write("Test log file")
    yield
    # Teardown delete file above:
    os.remove(dirs.GLOBAL_LOG_PATH)


def test_get_computer_information():
    response = client.get("/server/info")
    assert response.status_code == 200
    data = response.json()
    assert "cpu" in data
    assert "name" in data
    assert "platform" in data


def test_get_python_library_versions():
    response = client.get("/server/python_libraries")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert any(pkg["name"] == "fastapi" for pkg in data)


def test_get_pytorch_collect_env():
    response = client.get("/server/pytorch_collect_env")
    assert response.status_code == 200
    data = response.text
    assert "PyTorch version" in data


# @pytest.mark.asyncio
# async def test_watch_log():
#     response = client.get("/server/stream_log")
#     assert response.status_code == 200
#     assert response.headers["content-type"] == "text/event-stream"
