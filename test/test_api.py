import subprocess
import time
import pytest
import requests


@pytest.fixture(scope="session")
def live_server():
    # Get a free port
    import socket

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()

    # Start the server process
    print("about to run: ./run.sh -p", port)
    server_process = subprocess.Popen(["./run.sh", "-p", str(port)])

    # Give it time to start
    time.sleep(5)

    base_url = f"http://0.0.0.0:{port}"

    # Verify the server is running
    import requests

    try:
        response = requests.get(f"{base_url}/")
        assert response.status_code == 200
    except Exception as e:
        server_process.terminate()
        raise Exception(f"Failed to start server: {e}")

    yield base_url

    # Teardown - stop the server
    server_process.terminate()
    server_process.wait()


# Tests using the live server
def test_root(live_server):
    response = requests.get(f"{live_server}/")
    assert response.status_code == 200


def test_server_info(live_server):
    response = requests.get(f"{live_server}/server/info")
    assert response.status_code == 200
    data = response.json()
    assert "cpu" in data
    assert "memory" in data
    assert "disk" in data
    assert "gpu" in data


def test_server_python_libraries(live_server):
    response = requests.get(f"{live_server}/server/python_libraries")
    assert response.status_code == 200
    data = response.json()
    # assert it is an array of {"name": "package_name", "version": "version_number"} type things
    assert isinstance(data, list)
    for package in data:
        assert isinstance(package, dict)
        assert "name" in package
        assert "version" in package
        assert isinstance(package["name"], str)
        assert isinstance(package["version"], str)


def test_server_pytorch_collect_env(live_server):
    response = requests.get(f"{live_server}/server/pytorch_collect_env")
    assert response.status_code == 200
    data = response.text
    assert "PyTorch" in data
