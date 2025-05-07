import requests


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
