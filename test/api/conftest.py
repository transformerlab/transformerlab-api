import subprocess
import time
import pytest


@pytest.fixture(scope="session")
def live_server():
    # Get a free port
    import socket

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()

    # Start the server process
    print("about to run: ./run.sh -p", port)
    server_process = subprocess.Popen(["./run.sh", "-h", "127.0.0.1", "-p", str(port)])

    # Give it time to start
    time.sleep(5)

    base_url = f"http://127.0.0.1:{port}"

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
