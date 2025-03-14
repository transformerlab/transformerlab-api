import os
from fastapi.testclient import TestClient

os.environ["TFL_HOME_DIR"] = "./test/tmp"
os.environ["TFL_WORKSPACE_DIR"] = "./test/tmp"

from api import app

client = TestClient(app)


def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"msg": "Welcome to Transformer Lab!"}
