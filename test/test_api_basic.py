import os
from fastapi.testclient import TestClient

os.environ["TFL_HOME_DIR"] = "./test/tmp"
os.environ["TFL_WORKSPACE_DIR"] = "./test/tmp"

from api import app

client = TestClient(app)


# DISABLING: The root now serves the web app if it has been installed.
# def test_read_main():
#    response = client.get("/")
#    assert response.status_code == 200
#    assert response.json() == {"msg": "Welcome to Transformer Lab!"}
