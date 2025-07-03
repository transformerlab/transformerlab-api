import pytest
from fastapi.testclient import TestClient
import os

os.environ["TFL_HOME_DIR"] = "test/tmp/"
os.environ["TFL_WORKSPACE_DIR"] = "test/tmp"

from api import app


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c
