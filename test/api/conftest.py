import pytest
from fastapi.testclient import TestClient
from api import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


# No longer needed: live_server fixture for external server. All tests now use TestClient in-process.
# This file can be deleted or left empty.
