import requests


# Tests using the live server
def test_root(live_server):
    response = requests.get(f"{live_server}/")
    assert response.status_code == 200
