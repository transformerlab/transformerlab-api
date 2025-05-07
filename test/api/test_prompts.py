import requests


def test_prompts_list(live_server):
    resp = requests.get(f"{live_server}/prompts/list")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list) or isinstance(resp.json(), dict)
