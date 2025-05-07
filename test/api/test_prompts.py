import requests


def test_prompts_list(live_server):
    resp = requests.get(f"{live_server}/prompts/list")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list) or isinstance(resp.json(), dict)


def test_prompts_dummy(live_server):
    resp = requests.get(f"{live_server}/prompts/list?prompt_id=dummy")
    assert resp.status_code in (200, 400, 404)
