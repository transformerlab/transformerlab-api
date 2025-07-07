import pytest


def test_create_empty_workflow(client):
    resp = client.get("/experiment/1/workflows/create_empty", params={"name": "testwf"})
    assert resp.status_code == 200
    workflow_id = resp.json()
    assert workflow_id
    # Cleanup
    del_resp = client.get(f"/experiment/1/workflows/delete/{workflow_id}")
    assert del_resp.status_code == 200
    assert del_resp.json().get("message") == "OK"


def test_list_workflows(client):
    # Create a workflow to ensure at least one exists
    create_resp = client.get("/experiment/1/workflows/create_empty", params={"name": "listtest"})
    workflow_id = create_resp.json()
    resp = client.get("/experiment/1/workflows/list")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)
    # Cleanup
    del_resp = client.get(f"/experiment/1/workflows/delete/{workflow_id}")
    assert del_resp.status_code == 200
    assert del_resp.json().get("message") == "OK"


def test_delete_workflow(client):
    # Create a workflow to delete
    create_resp = client.get("/experiment/1/workflows/create_empty", params={"name": "todelete"})
    workflow_id = create_resp.json()
    del_resp = client.get(f"/experiment/1/workflows/delete/{workflow_id}")
    assert del_resp.status_code == 200
    assert del_resp.json().get("message") == "OK"
