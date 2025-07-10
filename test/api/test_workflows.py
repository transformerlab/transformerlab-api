import json


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


def test_workflow_update_name(client):
    create_resp = client.get("/experiment/1/workflows/create_empty", params={"name": "updatename"})
    workflow_id = create_resp.json()
    resp = client.get(f"/experiment/1/workflows/{workflow_id}/update_name", params={"new_name": "updatedname"})
    assert resp.status_code == 200
    assert resp.json().get("message") == "OK"
    # Cleanup
    client.get(f"/experiment/1/workflows/delete/{workflow_id}")


def test_workflow_add_and_delete_node(client):
    create_resp = client.get("/experiment/1/workflows/create_empty", params={"name": "addnode"})
    workflow_id = create_resp.json()
    node = {"type": "TASK", "name": "Test Task", "task": "test_task", "out": []}
    add_node_resp = client.get(f"/experiment/1/workflows/{workflow_id}/add_node", params={"node": json.dumps(node)})
    assert add_node_resp.status_code == 200
    # Get workflow config to find node id
    wf_resp = client.get("/experiment/1/workflows/list")
    workflow = next(w for w in wf_resp.json() if w["id"] == workflow_id)
    config = workflow["config"]
    if not isinstance(config, dict):
        config = json.loads(config)
    task_node = next(n for n in config["nodes"] if n["type"] == "TASK")
    node_id = task_node["id"]
    # Delete node
    del_node_resp = client.get(f"/experiment/1/workflows/{workflow_id}/{node_id}/delete_node")
    assert del_node_resp.status_code == 200
    # Cleanup
    client.get(f"/experiment/1/workflows/delete/{workflow_id}")


def test_workflow_edit_node_metadata(client):
    create_resp = client.get("/experiment/1/workflows/create_empty", params={"name": "editmeta"})
    workflow_id = create_resp.json()
    node = {"type": "TASK", "name": "Meta Task", "task": "test_task", "out": []}
    client.get(f"/experiment/1/workflows/{workflow_id}/add_node", params={"node": json.dumps(node)})
    wf_resp = client.get("/experiment/1/workflows/list")
    workflow = next(w for w in wf_resp.json() if w["id"] == workflow_id)
    config = workflow["config"]
    if not isinstance(config, dict):
        config = json.loads(config)
    task_node = next(n for n in config["nodes"] if n["type"] == "TASK")
    node_id = task_node["id"]
    meta = {"desc": "testdesc"}
    edit_resp = client.get(
        f"/experiment/1/workflows/{workflow_id}/{node_id}/edit_node_metadata", params={"metadata": json.dumps(meta)}
    )
    assert edit_resp.status_code == 200
    # Cleanup
    client.get(f"/experiment/1/workflows/delete/{workflow_id}")


def test_workflow_add_and_remove_edge(client):
    create_resp = client.get("/experiment/1/workflows/create_empty", params={"name": "edgecase"})
    workflow_id = create_resp.json()
    node1 = {"type": "TASK", "name": "Task1", "task": "task1", "out": []}
    node2 = {"type": "TASK", "name": "Task2", "task": "task2", "out": []}
    client.get(f"/experiment/1/workflows/{workflow_id}/add_node", params={"node": json.dumps(node1)})
    client.get(f"/experiment/1/workflows/{workflow_id}/add_node", params={"node": json.dumps(node2)})
    wf_resp = client.get("/experiment/1/workflows/list")
    workflow = next(w for w in wf_resp.json() if w["id"] == workflow_id)
    config = workflow["config"]
    if not isinstance(config, dict):
        config = json.loads(config)
    task_nodes = [n for n in config["nodes"] if n["type"] == "TASK"]
    node1_id, node2_id = task_nodes[0]["id"], task_nodes[1]["id"]
    add_edge_resp = client.post(
        f"/experiment/1/workflows/{workflow_id}/{node1_id}/add_edge", params={"end_node_id": node2_id}
    )
    assert add_edge_resp.status_code == 200
    remove_edge_resp = client.post(
        f"/experiment/1/workflows/{workflow_id}/{node1_id}/remove_edge", params={"end_node_id": node2_id}
    )
    assert remove_edge_resp.status_code == 200
    # Cleanup
    client.get(f"/experiment/1/workflows/delete/{workflow_id}")


def test_workflow_export_to_yaml(client):
    create_resp = client.get("/experiment/1/workflows/create_empty", params={"name": "yamltest"})
    workflow_id = create_resp.json()
    export_resp = client.get(f"/experiment/1/workflows/{workflow_id}/export_to_yaml")
    assert export_resp.status_code == 200
    # Cleanup
    client.get(f"/experiment/1/workflows/delete/{workflow_id}")


def test_workflow_add_eval_node_and_metadata(client):
    create_resp = client.get("/experiment/1/workflows/create_empty", params={"name": "evalnode"})
    workflow_id = create_resp.json()
    # Add EVAL node with realistic structure
    node = {"name": "hello", "task": "WarmPanda", "type": "EVAL", "metadata": {}, "out": []}
    add_node_resp = client.get(f"/experiment/1/workflows/{workflow_id}/add_node", params={"node": json.dumps(node)})
    assert add_node_resp.status_code == 200
    # Get workflow config to find node id
    wf_resp = client.get("/experiment/1/workflows/list")
    workflow = next(w for w in wf_resp.json() if w["id"] == workflow_id)
    config = workflow["config"]
    if not isinstance(config, dict):
        config = json.loads(config)
    eval_node = next(n for n in config["nodes"] if n["type"] == "EVAL")
    node_id = eval_node["id"]
    # Edit metadata
    meta = {"desc": "eval node test"}
    edit_resp = client.get(
        f"/experiment/1/workflows/{workflow_id}/{node_id}/edit_node_metadata", params={"metadata": json.dumps(meta)}
    )
    assert edit_resp.status_code == 200
    # Cleanup
    client.get(f"/experiment/1/workflows/delete/{workflow_id}")
