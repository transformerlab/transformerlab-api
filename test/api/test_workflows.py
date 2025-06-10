from fastapi.testclient import TestClient
import json
from api import app


def test_workflows_list():
    with TestClient(app) as client:
        # First create an experiment
        exp_resp = client.get("/experiment/create?name=test_workflows_list")
        assert exp_resp.status_code == 200
        exp_id = exp_resp.json()

        resp = client.get(f"/experiment/{exp_id}/workflows/list")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list) or isinstance(resp.json(), dict)


def test_workflows_delete():
    with TestClient(app) as client:
        # First create an experiment
        exp_resp = client.get("/experiment/create?name=test_workflows_delete")
        assert exp_resp.status_code == 200
        exp_id = exp_resp.json()

        # Create a workflow to delete
        create_resp = client.get(f"/experiment/{exp_id}/workflows/create?name=workflow_to_delete")
        assert create_resp.status_code == 200
        workflow_id = create_resp.json()

        # Try to delete the workflow
        resp = client.get(f"/experiment/{exp_id}/workflows/delete/{workflow_id}")
        assert resp.status_code == 200
        assert resp.json() == {"message": "OK"}

        # Try to delete a non-existent workflow
        resp = client.get(f"/experiment/{exp_id}/workflows/delete/non_existent_workflow")
        assert resp.status_code == 200
        assert resp.json() == {"message": "Workflow not found"}


def test_workflows_create():
    with TestClient(app) as client:
        # First create an experiment
        exp_resp = client.get("/experiment/create?name=test_workflows_create")
        assert exp_resp.status_code == 200
        exp_id = exp_resp.json()

        # Create workflow with required fields
        config = {
            "nodes": [{"type": "START", "id": "start", "name": "START", "out": []}],
            "status": "CREATED"
        }
        resp = client.get(f"/experiment/{exp_id}/workflows/create?name=test_workflow&config={json.dumps(config)}")
        assert resp.status_code == 200
        assert resp.json() is not None  # Just check that we get a valid response


def test_experiment_workflows_list():
    """Test the new experiment workflows list endpoint"""
    with TestClient(app) as client:
        # First create an experiment
        exp_resp = client.get("/experiment/create?name=test_experiment_workflows_list")
        assert exp_resp.status_code == 200
        exp_id = exp_resp.json()

        # Create a workflow in the experiment
        workflow_resp = client.get(f"/experiment/{exp_id}/workflows/create?name=test_workflow")
        assert workflow_resp.status_code == 200

        # Test the new experiment workflows list endpoint
        resp = client.get(f"/experiment/{exp_id}/workflows/list")
        assert resp.status_code == 200
        workflows = resp.json()
        assert isinstance(workflows, list)
        assert len(workflows) > 0
        assert workflows[0]["experiment_id"] == exp_id


def test_experiment_workflow_runs():
    """Test the new experiment workflow runs endpoint"""
    with TestClient(app) as client:
        # First create an experiment
        exp_resp = client.get("/experiment/create?name=test_experiment_workflow_runs")
        assert exp_resp.status_code == 200
        exp_id = exp_resp.json()

        # Create a workflow in the experiment
        workflow_resp = client.get(f"/experiment/{exp_id}/workflows/create?name=test_workflow")
        assert workflow_resp.status_code == 200
        workflow_id = workflow_resp.json()

        # Queue the workflow to create a run
        queue_resp = client.get(f"/experiment/{exp_id}/workflows/{workflow_id}/start")
        assert queue_resp.status_code == 200

        # Test the new experiment workflow runs endpoint
        resp = client.get(f"/experiment/{exp_id}/workflows/runs")
        assert resp.status_code == 200
        runs = resp.json()
        assert isinstance(runs, list)
        assert len(runs) > 0
        assert runs[0]["experiment_id"] == exp_id
        assert runs[0]["workflow_id"] == workflow_id


def test_workflow_node_operations():
    """Test node-related operations in a workflow"""
    with TestClient(app) as client:
        # Create experiment
        exp_resp = client.get("/experiment/create?name=test_workflow_node_operations")
        assert exp_resp.status_code == 200
        exp_id = exp_resp.json()

        # Create workflow
        workflow_resp = client.get(f"/experiment/{exp_id}/workflows/create?name=test_workflow")
        assert workflow_resp.status_code == 200
        workflow_id = workflow_resp.json()

        # Add a node
        node_data = {
            "type": "TASK",
            "name": "Test Task",
            "task": "test_task",  # Required field
            "out": []  # Required field
        }
        add_node_resp = client.get(f"/experiment/{exp_id}/workflows/{workflow_id}/add_node?node={json.dumps(node_data)}")
        assert add_node_resp.status_code == 200
        
        # Get the workflow to find the node ID
        workflow_resp = client.get(f"/experiment/{exp_id}/workflows/list")
        assert workflow_resp.status_code == 200
        workflows = workflow_resp.json()
        workflow = next(w for w in workflows if w["id"] == workflow_id)
        nodes = json.loads(workflow["config"])["nodes"]
        node_id = next(n["id"] for n in nodes if n["type"] == "TASK")

        # Update node metadata
        metadata = {"key": "value"}
        metadata_resp = client.get(f"/experiment/{exp_id}/workflows/{workflow_id}/{node_id}/edit_node_metadata?metadata={json.dumps(metadata)}")
        assert metadata_resp.status_code == 200

        # Update node
        new_node = {
            "id": node_id,
            "type": "TASK",
            "name": "Updated Task",
            "task": "test_task",
            "out": []
        }
        update_resp = client.post(f"/experiment/{exp_id}/workflows/{workflow_id}/{node_id}/update_node", json=new_node)
        assert update_resp.status_code == 200

        # Add edge
        edge_resp = client.post(f"/experiment/{exp_id}/workflows/{workflow_id}/START/add_edge?end_node_id={node_id}")
        assert edge_resp.status_code == 200

        # Remove edge
        remove_edge_resp = client.post(f"/experiment/{exp_id}/workflows/{workflow_id}/START/remove_edge?end_node_id={node_id}")
        assert remove_edge_resp.status_code == 200

        # Delete node
        delete_node_resp = client.get(f"/experiment/{exp_id}/workflows/{workflow_id}/{node_id}/delete_node")
        assert delete_node_resp.status_code == 200


def test_workflow_name_update():
    """Test updating a workflow's name"""
    with TestClient(app) as client:
        # Create experiment
        exp_resp = client.get("/experiment/create?name=test_workflow_name_update")
        assert exp_resp.status_code == 200
        exp_id = exp_resp.json()

        # Create workflow
        workflow_resp = client.get(f"/experiment/{exp_id}/workflows/create?name=old_name")
        assert workflow_resp.status_code == 200
        workflow_id = workflow_resp.json()

        # Update name
        update_resp = client.get(f"/experiment/{exp_id}/workflows/{workflow_id}/update_name?new_name=new_name")
        assert update_resp.status_code == 200
        assert update_resp.json() == {"message": "OK"}


def test_workflow_yaml_operations():
    """Test YAML import/export operations"""
    with TestClient(app) as client:
        # Create experiment
        exp_resp = client.get("/experiment/create?name=test_workflow_yaml_operations")
        assert exp_resp.status_code == 200
        exp_id = exp_resp.json()

        # Create workflow with required fields
        config = {
            "nodes": [{"type": "START", "id": "start", "name": "START", "out": []}],
            "status": "CREATED"
        }
        workflow_resp = client.get(f"/experiment/{exp_id}/workflows/create?name=test_workflow&config={json.dumps(config)}")
        assert workflow_resp.status_code == 200
        workflow_id = workflow_resp.json()

        # Queue the workflow to create a workflow run with the required fields
        queue_resp = client.get(f"/experiment/{exp_id}/workflows/{workflow_id}/start")
        assert queue_resp.status_code == 200

        # Export to YAML - using the experiment-scoped path
        export_resp = client.get(f"/experiment/{exp_id}/workflows/{workflow_id}/export_to_yaml")
        assert export_resp.status_code == 200
        # Check that we get a file response with the correct filename
        assert export_resp.headers.get("content-type") == "text/plain; charset=utf-8"
        assert export_resp.headers.get("content-disposition") == 'attachment; filename="test_workflow.yaml"'


def test_workflow_run_operations():
    """Test workflow run operations"""
    with TestClient(app) as client:
        # Create experiment
        exp_resp = client.get("/experiment/create?name=test_workflow_run_operations")
        assert exp_resp.status_code == 200
        exp_id = exp_resp.json()

        # Create workflow
        workflow_resp = client.get(f"/experiment/{exp_id}/workflows/create?name=test_workflow")
        assert workflow_resp.status_code == 200
        workflow_id = workflow_resp.json()

        # Start workflow
        start_resp = client.get(f"/experiment/{exp_id}/workflows/{workflow_id}/start")
        assert start_resp.status_code == 200

        # Get workflow runs
        runs_resp = client.get(f"/experiment/{exp_id}/workflows/runs")
        assert runs_resp.status_code == 200
        runs = runs_resp.json()
        assert isinstance(runs, list)
        assert len(runs) > 0
        run_id = runs[0]["id"]

        # Get specific run
        run_resp = client.get(f"/experiment/{exp_id}/workflows/runs/{run_id}")
        assert run_resp.status_code == 200
        run_data = run_resp.json()
        assert "run" in run_data
        assert "workflow" in run_data
        assert "jobs" in run_data


def test_workflow_next_step():
    """Test workflow next step execution"""
    with TestClient(app) as client:
        # Create experiment
        exp_resp = client.get("/experiment/create?name=test_workflow_next_step")
        assert exp_resp.status_code == 200
        exp_id = exp_resp.json()

        # Create workflow
        workflow_resp = client.get(f"/experiment/{exp_id}/workflows/create?name=test_workflow")
        assert workflow_resp.status_code == 200
        workflow_id = workflow_resp.json()

        # Start workflow
        start_resp = client.get(f"/experiment/{exp_id}/workflows/{workflow_id}/start")
        assert start_resp.status_code == 200

        # Execute next step
        next_step_resp = client.get(f"/experiment/{exp_id}/workflows/start_next_step")
        assert next_step_resp.status_code == 200
