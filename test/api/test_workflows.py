from fastapi.testclient import TestClient
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

        resp = client.get(f"/experiment/{exp_id}/workflows/create?name=test_workflow")
        assert resp.status_code in (200, 400, 404)


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
