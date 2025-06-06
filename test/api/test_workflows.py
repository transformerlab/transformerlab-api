from fastapi.testclient import TestClient
import os
from api import app


def test_workflows_list():
    with TestClient(app) as client:
        resp = client.get("/workflows/list")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list) or isinstance(resp.json(), dict)


def test_workflows_delete():
    with TestClient(app) as client:
        resp = client.get("/workflows/delete/dummy_workflow")
        assert resp.status_code in (200, 404)


def test_workflows_create():
    with TestClient(app) as client:
        resp = client.get("/workflows/create?name=test_workflow")
        assert resp.status_code in (200, 400, 404)


def test_experiment_workflows_list():
    """Test the new experiment workflows list endpoint"""
    with TestClient(app) as client:
        # First create an experiment
        exp_name = f"test_exp_{os.getpid()}"
        exp_resp = client.get(f"/experiment/create?name={exp_name}")
        assert exp_resp.status_code == 200
        exp_id = exp_resp.json()

        # Create a workflow in the experiment
        workflow_resp = client.get(f"/workflows/create?name=test_workflow&experiment_id={exp_id}")
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
        exp_name = f"test_exp_runs_{os.getpid()}"
        exp_resp = client.get(f"/experiment/create?name={exp_name}")
        assert exp_resp.status_code == 200
        exp_id = exp_resp.json()

        # Create a workflow in the experiment
        workflow_resp = client.get(f"/workflows/create?name=test_workflow&experiment_id={exp_id}")
        assert workflow_resp.status_code == 200
        workflow_id = workflow_resp.json()

        # Queue the workflow to create a run
        queue_resp = client.get(f"/workflows/{workflow_id}/start")
        assert queue_resp.status_code == 200

        # Test the new experiment workflow runs endpoint
        resp = client.get(f"/experiment/{exp_id}/workflows/runs")
        assert resp.status_code == 200
        runs = resp.json()
        assert isinstance(runs, list)
        assert len(runs) > 0
        assert runs[0]["experiment_id"] == exp_id
        assert runs[0]["workflow_id"] == workflow_id
