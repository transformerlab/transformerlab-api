from xmlrpc.server import SimpleXMLRPCDispatcher
from fastapi import APIRouter, Request, Response
from fastapi.responses import PlainTextResponse
import time
import os
from transformerlab.shared import dirs
import shutil


class XMLRPCRouter:
    """
    A router for FastAPI that handles XML-RPC requests.
    """

    def __init__(self, prefix="/rpc"):
        self.dispatcher = SimpleXMLRPCDispatcher(allow_none=True, encoding=None)
        self.router = APIRouter(prefix=prefix)

        # Register the POST endpoint to handle XML-RPC requests
        self.router.add_api_route("", self.handle_xmlrpc, methods=["POST"], response_class=PlainTextResponse)

    def register_function(self, function, name=None):
        """
        Register a function to be exposed via XML-RPC.

        Args:
            function: The function to register
            name: The name to register it under (defaults to the function name)
        """
        return self.dispatcher.register_function(function, name)

    def register_instance(self, instance, allow_dotted_names=False):
        """
        Register an instance to be exposed via XML-RPC.

        Args:
            instance: The instance to register
            allow_dotted_names: Whether to allow dotted names for methods
        """
        return self.dispatcher.register_instance(instance, allow_dotted_names)

    def register_introspection_functions(self):
        """
        Register the introspection functions.
        """
        return self.dispatcher.register_introspection_functions()

    def register_multicall_functions(self):
        """
        Register the multicall functions.
        """
        return self.dispatcher.register_multicall_functions()

    async def handle_xmlrpc(self, request: Request) -> Response:
        """
        Handle an XML-RPC request.

        Args:
            request: The FastAPI request object

        Returns:
            A PlainTextResponse with the XML-RPC response
        """
        # Read the request body
        request_body = await request.body()

        # Process the request and get the response
        response = self.dispatcher._marshaled_dispatch(request_body, getattr(request, "_dispatch_method", None))

        # Return the response with the correct content type
        return Response(content=response, media_type="text/xml")


# Add this function to create and return the configured XML-RPC router
def get_xmlrpc_router(prefix="/job_sdk"):
    """
    Create and return a configured XML-RPC router that can be included in a FastAPI app.

    Args:
        prefix: The URL prefix for the XML-RPC endpoint

    Returns:
        A configured FastAPI router instance
    """
    # Create a new XML-RPC router
    xmlrpc_router = XMLRPCRouter(prefix=prefix)

    # Example returning a string
    def hello(name="World"):
        return f"Hello, {name}!"

    # Example returning a dictionary
    def get_user(user_id):
        # This could fetch from a database in a real application
        users = {
            1: {"name": "Alice", "email": "alice@example.com", "active": True},
            2: {"name": "Bob", "email": "bob@example.com", "active": False},
        }
        return users.get(user_id, {"error": "User not found"})

    # Example returning a list of objects
    def list_users():
        return [
            {"id": 1, "name": "Alice", "email": "alice@example.com"},
            {"id": 2, "name": "Bob", "email": "bob@example.com"},
            {"id": 3, "name": "Charlie", "email": "charlie@example.com"},
        ]

    # Example returning nested complex data
    def get_project_data(project_id):
        if project_id == 123:
            return {
                "id": project_id,
                "name": "AI Model Training",
                "status": "running",
                "progress": 0.75,
                "tasks": [
                    {"id": 1, "name": "Data preprocessing", "complete": True},
                    {"id": 2, "name": "Model training", "complete": False},
                ],
                "metrics": {"accuracy": 0.92, "loss": 0.08},
                "created_at": "2025-04-01T12:00:00Z",
            }

        return {"error": "Project not found"}

    # Register all our functions
    xmlrpc_router.register_function(hello)
    xmlrpc_router.register_function(get_user)
    xmlrpc_router.register_function(list_users)
    xmlrpc_router.register_function(get_project_data)

    # Enable introspection (optional)
    xmlrpc_router.register_introspection_functions()

    # Return the router property
    return xmlrpc_router.router


def get_trainer_xmlrpc_router(prefix="/trainer_rpc", trainer_factory=None):
    """
    Create and return a configured XML-RPC router for the TLab trainer.

    Args:
        prefix: The URL prefix for the XML-RPC endpoint
        trainer_instance: Instance of TrainerTLabPlugin to expose via RPC

    Returns:
        A configured FastAPI router instance
    """
    import json
    from transformerlab.db import job_create_sync, job_update_status_sync
    import transformerlab.plugin_sdk.transformerlab.plugin as tlab_core

    # # Import the trainer if not provided
    # if trainer_instance is None:
    #     from transformerlab.plugin_sdk.transformerlab.sdk.v1.train import tlab_trainer

    #     trainer_instance = tlab_trainer

    if trainer_factory is None:
        # Define a factory function that returns a fresh trainer instance
        def create_trainer():
            from transformerlab.plugin_sdk.transformerlab.sdk.v1.train import tlab_trainer

            return tlab_trainer.__class__()  # Create a new instance of the same class

        trainer_factory = create_trainer

    # Create a new XML-RPC router
    xmlrpc_router = XMLRPCRouter(prefix=prefix)

    # Dictionary to store job_id -> trainer_instance mapping
    job_trainers = {}

    # Expose trainer methods via RPC-friendly wrappers
    def start_training(config_json):
        """Start a training job with the given configuration"""
        try:
            # Parse the JSON config
            config = json.loads(config_json) if isinstance(config_json, str) else config_json

            experiment_name = config.get("experiment_name", "alpha")
            experiment_id = tlab_core.get_experiment_id_from_name(experiment_name)

            # Set up the trainer parameters
            job_id = job_create_sync("TRAIN", "RUNNING", job_data=json.dumps(config), experiment_id=str(experiment_id))

            trainer_instance = trainer_factory()
            job_trainers[job_id] = trainer_instance

            trainer_instance.params["job_id"] = job_id
            trainer_instance.params["experiment_id"] = experiment_id
            trainer_instance.params["experiment_name"] = experiment_name
            for key, value in config.items():
                trainer_instance.params[key] = value
            trainer_instance._args_parsed = True
            trainer_instance.params.reported_metrics = []

            train_logging_dir = os.path.join(
                tlab_core.WORKSPACE_DIR,
                "experiments",
                experiment_name,
                "tensorboards",
                trainer_instance.params["template_name"],
            )

            trainer_instance.setup_train_logging(output_dir=train_logging_dir)

            # Initialize the job
            job = trainer_instance.job
            start_time = time.strftime("%Y-%m-%d %H:%M:%S")
            job.add_to_job_data("start_time", start_time)
            # job_update_status_sync(job_id, "RUNNING")
            job.update_progress(0)

            # Return success with job ID
            return {"status": "started", "job_id": job_id}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def get_training_status(job_id, progress_update):
        """Get the status of a training job"""
        try:
            trainer_instance = job_trainers.get(job_id)
            if not trainer_instance:
                # Fall back to creating a new one with just the job_id set
                trainer_instance = trainer_factory()
                trainer_instance.params["job_id"] = job_id

            job = trainer_instance._job

            # Get status and progress
            status = job.get_status()
            progress = job.get_progress()

            job.update_progress(progress_update)

            # Get any job data
            job_data = job.get_job_data()

            return {"job_id": job_id, "status": status, "progress": progress, "data": job_data}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def stop_training(job_id):
        """Stop a training job"""
        try:
            # job = trainer_instance.job
            job_update_status_sync(job_id, "STOPPED")
            return {"status": "stopping", "job_id": job_id}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def log_metrics(job_id, metrics_json):
        """Log metrics for a training job"""
        try:
            metrics = json.loads(metrics_json) if isinstance(metrics_json, str) else metrics_json
            # Store metrics in job data or a separate metrics store
            # trainer_instance.params["job_id"] = job_id
            # job = trainer_instance._job

            # # Get existing job data and update with metrics
            # job_data = job.get_job_data() or {}
            # if isinstance(job_data, str):
            #     job_data = json.loads(job_data)

            # if "metrics" not in job_data:
            #     job_data["metrics"] = []

            trainer_instance = job_trainers.get(job_id)
            if not trainer_instance:
                # Fall back to creating a new one with just the job_id set
                trainer_instance = trainer_factory()
                trainer_instance.params["job_id"] = job_id

            trainer_instance.params.reported_metrics.append(metrics)

            if "step" in metrics.keys():
                step = metrics["step"]
                # Do trainer_instance.log_metrics(metrics) for all keys except "step"
                for key, value in metrics.items():
                    if key != "step":
                        trainer_instance.log_metric(key, value, step)

            # Update job data
            # job.update_job_data(json.dumps(job_data))

            return {"status": "success", "job_id": job_id}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def complete_job(job_id, status="COMPLETE", message="Training completed successfully"):
        """Mark a training job as complete"""
        try:
            trainer_instance = job_trainers.get(job_id)
            if not trainer_instance:
                # Fall back to creating a new one with just the job_id set
                trainer_instance = trainer_factory()
                trainer_instance.params["job_id"] = job_id

            job = trainer_instance._job

            # Update job status
            job_update_status_sync(job_id, status)

            # Update job data with completion message
            job_data = job.get_job_data() or {}
            if isinstance(job_data, str):
                job_data = json.loads(job_data)

            job.add_to_job_data("metrics", str(trainer_instance.params.reported_metrics))

            # job_data["completion_message"] = message
            # job_data["completed_at"] = datetime.now().isoformat()
            end_time = time.strftime("%Y-%m-%d %H:%M:%S")
            job.add_to_job_data("end_time", end_time)

            # Update job data
            # job.update_job_data(json.dumps(job_data))
            job.update_progress(100)  # Ensure progress is set to 100%
            job.set_job_completion_status("success", message)

            return {"status": "success", "job_id": job_id}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def save_model(job_id, local_model_path):
        """Save the model to the specified path"""
        try:
            print("REQUEST COMES")
            trainer_instance = job_trainers.get(job_id)
            if not trainer_instance:
                # Fall back to creating a new one with just the job_id set
                trainer_instance = trainer_factory()
                trainer_instance.params["job_id"] = job_id

            models_dir = dirs.MODELS_DIR
            # Check if local_model_path is a directory
            model_save_path = os.path.join(models_dir, f"{trainer_instance.params['template_name']}_{job_id}")
            if os.path.isdir(local_model_path):
                # Copy all contents of the directory to the model_save_path
                shutil.copytree(local_model_path, model_save_path)
                # Read config.json
                if os.path.exists(os.path.join(local_model_path, "config.json")):
                    with open(os.path.join(local_model_path, "config.json"), "r") as f:
                        json_data = json.load(f)
                else:
                    raise FileNotFoundError("config.json not found in the specified directory")
                model_architecture = json_data.get("architectures", [None])[0]
                if model_architecture is None:
                    raise ValueError("Model architecture not found in config.json")

                trainer_instance.create_transformerlab_model(
                    f"{trainer_instance.params['template_name']}_{job_id}", model_architecture, json_data
                )
            # url = "http://localhost:8338/model/import_from_local_path"
            # # Check if local_model_path is a directory
            # response = requests.get(url, params = {'model_path': local_model_path})

            # print("RESPONSE", response)

            # if response.status_code == 200:
            # Model saved successfully
            return {"status": "success", "message": "Model saved successfully"}
            # else:
            #     # Failed to save model
            #     return {"status": "error", "message": "Failed to save model"}

        except Exception as e:
            return {"status": "error", "message": str(e)}

    # Register all our functions
    xmlrpc_router.register_function(start_training)
    xmlrpc_router.register_function(get_training_status)
    xmlrpc_router.register_function(stop_training)
    xmlrpc_router.register_function(log_metrics)
    xmlrpc_router.register_function(complete_job)
    xmlrpc_router.register_function(save_model)

    # Enable introspection
    xmlrpc_router.register_introspection_functions()

    # Return the router property
    return xmlrpc_router.router
