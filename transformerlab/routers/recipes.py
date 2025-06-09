from fastapi import APIRouter, BackgroundTasks
from transformerlab.shared import galleries
import transformerlab.db as db
from transformerlab.models import model_helper
import json
from transformerlab.routers import (
    data,
    model,
    serverinfo,
    train,
    plugins,
    evals,
    config,
    jobs,
    tasks,
    prompts,
    tools,
    batched_prompts,
    recipes,
    users,
)
from transformerlab.routers.experiment import workflows as experiment_workflows

router = APIRouter(prefix="/recipes", tags=["recipes"])


@router.get("/list")
async def list_recipes():
    """List all recipes for a given experiment name."""
    recipes_gallery = galleries.get_exp_recipe_gallery()
    return recipes_gallery


@router.get("/{id}")
async def get_recipe_by_id(id: int):
    """Fetch a recipe by its ID from the experiment recipe gallery."""
    recipes_gallery = galleries.get_exp_recipe_gallery()
    for recipe in recipes_gallery:
        if recipe.get("id") == id:
            return recipe
    return {"error": f"Recipe with id {id} not found."}


@router.get("/{id}/check_dependencies")
async def check_recipe_dependencies(id: int):
    """Check if the dependencies for a recipe are installed for a given environment."""
    # Get the recipe
    recipes_gallery = galleries.get_exp_recipe_gallery()
    recipe = next((r for r in recipes_gallery if r.get("id") == id), None)
    if not recipe:
        return {"error": f"Recipe with id {id} not found."}

    if len(recipe.get("dependencies", [])) == 0:
        return {"dependencies": []}

    # Get local models and datasets
    local_models = await model_helper.list_installed_models()
    local_model_names = set(model["model_id"] for model in local_models)
    local_datasets = await db.get_datasets()
    local_dataset_ids = set(ds["dataset_id"] for ds in local_datasets)

    # Get installed plugins using the same logic as /plugins/gallery
    from transformerlab.routers import plugins as plugins_router

    plugin_gallery = await plugins_router.plugin_gallery()
    installed_plugins = set(p["uniqueId"] for p in plugin_gallery if p.get("installed"))

    results = []
    for dep in recipe.get("dependencies", []):
        dep_type = dep.get("type")
        dep_name = dep.get("name")
        if dep_type == "workflow":
            # Skip workflow installation in this background job
            continue
        status = {"type": dep_type, "name": dep_name, "installed": False}
        if dep_type == "model":
            status["installed"] = dep_name in local_model_names
        elif dep_type == "dataset":
            # Check if dataset is installed
            status["installed"] = dep_name in local_dataset_ids
        elif dep_type == "plugin":
            status["installed"] = dep_name in installed_plugins
        results.append(status)
    return {"dependencies": results}


async def _install_recipe_dependencies_job(job_id, id):
    from transformerlab.routers import model as model_router
    from transformerlab.routers import data as data_router
    from transformerlab.routers import plugins as plugins_router

    try:
        await db.job_update_status(job_id, "RUNNING")
        recipes_gallery = galleries.get_exp_recipe_gallery()
        recipe = next((r for r in recipes_gallery if r.get("id") == id), None)
        if not recipe:
            await db.job_update_status(job_id, "FAILED", error_msg=f"Recipe with id {id} not found.")
            return
        if len(recipe.get("dependencies", [])) == 0:
            await db.job_update_job_data_insert_key_value(job_id, "results", [])
            await db.job_update_status(job_id, "COMPLETE")
            return

        local_models = await model_helper.list_installed_models()
        local_model_names = set(model["model_id"] for model in local_models)
        local_datasets = await db.get_datasets()
        local_dataset_ids = set(ds["dataset_id"] for ds in local_datasets)
        total = len(recipe.get("dependencies", []))
        progress = 0
        results = []
        for dep in recipe.get("dependencies", []):
            dep_type = dep.get("type")
            dep_name = dep.get("name")
            if dep_type == "workflow":
                # Skip workflow installation in this background job
                continue
            result = {"type": dep_type, "name": dep_name, "action": None, "status": None}
            try:
                if dep_type == "model":
                    if dep_name not in local_model_names:
                        download_result = await model_router.download_model_by_huggingface_id(model=dep_name)
                        result["action"] = "download_model"
                        result["status"] = download_result.get("status", "unknown")
                    else:
                        result["action"] = "already_installed"
                        result["status"] = "success"
                elif dep_type == "dataset":
                    if dep_name not in local_dataset_ids:
                        download_result = await data_router.dataset_download(dataset_id=dep_name)
                        result["action"] = "download_dataset"
                        result["status"] = download_result.get("status", "unknown")
                    else:
                        result["action"] = "already_installed"
                        result["status"] = "success"
                elif dep_type == "plugin":
                    install_result = await plugins_router.install_plugin(plugin_id=dep_name)
                    result["action"] = "install_plugin"
                    result["status"] = install_result.get("status", "unknown")
            except Exception as e:
                result["action"] = "error"
                result["status"] = str(e)
            results.append(result)
            progress += 1
            await db.job_update_progress(job_id, int(progress * 100 / total))
            await db.job_update_job_data_insert_key_value(job_id, "results", results)
        await db.job_update_status(job_id, "COMPLETE")
    except Exception as e:
        await db.job_update_status(job_id, "FAILED", error_msg=str(e))


@router.get("/{id}/install_dependencies")
async def bg_install_recipe_dependencies(id: int, background_tasks: BackgroundTasks):
    """Install dependencies for a recipe in the background and track progress."""

    job_id = await db.job_create(
        type="INSTALL_RECIPE_DEPS",
        status="QUEUED",
        job_data=json.dumps({"recipe_id": id, "results": [], "progress": 0}),
        experiment_id="",
    )
    # Start background task
    background_tasks.add_task(_install_recipe_dependencies_job, job_id, id)
    return {"job_id": job_id, "status": "started"}


@router.get("/jobs/{job_id}/status")
async def get_install_job_status(job_id: int):
    """Get the status and progress of a dependency installation job."""
    job = await db.job_get(job_id)
    if not job:
        return {"error": f"Job {job_id} not found."}
    return {
        "job_id": job_id,
        "status": job["status"],
        "progress": job.get("progress", 0),
        "results": job["job_data"].get("results", []),
        "error_msg": job["job_data"].get("error_msg"),
    }


@router.post("/{id}/create_experiment")
async def create_experiment_for_recipe(id: int, experiment_name: str):
    """Create a new experiment with the given name and blank config, and install workflow dependencies."""
    from transformerlab.routers import workflows as workflows_router
    from transformerlab.routers.experiment import experiment as experiment_router

    # Check if experiment already exists
    existing = await db.experiment_get_by_name(experiment_name)
    if existing:
        return {"status": "error", "message": f"Experiment '{experiment_name}' already exists.", "data": {}}
    # Create experiment with blank config
    experiment_id = await db.experiment_create(name=experiment_name, config={})

    # Get the recipe
    recipes_gallery = galleries.get_exp_recipe_gallery()
    recipe = next((r for r in recipes_gallery if r.get("id") == id), None)
    if not recipe:
        return {"status": "error", "message": f"Recipe with id {id} not found.", "data": {}}

    # Populate Notes file if recipe contains notes
    notes_result = None
    if recipe.get("notes"):
        try:
            # Use the experiment router's save_file_contents function to create the Notes file
            notes_result = await experiment_router.experiment_save_file_contents(
                id=experiment_id, filename="readme.md", file_contents=recipe.get("notes")
            )
        except Exception:
            notes_result = {"error": "Failed to create Notes file."}

    # Set foundation model if present in dependencies
    model_set_result = None
    local_models = await model_helper.list_installed_models()
    local_model_dict = {m["model_id"]: m for m in local_models}
    for dep in recipe.get("dependencies", []):
        if dep.get("type") == "model":
            model_id = dep.get("name")
            model = local_model_dict.get(model_id)
            # Check if the model is installed
            if not model:
                model_set_result = {"error": f"Model '{model_id}' not found in local models."}
                break
            model_name = model.get("model_id", "")
            model_filename = ""
            if model.get("stored_in_filesystem"):
                model_filename = model.get("local_path", "")
            elif model.get("json_data", {}).get("model_filename"):
                model_filename = model["json_data"]["model_filename"]
            architecture = model.get("json_data", {}).get("architecture", "")

            # Update experiment config fields using the experiment update_config route
            await experiment_router.experiments_update_config(experiment_id, "foundation", model_name)
            await experiment_router.experiments_update_config(
                experiment_id, "foundation_model_architecture", architecture
            )
            await experiment_router.experiments_update_config(experiment_id, "foundation_filename", model_filename)
            model_set_result = {
                "foundation": model_name,
                "foundation_model_architecture": architecture,
                "foundation_filename": model_filename,
            }
            break  # Only set the first model dependency

    workflow_results = []
    for dep in recipe.get("dependencies", []):
        if dep.get("type") == "workflow":
            workflow_config = dep.get("config")
            dep_name = dep.get("name")
            result = {"name": dep_name, "action": "install_workflow"}
            if workflow_config is not None:
                try:
                    workflow_id = await experiment_workflows.workflow_create(
                        name=dep_name,
                        config=json.dumps(workflow_config),
                        experimentId=experiment_id,
                    )
                    result["status"] = f"success: {workflow_id}"
                except Exception as e:
                    result["status"] = f"error: {str(e)}"
            else:
                result["status"] = "error: config not provided"
            workflow_results.append(result)

    # Process tasks and create tasks in database
    task_results = []
    tasks = recipe.get("tasks", [])

    # Extract dataset from dependencies (assuming only one dataset)
    dataset_name = ""
    dataset_deps = [dep for dep in recipe.get("dependencies", []) if dep.get("type") == "dataset"]
    if dataset_deps:
        dataset_name = dataset_deps[0].get("name", "")

    for i, task in enumerate(tasks):
        task_type = task.get("task_type")
        if task_type in ["TRAIN", "EVAL", "GENERATE"]:
            try:
                # Parse the config_json to extract template metadata
                config_json = task.get("config_json", "{}")
                parsed_config = json.loads(config_json)

                # Convert any lists or dicts in the config to JSON strings
                for key, value in parsed_config.items():
                    if key != "script_parameters" and isinstance(value, (list, dict)):
                        parsed_config[key] = json.dumps(value)

                # Convert list/dict values inside script_parameters to strings
                if "script_parameters" in parsed_config and isinstance(parsed_config["script_parameters"], dict):
                    for param_key, param_value in parsed_config["script_parameters"].items():
                        if isinstance(param_value, (list, dict)):
                            parsed_config["script_parameters"][param_key] = json.dumps(param_value)

                # Extract task name from recipe
                task_name = task.get("name")
                
                # Create inputs JSON (what the task needs as inputs)
                inputs = {
                    "model_name": parsed_config.get("model_name", ""),
                    "model_architecture": parsed_config.get("model_architecture", ""),
                    "dataset_name": dataset_name,  # Using dataset from dependencies
                }

                # For EVAL tasks, add evaluation specific inputs
                if task_type == "EVAL":
                    inputs.update(
                        {
                            "tasks": parsed_config.get("tasks", ""),
                            "limit": parsed_config.get("limit", ""),
                            "run_name": parsed_config.get("run_name", ""),
                        }
                    )
                # For GENERATE tasks, add generation specific inputs
                elif task_type == "GENERATE":
                    inputs.update(
                        {
                            "num_goldens": parsed_config.get("num_goldens", ""),
                            "scenario": parsed_config.get("scenario", ""),
                            "task": parsed_config.get("task", ""),
                            "run_name": parsed_config.get("run_name", ""),
                        }
                    )

                # Create outputs JSON (what the task produces)
                outputs = {}

                if task_type == "EVAL":
                    outputs["eval_results"] = "{}"
                elif task_type == "GENERATE":
                    outputs["generated_outputs"] = "[]"

                # Get plugin name
                plugin_name = parsed_config.get("plugin_name", "")

                # Create task in database
                await db.add_task(
                    name=task_name,
                    Type=task_type,
                    inputs=json.dumps(inputs),
                    config=json.dumps(parsed_config),
                    plugin=plugin_name,
                    outputs=json.dumps(outputs),
                    experiment_id=experiment_id,
                )

                task_results.append(
                    {
                        "task_index": i + 1,
                        "task_name": task_name,
                        "action": "create_task",
                        "status": "success",
                        "task_type": task_type,
                        "dataset_used": dataset_name,
                        "plugin": plugin_name,
                    }
                )

            except Exception:
                task_results.append(
                    {
                        "task_index": i + 1,
                        "action": "create_task",
                        "status": f"error: Failed to create {task_type.lower()} task.",
                    }
                )

    # Process workflows and create workflows in database
    workflow_creation_results = []
    workflows = recipe.get("workflows", [])
    
    for workflow_def in workflows:
        try:
            workflow_name = workflow_def.get("name", "")
            workflow_config = workflow_def.get("config", {"nodes": []})
            
            # Create workflow in database using the workflow_create function
            workflow_id = await experiment_workflows.workflow_create(
                name=workflow_name,
                config=json.dumps(workflow_config),
                experimentId=experiment_id
            )
            
            # Log the workflow creation results
            workflow_creation_results.append({
                "workflow_name": workflow_name,
                "action": "create_workflow",
                "status": "success",
                "workflow_id": workflow_id
            })
            
        except Exception:
            workflow_creation_results.append({
                "workflow_name": workflow_def.get("name", "Unknown"),
                "action": "create_workflow",
                "status": "error: Failed to create workflow."
            })

    return {
        "status": "success",
        "message": "",
        "data": {
            "experiment_id": experiment_id,
            "name": experiment_name,
            "model_set_result": model_set_result,
            "workflow_results": workflow_results,
            "task_results": task_results,
            "workflow_creation_results": workflow_creation_results,
            "notes_result": notes_result,
        },
    }


## OLDER CODE WITHOUT A JOB SYSTEM

# @router.get("/{id}/install_dependencies")
# async def install_recipe_dependencies(id: int):
#     """Install model and dataset dependencies for a recipe."""
#     from transformerlab.routers import model as model_router
#     from transformerlab.routers import data as data_router

#     # Get the recipe
#     recipes_gallery = galleries.get_exp_recipe_gallery()
#     recipe = next((r for r in recipes_gallery if r.get("id") == id), None)
#     if not recipe:
#         return {"error": f"Recipe with id {id} not found."}

#     if len(recipe.get("dependencies", [])) == 0:
#         return {"results": []}

#     # Get local models and datasets
#     local_models = await model_helper.list_installed_models()
#     local_model_names = set(model["model_id"] for model in local_models)
#     local_datasets = await db.get_datasets()
#     local_dataset_ids = set(ds["dataset_id"] for ds in local_datasets)

#     install_results = []
#     for dep in recipe.get("dependencies", []):
#         dep_type = dep.get("type")
#         dep_name = dep.get("name")
#         if dep_type == "workflow":
#             # Skip workflow installation in this background job
#             continue
#         result = {"type": dep_type, "name": dep_name, "action": None, "status": None}
#         if dep_type == "model":
#             if dep_name not in local_model_names:
#                 download_result = await model_router.download_model_by_huggingface_id(model=dep_name)
#                 result["action"] = "download_model"
#                 result["status"] = download_result.get("status", "unknown")
#             else:
#                 result["action"] = "already_installed"
#                 result["status"] = "success"
#         elif dep_type == "dataset":
#             if dep_name not in local_dataset_ids:
#                 download_result = await data_router.dataset_download(dataset_id=dep_name)
#                 result["action"] = "download_dataset"
#                 result["status"] = download_result.get("status", "unknown")
#             else:
#                 result["action"] = "already_installed"
#                 result["status"] = "success"
#         elif dep_type == "plugin":
#             from transformerlab.routers import plugins as plugins_router

#             install_result = await plugins_router.install_plugin(plugin_id=dep_name)
#             result["action"] = "install_plugin"
#             result["status"] = install_result.get("status", "unknown")
#         install_results.append(result)
#     return {"results": install_results}
