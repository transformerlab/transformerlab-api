from fastapi import APIRouter, BackgroundTasks
from transformerlab.shared import galleries
import transformerlab.db as db
from transformerlab.models import model_helper
import json

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
    experiment_id = await db.experiment_create(name=experiment_name, config="{}")

    # Get the recipe
    recipes_gallery = galleries.get_exp_recipe_gallery()
    recipe = next((r for r in recipes_gallery if r.get("id") == id), None)
    if not recipe:
        return {"status": "error", "message": f"Recipe with id {id} not found.", "data": {}}

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
                    workflow_id = await workflows_router.workflow_create(
                        name=dep_name,
                        config=json.dumps(workflow_config),
                        experiment_id=experiment_id,
                    )
                    result["status"] = f"success: {workflow_id}"
                except Exception as e:
                    result["status"] = f"error: {str(e)}"
            else:
                result["status"] = "error: config not provided"
            workflow_results.append(result)

    return {
        "status": "success",
        "message": "",
        "data": {
            "experiment_id": experiment_id,
            "name": experiment_name,
            "model_set_result": model_set_result,
            "workflow_results": workflow_results,
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
