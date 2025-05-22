from fastapi import APIRouter
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
async def check_recipe_dependencies(id: int, experiment_name: str):
    """Check if the dependencies for a recipe are installed for a given experiment."""
    # Get the recipe
    recipes_gallery = galleries.get_exp_recipe_gallery()
    recipe = next((r for r in recipes_gallery if r.get("id") == id), None)
    if not recipe:
        return {"error": f"Recipe with id {id} not found."}

    if len(recipe.get("dependencies", [])) == 0:
        return {"dependencies": []}

    # Get experiment config
    experiment = await db.experiment_get_by_name(experiment_name)
    if not experiment:
        return {"error": f"Experiment '{experiment_name}' not found."}
    config = json.loads(experiment["config"]) if isinstance(experiment["config"], str) else experiment["config"]

    # Get local models and datasets
    local_models = await model_helper.list_installed_models()
    local_model_names = set(model["model_id"] for model in local_models)
    local_datasets = await db.get_datasets()
    local_dataset_ids = set(ds["dataset_id"] for ds in local_datasets)

    # Get installed plugins using the same logic as /plugins/gallery
    from transformerlab.routers import plugins as plugins_router

    plugin_gallery = await plugins_router.plugin_gallery()
    installed_plugins = set(p["uniqueId"] for p in plugin_gallery if p.get("installed"))

    # Get installed workflows for the experiment
    workflows = await db.workflows_get_from_experiment(experiment["id"])
    installed_workflow_names = set(wf["name"] for wf in workflows if wf.get("name"))

    results = []
    for dep in recipe.get("dependencies", []):
        dep_type = dep.get("type")
        dep_name = dep.get("name")
        status = {"type": dep_type, "name": dep_name, "installed": False}
        if dep_type == "model":
            # Check if the experiment's foundation model matches and is installed
            foundation = config.get("foundation", "")
            status["used_in_experiment"] = foundation == dep_name
            status["installed"] = dep_name in local_model_names
        elif dep_type == "dataset":
            # Check if dataset is installed
            status["installed"] = dep_name in local_dataset_ids
        elif dep_type == "plugin":
            status["installed"] = dep_name in installed_plugins
        elif dep_type == "workflow":
            status["installed"] = dep_name in installed_workflow_names
        results.append(status)
    return {"dependencies": results}
