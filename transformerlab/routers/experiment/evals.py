from transformerlab.routers.experiment import rag, documents, plugins, conversations, export
from transformerlab.shared import dirs
from transformerlab.shared import shared
import transformerlab.db as db
from fastapi import APIRouter, Body
from typing import Annotated, Any
import time
import sys
import subprocess
from pathlib import Path
import os
import json


router = APIRouter(prefix="/evals", tags=["evals"])


@router.post("/add")
async def experiment_add_evaluation(id: int, plugin: Any = Body()):
    """ Add an evaluation to an experiment. This will create a new directory in the experiment
    and add global plugin to the specific experiment. By copying the plugin to the experiment
    directory, we can modify the plugin code for the specific experiment without affecting
    other experiments that use the same plugin. """

    experiment = await db.experiment_get(id)

    if experiment is None:
        return {"message": f"Experiment {id} does not exist"}

    experiment_config = json.loads(experiment["config"])

    if "evaluations" not in experiment_config:
        experiment_config["evaluations"] = "[]"

    evaluations = json.loads(experiment_config["evaluations"])

    name = plugin["name"]
    plugin_name = plugin["plugin"]
    script_parameters = plugin["script_parameters"]

    slug = shared.slugify(name)

    evaluation = {
        "name": slug,
        "plugin": plugin_name,
        "script_parameters": script_parameters
    }

    evaluations.append(evaluation)

    await db.experiment_update_config(id, "evaluations", json.dumps(evaluations))

    return {"message": f"Experiment {id} updated with plugin {plugin_name}"}


@router.get("/delete")
async def experiment_delete_eval(id: int, eval_name: str):
    """ Delete an evaluation from an experiment. This will delete the directory in the experiment
    and remove the global plugin from the specific experiment. """
    experiment = await db.experiment_get(id)

    if experiment is None:
        return {"message": f"Experiment {id} does not exist"}

    experiment_config = json.loads(experiment["config"])

    if "evaluations" not in experiment_config:
        return {"message": f"Experiment {id} has no evaluations"}

    evaluations = json.loads(experiment_config["evaluations"])

    # remove the evaluation from the list:
    evaluations = [e for e in evaluations if e["name"] != eval_name]

    await db.experiment_update_config(id, "evaluations", json.dumps(evaluations))

    return {"message": f"Evaluation {eval_name} deleted from experiment {id}"}

# @TODO delete the following function and use the plugin file function


@router.get("/get_evaluation_plugin_file_contents")
async def get_evaluation_plugin_file_contents(id: int, plugin_name: str):
    # first get the experiment name:
    data = await db.experiment_get(id)

    # if the experiment does not exist, return an error:
    if data is None:
        return {"message": f"Experiment {id} does not exist"}

    experiment_name = data["name"]

    # print(f"{EXPERIMENTS_DIR}/{experiment_name}/evals/{eval_name}/main.py")

    file_name = "main.py"
    plugin_path = dirs.plugin_dir_by_name(plugin_name)

    # now get the file contents
    try:
        with open(os.path.join(plugin_path, file_name), "r") as f:
            file_contents = f.read()
    except FileNotFoundError:
        return "error file not found"

    return file_contents


@router.get("/run_evaluation_script")
async def run_evaluation_script(id: int, plugin_name: str, eval_name: str):
    experiment_details = await db.experiment_get(id=id)

    if experiment_details is None:
        return {"message": f"Experiment {id} does not exist"}
    config = json.loads(experiment_details["config"])

    experiment_name = experiment_details["name"]
    model_name = config["foundation"]
    model_type = config["foundation_model_architecture"]
    model_adapter = config["adaptor"]

    # @TODO: This whole thing can be re-written to use the shared function to run a plugin

    # Create the input file for the script:
    input_file = dirs.TEMP_DIR + "/plugin_input_" + str(plugin_name) + ".json"

    # The following two ifs convert nested JSON strings to JSON objects -- this is a hack
    # and should be done in the API itself
    if "config" in experiment_details:
        experiment_details["config"] = json.loads(
            experiment_details["config"])
        if "inferenceParams" in experiment_details["config"]:
            experiment_details["config"]["inferenceParams"] = json.loads(
                experiment_details["config"]["inferenceParams"])
        if "evaluations" in experiment_details["config"]:
            experiment_details["config"]["evaluations"] = json.loads(
                experiment_details["config"]["evaluations"])

    all_evaluations = experiment_details["config"]["evaluations"]
    this_evaluation = None
    for evaluation in all_evaluations:
        if evaluation["name"] == eval_name:
            this_evaluation = evaluation
            break

    if this_evaluation is None:
        return {"message": f"Error: evaluation {eval_name} does not exist in experiment"}

    template_config = this_evaluation["script_parameters"]

    input_contents = {"experiment": experiment_details,
                      "config": template_config}
    with open(input_file, 'w') as outfile:
        json.dump(input_contents, outfile, indent=4)

    # For now, even though we have the file above, we are also going to pass all params
    # as command line arguments to the script.

    # Create a list of all the parameters:
    extra_args = []
    for key in template_config:
        extra_args.append("--" + key)
        extra_args.append(template_config[key])

    print(template_config)

    script_directory = dirs.plugin_dir_by_name(plugin_name)
    script_path = f"{script_directory}/main.py"
    extra_args.extend(["--experiment_name", experiment_name, "--eval_name", eval_name, "--input_file", input_file,
                       "--model_name", model_name, "--model_architecture", model_type, "--model_adapter", model_adapter])

    subprocess_command = [sys.executable, script_path] + extra_args

    print(f">Running {subprocess_command}")

    with open(f"{script_directory}/output.txt", "w") as f:
        subprocess.run(args=subprocess_command, stdout=f)
