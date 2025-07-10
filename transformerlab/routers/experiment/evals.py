import asyncio
import json
import os
import subprocess
import sys
from typing import Any

from transformerlab.db.db import experiment_get, experiment_update_config
from fastapi import APIRouter, Body
from fastapi.responses import FileResponse
from transformerlab.db.jobs import job_get
from transformerlab.shared import dirs

from werkzeug.utils import secure_filename

router = APIRouter(prefix="/evals", tags=["evals"])


@router.post("/add")
async def experiment_add_evaluation(experimentId: int, plugin: Any = Body()):
    """Add an evaluation to an experiment. This will create a new directory in the experiment
    and add global plugin to the specific experiment. By copying the plugin to the experiment
    directory, we can modify the plugin code for the specific experiment without affecting
    other experiments that use the same plugin."""

    experiment = await experiment_get(experimentId)

    if experiment is None:
        return {"message": f"Experiment {experimentId} does not exist"}

    experiment_config = json.loads(experiment["config"])

    if "evaluations" not in experiment_config:
        experiment_config["evaluations"] = "[]"

    evaluations = json.loads(experiment_config["evaluations"])

    name = plugin["name"]
    plugin_name = plugin["plugin"]
    script_parameters = plugin["script_parameters"]

    # slug = shared.slugify(name)

    # If name is greater than 100 characters, truncate it
    # if len(slug) > 100:
    #     slug = slug[:100]
    #     print("Evals name is too long, truncating to 100 characters")

    evaluation = {"name": name, "plugin": plugin_name, "script_parameters": script_parameters}

    evaluations.append(evaluation)

    await experiment_update_config(experimentId, "evaluations", json.dumps(evaluations))

    return {"message": f"Experiment {experimentId} updated with plugin {plugin_name}"}


@router.get("/delete")
async def experiment_delete_eval(experimentId: int, eval_name: str):
    """Delete an evaluation from an experiment. This will delete the directory in the experiment
    and remove the global plugin from the specific experiment."""
    experiment = await experiment_get(experimentId)

    if experiment is None:
        return {"message": f"Experiment {experimentId} does not exist"}

    experiment_config = json.loads(experiment["config"])

    if "evaluations" not in experiment_config:
        return {"message": f"Experiment {experimentId} has no evaluations"}

    evaluations = json.loads(experiment_config["evaluations"])

    # remove the evaluation from the list:
    evaluations = [e for e in evaluations if e["name"] != eval_name]

    await experiment_update_config(experimentId, "evaluations", json.dumps(evaluations))

    return {"message": f"Evaluation {eval_name} deleted from experiment {experimentId}"}


# @TODO delete the following function and use the plugin file function


@router.post("/edit")
async def edit_evaluation_task(experimentId: int, plugin: Any = Body()):
    """Get the contents of the evaluation"""
    try:
        experiment = await experiment_get(experimentId)

        # if the experiment does not exist, return an error:
        if experiment is None:
            return {"message": f"Experiment {experimentId} does not exist"}

        eval_name = plugin["evalName"]
        updated_json = plugin["script_parameters"]

        plugin_name = updated_json["plugin_name"]
        template_name = updated_json["template_name"]

        experiment_config = json.loads(experiment["config"])

        # updated_json = json.loads(updated_json)

        if "evaluations" not in experiment_config:
            return {"message": f"Experiment {experimentId} has no evaluations"}

        evaluations = json.loads(experiment_config["evaluations"])

        # Remove fields model_name, model_architecture and plugin_name from the updated_json
        # as they are not needed in the evaluations list
        # updated_json.pop("model_name", None)
        # updated_json.pop("model_architecture", None)
        # updated_json.pop("plugin_name", None)
        # updated_json.pop("template_name", None)

        for evaluation in evaluations:
            if evaluation["name"] == eval_name and evaluation["plugin"] == plugin_name:
                evaluation["script_parameters"] = updated_json
                evaluation["name"] = template_name

        await experiment_update_config(experimentId, "evaluations", json.dumps(evaluations))

        return {"message": "OK"}
    except Exception as e:
        print("Error in edit_evaluation_task", e)
        raise e


@router.get("/get_evaluation_plugin_file_contents")
async def get_evaluation_plugin_file_contents(experimentId: int, plugin_name: str):
    # first get the experiment name:
    data = await experiment_get(experimentId)

    # if the experiment does not exist, return an error:
    if data is None:
        return {"message": f"Experiment {experimentId} does not exist"}

    # experiment_name = data["name"]

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
async def run_evaluation_script(experimentId: int, plugin_name: str, eval_name: str, job_id: str):
    job_config = (await job_get(job_id))["job_data"]
    eval_config = job_config.get("config", {})
    print(eval_config)
    experiment_details = await experiment_get(id=experimentId)

    if experiment_details is None:
        return {"message": f"Experiment {experimentId} does not exist"}
    config = json.loads(experiment_details["config"])

    experiment_name = experiment_details["name"]
    model_name = config["foundation"]
    if "model_name" in eval_config.keys():
        model_name = eval_config["model_name"]

    if config["foundation_filename"] is None or config["foundation_filename"].strip() == "":
        model_file_path = ""
    else:
        model_file_path = config["foundation_filename"]
    model_type = config["foundation_model_architecture"]
    if "model_architecture" in eval_config.keys():
        model_type = eval_config["model_architecture"]

    model_adapter = config.get("adaptor_name", "")
    if "adaptor_name" in eval_config.keys():
        model_adapter = eval_config["adaptor_name"]

    # @TODO: This whole thing can be re-written to use the shared function to run a plugin

    # Create the input file for the script:
    input_file = dirs.TEMP_DIR + "/plugin_input_" + str(plugin_name) + ".json"

    # The following two ifs convert nested JSON strings to JSON objects -- this is a hack
    # and should be done in the API itself
    if "config" in experiment_details:
        experiment_details["config"] = json.loads(experiment_details["config"])
        if "inferenceParams" in experiment_details["config"]:
            experiment_details["config"]["inferenceParams"] = json.loads(
                experiment_details["config"]["inferenceParams"]
            )
        if "evaluations" in experiment_details["config"]:
            experiment_details["config"]["evaluations"] = json.loads(experiment_details["config"]["evaluations"])

    template_config = eval_config["script_parameters"]
    job_output_file = await get_job_output_file_name(job_id, plugin_name)

    input_contents = {"experiment": experiment_details, "config": template_config}
    with open(input_file, "w") as outfile:
        json.dump(input_contents, outfile, indent=4)

    # For now, even though we have the file above, we are also going to pass all params
    # as command line arguments to the script.

    # Create a list of all the parameters:
    script_directory = dirs.plugin_dir_by_name(plugin_name)
    extra_args = ["--plugin_dir", script_directory]
    for key in template_config:
        extra_args.append("--" + key)
        extra_args.append(template_config[key])

    # print(template_config)

    extra_args.extend(
        [
            "--experiment_name",
            experiment_name,
            "--eval_name",
            eval_name,
            "--input_file",
            input_file,
            "--model_name",
            model_name,
            "--model_path",
            model_file_path,
            "--model_architecture",
            model_type,
            "--model_adapter",
            model_adapter,
            "--job_id",
            str(job_id),
        ]
    )

    # Check if plugin has a venv directory
    venv_path = os.path.join(script_directory, "venv")
    if os.path.exists(venv_path) and os.path.isdir(venv_path):
        print(f">Plugin has virtual environment, activating venv from {venv_path}")
        # Use bash to activate venv and then run the command
        venv_python = os.path.join(venv_path, "bin", "python")
        # Construct command that first activates venv then runs script
        subprocess_command = [venv_python, dirs.PLUGIN_HARNESS] + extra_args
    else:
        print(">Using system Python interpreter")
        subprocess_command = [sys.executable, dirs.PLUGIN_HARNESS] + extra_args

    print(f">Running {subprocess_command}")

    output_file = await dirs.eval_output_file(experiment_name, eval_name)
    print(f">EVAL Output file: {job_output_file}")

    with open(job_output_file, "w") as f:
        process = await asyncio.create_subprocess_exec(*subprocess_command, stdout=f, stderr=subprocess.PIPE)
        await process.communicate()

    with open(output_file, "w") as f:
        # Copy all contents from job_output_file to output_file
        with open(job_output_file, "r") as job_output:
            for line in job_output:
                f.write(line)


async def get_job_output_file_name(job_id: str, plugin_name: str):
    try:
        job_id = secure_filename(str(job_id))
        plugin_name = secure_filename(plugin_name)

        plugin_dir = dirs.plugin_dir_by_name(plugin_name)
        jobs_dir_output_file_name = os.path.join(dirs.WORKSPACE_DIR, "jobs", str(job_id))

        # job output is stored in separate files with a job number in the name...
        if os.path.exists(os.path.join(jobs_dir_output_file_name, f"output_{job_id}.txt")):
            output_file = os.path.join(jobs_dir_output_file_name, f"output_{job_id}.txt")
        elif os.path.exists(os.path.join(plugin_dir, f"output_{job_id}.txt")):
            output_file = os.path.join(plugin_dir, f"output_{job_id}.txt")
        # but it used to be all stored in a single file called output.txt, so check that as well
        elif os.path.exists(os.path.join(plugin_dir, "output.txt")):
            output_file = os.path.join(plugin_dir, "output.txt")
        else:
            raise ValueError(f"No output file found for job {job_id}")
        return output_file
    except Exception as e:
        raise e


@router.get("/get_output")
async def get_output(experimentId: int, eval_name: str):
    """Get the output of an evaluation"""
    eval_name = secure_filename(eval_name)  # sanitize the input
    data = await experiment_get(experimentId)
    # if the experiment does not exist, return an error:
    if data is None:
        return {"message": f"Experiment {experimentId} does not exist"}

    experiment_name = data["name"]

    eval_output_file = await dirs.eval_output_file(experiment_name, eval_name)
    if not os.path.exists(eval_output_file):
        return {"message": "Output file does not exist"}

    print(f"Returning output file: {eval_output_file}.")

    # return the whole file as a file response:
    return FileResponse(eval_output_file)
