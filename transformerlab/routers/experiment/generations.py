import asyncio
import json
import os
import subprocess
import sys
import urllib
from typing import Any

import transformerlab.db as db
from fastapi import APIRouter, Body
from fastapi.responses import FileResponse
from transformerlab.shared import dirs, shared

from werkzeug.utils import secure_filename

router = APIRouter(prefix="/generations", tags=["generations"])


@router.post("/add")
async def experiment_add_generation(experimentId: int, plugin: Any = Body()):
    """Add an evaluation to an experiment. This will create a new directory in the experiment
    and add global plugin to the specific experiment. By copying the plugin to the experiment
    directory, we can modify the plugin code for the specific experiment without affecting
    other experiments that use the same plugin."""

    experiment = await db.experiment_get(experimentId)

    if experiment is None:
        return {"message": f"Experiment {experimentId} does not exist"}

    experiment_config = json.loads(experiment["config"])

    if "generations" not in experiment_config:
        experiment_config["generations"] = "[]"

    generations = json.loads(experiment_config["generations"])

    name = plugin["name"]
    plugin_name = plugin["plugin"]
    script_parameters = plugin["script_parameters"]

    slug = shared.slugify(name)

    # If name is greater than 100 characters, truncate it
    if len(slug) > 100:
        slug = slug[:100]
        print("Generation name is too long, truncating to 100 characters")

    generation = {"name": slug, "plugin": plugin_name, "script_parameters": script_parameters}

    generations.append(generation)

    await db.experiment_update_config(experimentId, "generations", json.dumps(generations))

    return {"message": f"Experiment {experimentId} updated with plugin {plugin_name}"}


@router.get("/delete")
async def experiment_delete_generation(experimentId: int, generation_name: str):
    """Delete an evaluation from an experiment. This will delete the directory in the experiment
    and remove the global plugin from the specific experiment."""
    try:
        print("Deleting generation", experimentId, generation_name)
        experiment = await db.experiment_get(experimentId)

        if experiment is None:
            return {"message": f"Experiment {experimentId} does not exist"}

        experiment_config = json.loads(experiment["config"])

        if "generations" not in experiment_config:
            return {"message": f"Experiment {experimentId} has no generations"}

        generations = json.loads(experiment_config["generations"])

        # remove the evaluation from the list:
        generations = [e for e in generations if e["name"] != generation_name]

        await db.experiment_update_config(experimentId, "generations", json.dumps(generations))

        return {"message": f"Generation {generations} deleted from experiment {experimentId}"}
    except Exception as e:
        print("Error in delete_generation_task", e)
        raise e


# @TODO delete the following function and use the plugin file function


@router.post("/edit")
async def edit_evaluation_generation(experimentId: int, plugin: Any = Body()):
    """Get the contents of the evaluation"""
    try:
        experiment = await db.experiment_get(experimentId)

        # if the experiment does not exist, return an error:
        if experiment is None:
            return {"message": f"Experiment {experimentId} does not exist"}

        eval_name = plugin["evalName"]
        updated_json = plugin["script_parameters"]

        plugin_name = updated_json["plugin_name"]
        template_name = updated_json["template_name"]

        experiment_config = json.loads(experiment["config"])

        # updated_json = json.loads(updated_json)

        if "generations" not in experiment_config:
            return {"message": f"Experiment {experimentId} has no generations"}

        generations = json.loads(experiment_config["generations"])

        # Remove fields model_name, model_architecture and plugin_name from the updated_json
        # as they are not needed in the generations list
        updated_json.pop("model_name", None)
        updated_json.pop("model_architecture", None)
        updated_json.pop("plugin_name", None)
        updated_json.pop("template_name", None)

        for generation in generations:
            if generation["name"] == eval_name and generation["plugin"] == plugin_name:
                generation["script_parameters"] = updated_json
                generation["name"] = template_name

        await db.experiment_update_config(experimentId, "generations", json.dumps(generations))

        return {"message": "OK"}
    except Exception as e:
        print("Error in edit_evaluation_task", e)
        raise e


@router.get("/get_generation_plugin_file_contents")
async def get_generation_plugin_file_contents(experimentId: int, plugin_name: str):
    # first get the experiment name:
    data = await db.experiment_get(experimentId)

    # if the experiment does not exist, return an error:
    if data is None:
        return {"message": f"Experiment {experimentId} does not exist"}

    # experiment_name = data["name"]

    # print(f"{EXPERIMENTS_DIR}/{experiment_name}/generation/{eval_name}/main.py")

    file_name = "main.py"
    plugin_path = dirs.plugin_dir_by_name(plugin_name)

    # now get the file contents
    try:
        with open(os.path.join(plugin_path, file_name), "r") as f:
            file_contents = f.read()
    except FileNotFoundError:
        return "error file not found"

    return file_contents


@router.get("/run_generation_script")
async def run_generation_script(experimentId: int, plugin_name: str, generation_name: str, job_id: str):
    plugin_name = secure_filename(plugin_name)
    generation_name = secure_filename(generation_name)

    experiment_details = await db.experiment_get(id=experimentId)

    if experiment_details is None:
        return {"message": f"Experiment {experimentId} does not exist"}
    config = json.loads(experiment_details["config"])

    experiment_name = experiment_details["name"]
    model_name = config["foundation"]

    if config["foundation_filename"] is None or config["foundation_filename"].strip() == "":
        model_file_path = ""
    else:
        model_file_path = config["foundation_filename"]
    model_type = config["foundation_model_architecture"]
    model_adapter = config["adaptor"]

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
        if "generations" in experiment_details["config"]:
            experiment_details["config"]["generations"] = json.loads(experiment_details["config"]["generations"])

    all_generations = experiment_details["config"]["generations"]
    this_generation = None
    for generation in all_generations:
        if generation["name"] == generation_name:
            this_generation = generation
            break

    if this_generation is None:
        return {"message": f"Error: generation {generation_name} does not exist in experiment"}
    template_config = this_generation["script_parameters"]
    # print("GET OUTPUT JOB DATA", await get_job_output_file_name("2", plugin_name, eval_name, template_config))
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
            "--generation_name",
            generation_name,
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

    subprocess_command = [sys.executable, dirs.PLUGIN_HARNESS] + extra_args

    print(f">Running {subprocess_command}")

    output_file = await dirs.generation_output_file(experiment_name, generation_name)

    with open(job_output_file, "w") as f:
        process = await asyncio.create_subprocess_exec(*subprocess_command, stdout=f, stderr=subprocess.PIPE)
        await process.communicate()

    with open(output_file, "w") as f:
        # Copy all contents from job_output_file to output_file
        with open(job_output_file, "r") as job_output:
            for line in job_output:
                f.write(line)
        # process = await asyncio.create_subprocess_exec(
        #     *subprocess_command,
        #     stdout=f,
        #     stderr=subprocess.PIPE
        # )
        # await process.communicate()


async def get_job_output_file_name(job_id: str, plugin_name: str):
    job_id = secure_filename(str(job_id))
    plugin_name = secure_filename(plugin_name)
    try:
        plugin_dir = dirs.plugin_dir_by_name(plugin_name)

        # job output is stored in separate files with a job number in the name...
        if os.path.exists(os.path.join(plugin_dir, f"output_{job_id}.txt")):
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
async def get_output(experimentId: int, generation_name: str):
    """Get the output of an evaluation"""
    data = await db.experiment_get(experimentId)
    # if the experiment does not exist, return an error:
    if data is None:
        return {"message": f"Experiment {experimentId} does not exist"}

    experiment_name = data["name"]

    # sanitize the input:
    generation_name = urllib.parse.unquote(generation_name)

    generation_output_file = await dirs.generation_output_file(experiment_name, generation_name)
    if not os.path.exists(generation_output_file):
        return {"message": "Output file does not exist"}

    print(f"Returning output file: {generation_output_file}.")

    # return the whole file as a file response:
    return FileResponse(generation_output_file)
