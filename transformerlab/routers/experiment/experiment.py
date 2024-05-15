import json
import os
from pathlib import Path
import subprocess
import sys
import time

from typing import Annotated, Any

from fastapi import APIRouter, Body

import transformerlab.db as db
from transformerlab.shared import shared
from transformerlab.shared import dirs
from transformerlab.routers.experiment import rag, documents, plugins, conversations


router = APIRouter(prefix="/experiment", tags=["experiment"])

router.include_router(
    router=rag.router, prefix="/{experimentId}", tags=["rag"])
router.include_router(
    router=documents.router, prefix="/{experimentId}", tags=["documents"])
router.include_router(
    router=plugins.router, prefix="/{id}", tags=["plugins"])
router.include_router(
    router=conversations.router, prefix="/{experimentId}", tags=["conversations"])

EXPERIMENTS_DIR: str = dirs.EXPERIMENTS_DIR


@router.get("/")
async def experiments_get_all():
    return await db.experiment_get_all()


@router.get("/create")
async def experiments_create(name: str):
    newid = await db.experiment_create(name, "{}")
    return newid


@router.get("/{id}")
async def experiment_get(id: int):
    data = await db.experiment_get(id)

    # convert the JSON string called config to json object
    data["config"] = json.loads(data["config"])
    return data


@router.get("/{id}/delete")
async def experiments_delete(id: int):
    await db.experiment_delete(id)
    return {"message": f"Experiment {id} deleted"}


@router.get("/{id}/update")
async def experiments_update(id: int, name: str):
    await db.experiment_update(id, name)
    return {"message": f"Experiment {id} updated to {name}"}


@router.get("/{id}/update_config")
async def experiments_update_config(id: int, key: str, value: str):
    await db.experiment_update_config(id, key, value)
    return {"message": f"Experiment {id} updated"}


@router.post("/{id}/prompt")
async def experiments_save_prompt_template(id: int, template: Annotated[str, Body()]):
    await db.experiment_save_prompt_template(id, template)
    return {"message": f"Experiment {id} prompt template saved"}


@router.post("/{id}/save_file_contents")
async def experiment_save_file_contents(id: int, filename: str, file_contents: Annotated[str, Body()]):
    # first get the experiment name:
    data = await db.experiment_get(id)

    # if the experiment does not exist, return an error:
    if data is None:
        return {"message": f"Experiment {id} does not exist"}

    experiment_name = data["name"]

    # remove file extension from file:
    [filename, file_ext] = os.path.splitext(filename)

    if (file_ext != '.py') and (file_ext != '.ipynb') and (file_ext != '.md'):
        return {"message": f"File extension {file_ext} not supported"}

    # clean the file name:
    filename = shared.slugify(filename)

    # make directory if it does not exist:
    if not os.path.exists(f"{EXPERIMENTS_DIR}/{experiment_name}"):
        os.makedirs(f"{EXPERIMENTS_DIR}/{experiment_name}")

    # now save the file contents, overwriting if it already exists:
    with open(f"{EXPERIMENTS_DIR}/{experiment_name}/{filename}{file_ext}", "w") as f:
        f.write(file_contents)

    return {"message": f"{EXPERIMENTS_DIR}/{experiment_name}/{filename}{file_ext} file contents saved"}


@router.get("/{id}/file_contents")
async def experiment_get_file_contents(id: int, filename: str):
    # first get the experiment name:
    data = await db.experiment_get(id)

    # if the experiment does not exist, return an error:
    if data is None:
        return {"message": f"Experiment {id} does not exist"}

    experiment_name = data["name"]

    # remove file extension from file:
    [filename, file_ext] = os.path.splitext(filename)

    allowed_extensions = ['.py', '.ipynb', '.md', '.txt']

    if file_ext not in allowed_extensions:
        return {"message": f"File extension {file_ext} for {filename} not supported"}

    # clean the file name:
    # filename = shared.slugify(filename)

    # The following prevents path traversal attacks:
    experiment_dir = dirs.experiment_dir_by_name(experiment_name)
    final_path = Path(experiment_dir).joinpath(
        filename + file_ext).resolve().relative_to(experiment_dir)

    final_path = experiment_dir + "/" + str(final_path)
    print("Listing Contents of File: " + final_path)

    # now get the file contents
    try:
        with open(final_path, "r") as f:
            file_contents = f.read()
    except FileNotFoundError:
        return ""

    return file_contents


@router.post("/{id}/add_evaluation")
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


@router.get("/{id}/delete_eval_from_experiment")
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


@router.get("/{id}/get_evaluation_plugin_file_contents")
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


@router.get("/{id}/run_evaluation_script")
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

    script_directory = dirs.plugin_dir_by_name(plugin_name)
    script_path = f"{script_directory}/main.py"
    extra_args.extend(["--experiment_name", experiment_name, "--eval_name", eval_name, "--input_file", input_file,
                       "--model_name", model_name, "--model_architecture", model_type, "--model_adapter", model_adapter])

    subprocess_command = [sys.executable, script_path] + extra_args

    print(f">Running {subprocess_command}")

    with open(f"{script_directory}/output.txt", "w") as f:
        subprocess.run(args=subprocess_command, stdout=f)


@router.get("/{id}/run_exporter_script")
async def run_exporter_script(id: int, plugin_name: str, plugin_architecture: str, plugin_params: str = "{}"):
    """
    plugin_name: the id of the exporter plugin to run
    plugin_architecture: A string containing the standard name of plugin architecture
    plugin_params: a string of JSON containing parameters for this plugin (found in plugins info.json)
    """

    # Load experiment details into config
    experiment_details = await db.experiment_get(id=id)
    if experiment_details is None:
        return {"message": f"Experiment {id} does not exist"}

    # Get input model parameters
    config = json.loads(experiment_details["config"])
    input_model_id = config["foundation"]
    input_model_id_without_author = input_model_id.split("/")[-1]
    input_model_architecture = config["foundation_model_architecture"]

    # TODO: Verify that the model uses a supported format
    # According to MLX docs (as of Jan 16/24) supported formats are:
    # Mistral, Llama, Phi-2

    # Convert JSON parameters
    # And set default parameters for anything that didn't get passed in
    params = json.loads(plugin_params)

    # Generate output model details
    conversion_time = int(time.time())
    output_model_architecture = plugin_architecture
    output_model_id = f"{output_model_architecture}-{input_model_id_without_author}-{conversion_time}"
    output_model_name = f"{input_model_id_without_author} - {output_model_architecture}"
    output_filename = ""

    # GGUF is special: it generates a different format with only one file
    # For everything to work we need the model ID and output filename to match
    if (output_model_architecture == "GGUF"):
        output_model_id = f"{input_model_id_without_author}-{conversion_time}.gguf"
        output_filename = output_model_id

    # Figure out plugin and model output directories
    script_directory = dirs.plugin_dir_by_name(plugin_name)
    script_path = f"{script_directory}/main.py"

    output_path = os.path.join(dirs.MODELS_DIR, output_model_id)
    os.makedirs(output_path)

    # Create a job in the DB with the details of this export
    job_data = dict(
        exporter_name=plugin_name,
        input_model_id=input_model_id,
        input_model_architecture=input_model_architecture,
        output_model_id=output_model_id,
        output_model_architecture=output_model_architecture,
        output_model_name=output_model_name,
        output_model_path=output_path,
        params=params
    )
    job_data_json = json.dumps(job_data)
    job_id = await db.export_job_create(
        experiment_id=id,
        job_data_json=job_data_json
    )

    # Setup arguments to pass to plugin
    args = [
        "--model_name", input_model_id,
        "--model_architecture", input_model_architecture,
        "--output_dir", output_path,
        "--output_model_id", output_model_id
    ]

    # Add additional parameters that are unique to the plugin (defined in info.json and passed in via plugin_params)
    for key in params:
        new_param = [f"--{key}", params[key]]
        args.extend(new_param)

    # Run the export plugin
    subprocess_command = [script_path] + args
    try:
        process = await shared.async_run_python_script_and_update_status(python_script=subprocess_command, job_id=job_id, begin_string="Exporting")
    except Exception as e:
        fail_msg = f"Failed to export model. Exception: {e}"
        await db.job_update_status(job_id=job_id, status="FAILED")
        print(fail_msg)
        return {"message": fail_msg}

    if process.returncode != 0:
        fail_msg = f"Failed to export model. Return code: {process.returncode}"
        await db.job_update_status(job_id=job_id, status="FAILED")
        print(fail_msg)
        return {"message": fail_msg}

    # Model create was successful!
    # Create an info.json file so this can be read by the system
    # TODO: Add parameters to json_data
    output_model_full_id = f"TransformerLab/{output_model_id}"
    model_description = [{
        "model_id": output_model_full_id,
        "model_filename": output_filename,
        "name": output_model_name,
        "json_data": {
            "uniqueID": output_model_full_id,
            "name": output_model_name,
            "description": f"{output_model_architecture} model generated by TransformerLab based on {input_model_id}",
            "architecture": output_model_architecture,
            "huggingface_repo": "",
            "params": plugin_params
        }
    }]
    model_description_file = open(os.path.join(output_path, "info.json"), "w")
    json.dump(model_description, model_description_file)
    model_description_file.close()

    return {"message": "success", "job_id": job_id}


@router.get("/{id}/export/jobs")
async def get_export_jobs(id: int):
    jobs = await db.jobs_get_all_by_experiment_and_type(id, 'EXPORT_MODEL')
    return jobs


@router.get("/{id}/export/job")
async def get_export_job(id: int, jobId: str):
    job = await db.job_get(jobId)
    return job
