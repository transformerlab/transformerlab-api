import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from distutils.dir_util import copy_tree, remove_tree

from typing import Annotated, Any

from fastapi import APIRouter, Body

from jinja2 import Template

import transformerlab.db as db

from transformerlab.shared import shared

from transformerlab.routers.plugins import install_plugin

router = APIRouter(prefix="/experiment", tags=["experiment"])

EXPERIMENTS_DIR: str = "workspace/experiments"


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
    rootdir = os.environ.get("LLM_LAB_ROOT_PATH")
    experiment_dir = rootdir + f"/{EXPERIMENTS_DIR}/" + experiment_name
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

    # now get the file contents
    try:
        with open(f"workspace/plugins/{plugin_name}/main.py", "r") as f:
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
    if not os.path.exists("workspace/temp"):
        os.makedirs("workspace/temp")
    input_file = "workspace/temp/plugin_input_" + str(plugin_name) + ".json"

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

    # now run the script
    root_dir = os.environ.get("LLM_LAB_ROOT_PATH")
    if root_dir is None:
        return {"message": "LLM_LAB_ROOT_PATH not set"}

    script_directory = f"{root_dir}/workspace/plugins/{plugin_name}"
    script_path = f"{script_directory}/main.py"
    extra_args.extend(["--experiment_name", experiment_name, "--eval_name", eval_name, "--input_file", input_file,
                       "--model_name", model_name, "--model_architecture", model_type, "--model_adapter", model_adapter])

    subprocess_command = [sys.executable, script_path] + extra_args

    print(f">Running {subprocess_command}")

    with open(f"{script_directory}/output.txt", "w") as f:
        subprocess.run(args=subprocess_command, stdout=f)

# run_export_script
# This gets a plugin_name and a json list of parameters
# and sets everything up for an exporter plugin to run


@router.get("/{id}/run_exporter_script")
async def run_exporter_script(id: int, plugin_name: str, plugin_params: str = "{}"):

    # Load experiment details into config
    experiment_details = await db.experiment_get(id=id)
    if experiment_details is None:
        return {"message": f"Experiment {id} does not exist"}

    # Get input model parameters
    experiment_name = experiment_details["name"]
    config = json.loads(experiment_details["config"])
    input_model_id = config["foundation"]
    input_model_architecture = config["foundation_model_architecture"]

    # Generate output model details
    conversion_time = int(time.time())
    # TEMPORARY HACK to get model architecture assumes exporter is named <format>_exporter
    output_model_architecture = plugin_name.split('_')[0]
    output_model_dir = f"{output_model_architecture}-{input_model_id}-{conversion_time}"
    output_quant_bits = 4

    # Convert JSON parameters
    # And set default parameters for anything that didn't get passed in
    params = json.loads(plugin_params)
    if ("output_model_architecture" not in params):
        params["output_model_architecture"] = output_model_architecture
    if ("output_quant_bits" not in params):
        params["output_quant_bits"] = output_quant_bits
    if ("output_dir" not in params):
        params["output_dir"] = output_model_dir
    if ("output_model_id" not in params):
        params["output_model_id"] = f"TransformerLab/({output_model_dir})"
    if ("output_model_name" not in params):
        params["output_model_name"] = f"{input_model_id} - {output_model_architecture} {output_quant_bits}bit"

    root_dir = os.environ.get("LLM_LAB_ROOT_PATH")
    if root_dir is None:
        return {"message": "LLM_LAB_ROOT_PATH not set"}

    script_directory = f"{root_dir}/workspace/plugins/{plugin_name}"
    script_path = f"{script_directory}/main.py"

    # Create a database job
    job_id = await db.export_job_create(
        experiment_id=id, 
        exporter_name=plugin_name, 
        input_model_id=input_model_id, 
        input_model_architecture=input_model_architecture,
        params=params
    )

    # Setup arguments to pass to plugin
    args = ["--model_name", input_model_id, "--model_architecture", input_model_architecture,
            "--quant_bits", "4", "--job_id", str(job_id)]
    subprocess_command = [script_path] + args

    try:
        process = await shared.async_run_python_script_and_update_status(python_script=subprocess_command, job_id=job_id, begin_string="Exporting")
    except Exception as e:
        await db.job_update(job_id=job_id, status="FAILED")
        return {"message": f"Failed to export model. Exception: {e}"}

    if process.returncode != 0:
        return {"message": f"Failed to export model. Return code: {process.returncode}"}

    return {"message": "success", "job_id": job_id}


@router.get("/{id}/export/jobs")
async def get_export_jobs(id: int):
    jobs = await db.jobs_get_all_by_experiment_and_type(id, 'EXPORT_MODEL')
    return jobs


@router.get("/{id}/export/job")
async def get_export_job(id: int, jobId: str):
    job = await db.job_get(jobId)
    return job


@router.get(path="/{id}/get_conversations")
async def get_conversations(id: int):
    # first get the experiment name:
    data = await db.experiment_get(id)

    # if the experiment does not exist, return an error:
    if data is None:
        return {"message": f"Experiment {id} does not exist"}

    experiment_name = data["name"]

    rootdir = os.environ.get("LLM_LAB_ROOT_PATH")
    experiment_dir = rootdir + f"/{EXPERIMENTS_DIR}/" + experiment_name
    conversation_dir = experiment_dir + "/conversations/"

    # make directory if it does not exist:
    if not os.path.exists(f"{conversation_dir}"):
        os.makedirs(f"{conversation_dir}")

    # now get a list of all the files in the conversations directory
    conversations_files = []
    for filename in os.listdir(conversation_dir):
        if filename.endswith(".json"):
            conversations_files.append(filename)

    conversations_contents = []

    # now read each conversation and create a list of all conversations
    # and their contents
    for i in range(len(conversations_files)):
        with open(conversation_dir + conversations_files[i], "r") as f:
            new_conversation = {}
            new_conversation['id'] = conversations_files[i]
            # remove .json from end of id
            new_conversation['id'] = new_conversation['id'][:-5]
            new_conversation['contents'] = json.load(f)
            # use file timestamp to get a date
            new_conversation['date'] = os.path.getmtime(
                conversation_dir + conversations_files[i])
            conversations_contents.append(new_conversation)

    # sort the conversations by date
    conversations_contents.sort(key=lambda x: x['date'], reverse=True)

    return conversations_contents


@router.post(path="/{id}/save_conversation")
async def save_conversation(id: int, conversation_id: Annotated[str, Body()], conversation: Annotated[str, Body()]):
    # first get the experiment name:
    data = await db.experiment_get(id)

    # if the experiment does not exist, return an error:
    if data is None:
        return {"message": f"Experiment {id} does not exist"}

    experiment_name = data["name"]

    # The following prevents path traversal attacks:
    rootdir = os.environ.get("LLM_LAB_ROOT_PATH")
    experiment_dir = rootdir + f"/{EXPERIMENTS_DIR}/" + experiment_name
    conversation_dir = "conversations/"
    final_path = Path(experiment_dir).joinpath(
        conversation_dir + conversation_id + '.json').resolve().relative_to(experiment_dir)

    final_path = experiment_dir + "/" + str(final_path)

    # now save the conversation
    with open(final_path, "w") as f:
        f.write(conversation)

    return {"message": f"Conversation {conversation_id} saved"}


@router.delete(path="/{id}/delete_conversation")
async def delete_conversation(id: int, conversation_id: str):
    # first get the experiment name:
    data = await db.experiment_get(id)

    # if the experiment does not exist, return an error:
    if data is None:
        return {"message": f"Experiment {id} does not exist"}

    experiment_name = data["name"]

    # The following prevents path traversal attacks:
    rootdir = os.environ.get("LLM_LAB_ROOT_PATH")
    experiment_dir = rootdir + f"/{EXPERIMENTS_DIR}/" + experiment_name
    conversation_dir = "conversations/"
    final_path = Path(experiment_dir).joinpath(
        conversation_dir + conversation_id + '.json').resolve().relative_to(experiment_dir)

    final_path = experiment_dir + "/" + str(final_path)

    # now delete the conversation
    os.remove(final_path)

    return {"message": f"Conversation {conversation_id} deleted"}


# Rewrite of the plugin script management functions:

@router.get("/{id}/scripts/list")
async def experiment_list_scripts(id: int, type: str = None, filter: str = None):
    """ List all the scripts in the experiment """
    # first get the experiment name:
    data = await db.experiment_get(id)

    # if the experiment does not exist, return an error:
    if data is None:
        return {"message": f"Experiment {id} does not exist"}

    experiment_name = data["name"]

    # parse the filter variable which is formatted as key:value
    # for example, model_architecture:LLamaArchitecture
    filter_key = None
    filter_value = None
    if filter is not None:
        [filter_key, filter_value] = filter.split(':')

    # print(f"Filtering by {filter_key} with value {filter_value}")

    # The following prevents path traversal attacks:
    rootdir = os.environ.get("LLM_LAB_ROOT_PATH")

    if rootdir is None:
        return {"message": "LLM_LAB_ROOT_PATH not set"}

    scripts_dir = rootdir + f"/workspace/plugins"

    # now get a list of all the directories in the scripts directory:
    scripts_full_json = []

    # If the scripts dir doesn't exist, return empty:
    if not os.path.exists(scripts_dir):
        return scripts_full_json

    for filename in os.listdir(scripts_dir):
        if os.path.isdir(os.path.join(scripts_dir, filename)):
            # check the type of each index.json in each script dir
            try:
                plugin_info = json.load(
                    open(f"{scripts_dir}/{filename}/index.json", "r"))
            except FileNotFoundError:
                continue
            except json.decoder.JSONDecodeError:
                print(f"Error decoding {scripts_dir}/{filename}/index.json")
                continue

            plugin_type = None
            if "type" in plugin_info:
                plugin_type = plugin_info['type']

            # if the type of plugin matches with the type filter, or no filter is provided then continue:
            if type is None or type == plugin_type:
                # check if the plugin has the additional filter key as a property
                if filter_key is None:
                    scripts_full_json.append(plugin_info)
                else:
                    # check if the filter key is in the plugin_info:
                    if filter_key in plugin_info:
                        # check if, in the info, the value is an array
                        # If it is an array, then check for the value by iterating through
                        if isinstance(plugin_info[filter_key], list):
                            if filter_value is None or filter_value in plugin_info[filter_key]:
                                scripts_full_json.append(plugin_info)
                        # otherwise, check if the value matches
                        else:
                            if filter_value is None or filter_value == plugin_info[filter_key]:
                                scripts_full_json.append(plugin_info)
                    else:
                        print('item does not have key ' + filter_key)

    return scripts_full_json


@router.get(path="/{id}/install_plugin_to_experiment")
async def install_plugin_to_experiment(id: int, plugin_name: str):
    await install_plugin(plugin_name)
    return {"message": f"Plugin {plugin_name} installed"}


@router.get(path="/{id}/delete_plugin_from_experiment")
async def delete_plugin_from_experiment(id: int, plugin_name: str):
    rootdir = os.environ.get("LLM_LAB_ROOT_PATH")
    final_path = rootdir + "/workspace/plugins/" + plugin_name
    remove_tree(final_path)
    return {"message": f"Plugin {plugin_name} deleted from experiment {id}"}


@router.get("/download", summary="Download a dataset to the LLMLab server.")
async def plugin_download(plugin_slug: str):
    """Download a plugin and install to a local list of available plugins"""
    # Get plugin from plugin gallery:
    plugin = await db.get_plugin(plugin_slug)
    # Right now this plugin object doesn't contain the URL to the plugin, so we need to get that from the plugin gallery:
    # Fix this later by storing the information locally in the database
    plugin_gallery_json = open(
        "transformerlab/galleries/plugin-gallery.json", "r")
    plugin_gallery = json.load(plugin_gallery_json)

    # We hardcode this to the first object -- fix later
    url = plugin_gallery[0]['url']
    plugin_type = plugin_gallery[0]['type']

    client = httpx.AsyncClient()

    # Download index.json from the URL above:
    # Hack: this won't work because we can't download a file from our own server
    print('Getting the plugin index.json file at ' + url + "index.json")
    # index_json = urlopen(url + "index.json").read()
    response = await client.get(url + "index.json")
    index_json = response.text
    print('Downloaded...')

    # Convert index.json to a dict:
    index = json.loads(index_json)

    # Add index.json to the list of files to download:
    index["files"].append("index.json")

    for file in index["files"]:
        # Download each file
        print("Downloading " + file + "...")
        response = await client.get(url + file)
        file_contents = response.text
        # Save each file to workspace/plugins/<plugin_slug>/<file>
        os.makedirs(
            f"workspace/plugins/{plugin_slug}/", mode=0o755, exist_ok=True)
        with open(f"workspace/plugins/{plugin_slug}/{file}", "w") as f:
            f.write(file_contents)

    await db.save_plugin(plugin_slug, plugin_type)

    await client.aclose()

    return {"message": "OK"}


#
# *****************************************************************************
# Everything below is used to manage plugins in the experiment/{id}/plugins/ directory
# *****************************************************************************


allowed_extensions: list[str] = [
    '.py', '.pyj2', '.ipynb', '.md', '.txt', '.sh', '.json']


@router.post("/{experimentId}/scripts/{pluginId}/save_file_contents")
async def plugin_save_file_contents(experimentId: str, pluginId: str, filename: str, file_contents: Annotated[str, Body()]):
    global allowed_extensions

    data = await db.experiment_get(experimentId)
    # if the experiment does not exist, return an error:
    if data is None:
        return {"message": f"Experiment {experimentId} does not exist"}

    experiment_name = data["name"]

    # remove file extension from file:
    [filename, file_ext] = os.path.splitext(filename)

    if file_ext not in allowed_extensions:
        return {"message": f"File extension {file_ext} for {filename} not supported"}

    # clean the file name:
    filename = shared.slugify(filename)
    pluginId = shared.slugify(pluginId)

    script_path = f"workspace/plugins/{pluginId}"

    # make directory if it does not exist:
    if not os.path.exists(f"{script_path}"):
        os.makedirs(f"{script_path}")

    # now save the file contents, overwriting if it already exists:
    with open(f"{script_path}/{filename}{file_ext}", "w") as f:
        print(f"Writing {script_path}/{filename}{file_ext}")
        f.write(file_contents)

    return {"message": f"{script_path}/{filename}{file_ext} file contents saved"}


@router.get("/{experimentId}/scripts/{pluginId}/file_contents")
async def plugin_get_file_contents(experimentId: str, pluginId: str, filename: str):
    global allowed_extensions

    data = await db.experiment_get(experimentId)
    # if the experiment does not exist, return an error:
    if data is None:
        return {"message": f"Experiment {experimentId} does not exist"}

    experiment_name = data["name"]

    # remove file extension from file:
    [filename, file_ext] = os.path.splitext(filename)

    if file_ext not in allowed_extensions:
        return {"message": f"File extension {file_ext} for {filename} not supported"}

    # The following prevents path traversal attacks:
    rootdir = os.environ.get("LLM_LAB_ROOT_PATH")
    plugin_dir = rootdir + f"/workspace/plugins/" + pluginId
    final_path = Path(plugin_dir).joinpath(
        filename + file_ext).resolve().relative_to(plugin_dir)

    final_path = plugin_dir + "/" + str(final_path)

    # now get the file contents
    try:
        with open(final_path, "r") as f:
            file_contents = f.read()
    except FileNotFoundError:
        return "FILE NOT FOUND"

    return file_contents


@router.get("/{experimentId}/scripts/{pluginId}/list_files")
async def plugin_list_files(experimentId: str, pluginId: str):
    global allowed_extensions

    data = await db.experiment_get(experimentId)
    # if the experiment does not exist, return an error:
    if data is None:
        return {"message": f"Experiment {experimentId} does not exist"}

    experiment_name = data["name"]
    rootdir = os.environ.get("LLM_LAB_ROOT_PATH")
    scripts_dir = rootdir + f"/workspace/plugins/" + pluginId

    # check if directory exists:
    if not os.path.exists(scripts_dir):
        return []

    # now get the list of files:
    files = []
    for file in os.listdir(scripts_dir):
        [filename, file_ext] = os.path.splitext(file)
        if file_ext in allowed_extensions:
            files.append(filename + file_ext)

    return files


@router.get("/{experimentId}/scripts/{pluginId}/create_new_file")
async def plugin_create_new_file(experimentId: str, pluginId: str, filename: str):
    global allowed_extensions

    data = await db.experiment_get(experimentId)
    # if the experiment does not exist, return an error:
    if data is None:
        return {"message": f"Experiment {experimentId} does not exist"}

    experiment_name = data["name"]

    # remove file extension from file:
    [filename, file_ext] = os.path.splitext(filename)

    if file_ext not in allowed_extensions:
        return {"error": "true", "message": f"File extension {file_ext} for {filename} not supported. Please use one of the following extensions: {allowed_extensions}"}

    # clean the file name:
    filename = shared.slugify(filename)
    pluginId = shared.slugify(pluginId)

    script_path = f"workspace/plugins/{pluginId}"

    # make directory if it does not exist:
    if not os.path.exists(f"{script_path}"):
        os.makedirs(
            f"{script_path}")

    # now save the file contents, overwriting if it already exists:
    with open(f"{script_path}/{filename}{file_ext}", "w+") as f:
        # f.write("")
        pass

    return {"message": f"{script_path}/{filename}{file_ext} file created"}


@router.get(path="/{experimentId}/scripts/{pluginId}/delete_file")
async def plugin_delete_file(experimentId: str, pluginId: str, filename: str):
    global allowed_extensions

    data = await db.experiment_get(experimentId)
    # if the experiment does not exist, return an error:
    if data is None:
        return {"message": f"Experiment {experimentId} does not exist"}

    experiment_name = data["name"]

    # remove file extension from file:
    [filename, file_ext] = os.path.splitext(filename)

    if file_ext not in allowed_extensions:
        return {"error": "true", "message": f"File extension {file_ext} for {filename} not supported. Please use one of the following extensions: {allowed_extensions}"}

    # clean the file name:
    filename = shared.slugify(filename)
    pluginId = shared.slugify(pluginId)

    script_path = f"workspace/plugins/{pluginId}"

    # make directory if it does not exist:
    if not os.path.exists(f"{script_path}"):
        return {"error": "true", "message": f"{script_path} does not exist"}

    # now delete the file contents
    os.remove(f"{script_path}/{filename}{file_ext}")

    return {"message": f"{script_path}/{filename}{file_ext} file deleted"}


@router.get(path="/{experimentId}/scripts/new_plugin")
async def plugin_new_plugin_directory(experimentId: str, pluginId: str):
    global allowed_extensions

    data = await db.experiment_get(experimentId)
    # if the experiment does not exist, return an error:
    if data is None:
        return {"message": f"Experiment {experimentId} does not exist"}

    experiment_name = data["name"]

    # clean the file name:
    pluginId = shared.slugify(value=pluginId)

    script_path = f"workspace/plugins/{pluginId}"

    # make directory if it does not exist:
    if not os.path.exists(f"{script_path}"):
        os.makedirs(f"{script_path}")

    index_json = {
        "uniqueId": pluginId,
        "name": pluginId,
        "description": "",
        "plugin-format": "python",
        "type": "trainer",
        "files": [],
        "parameters": [],
    }

    # add an index.json file:
    with open(f"{script_path}/index.json", "w+") as f:
        print(f"Writing {script_path}/index.json")
        json_content = json.dumps(index_json, indent=4)
        print(json_content)
        f.write(json.dumps(index_json, indent=4))

    return {"message": f"{script_path} directory created"}
