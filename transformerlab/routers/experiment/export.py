import json
import os
import time


from fastapi import APIRouter

import transformerlab.db as db
from transformerlab.shared import shared
from transformerlab.shared import dirs

from werkzeug.utils import secure_filename

router = APIRouter(prefix="/export", tags=["export"])


@router.get("/run_exporter_script")
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

    # The exporter plugin needs to know where to find the model
    input_model_path = config.get("foundation_filename", "")
    if not input_model_path:
        input_model_path = input_model_id

    # TODO: Verify that the model uses a supported format

    # Convert JSON parameters
    # And set default parameters for anything that didn't get passed in
    params = json.loads(plugin_params)
    q_type = ""
    if 'outtype' in params:
        q_type = params['outtype']
    elif 'q_bits' in params:
        q_type = str(params['q_bits']) + "bit"


    # Generate output model details
    conversion_time = int(time.time())
    output_model_architecture = plugin_architecture
    output_model_id = f"{output_model_architecture}-{input_model_id_without_author}-{conversion_time}"
    if len(q_type) > 0:
        output_model_id = f"{output_model_id}-{q_type}"
    output_model_name = f"{input_model_id_without_author} - {output_model_architecture}"
    if len(q_type) > 0:
        output_model_name = f"{output_model_name} - {q_type}"
    output_filename = ""

    # Check if a custom model name was provided
    custom_name = params.get('new_output_model_id')

    # GGUF is special: it generates a different format with only one file
    # For everything to work we need the model ID and output filename to match
    if output_model_architecture == "GGUF":
        if custom_name:
            output_model_id = f"{custom_name}.gguf"
        else:
            output_model_id = f"{input_model_id_without_author}-{conversion_time}.gguf"
            if len(q_type) > 0:
                output_model_id = f"{input_model_id_without_author}-{conversion_time}-{q_type}.gguf"

    # Llamafile needs special handling
    elif output_model_architecture == "llamafile":
        # For llamafile, we need to create a directory and put a .llamafile file inside it
        if custom_name:
            # Use custom name as the directory name (without extension)
            base_dir_name = custom_name
            # The actual file will have .llamafile extension
            output_filename = f"{custom_name}.llamafile"
        else:
            # Use model name as directory name
            base_dir_name = input_model_id_without_author
            output_filename = f"{input_model_id_without_author}.llamafile"
            
        # Now set output_model_id to be the directory name
        output_model_id = base_dir_name

    # Generic handling for any other exporter that supports custom names
    elif custom_name:
        output_model_id = custom_name

    # Check if a model with the same output_model_id already exists
    # If so, append _1, _2, etc. to the end of the name

    base_name, ext = os.path.splitext(output_model_id)
    counter = 1
    candidate = f"{base_name}{ext}"

    while os.path.exists(os.path.join(dirs.MODELS_DIR, candidate)):
        candidate = f"{base_name}_{counter}{ext}"
        counter += 1

    output_model_id = candidate
    
    # Update output_filename if this is a llamafile (since we need to match directory and filename)
    if output_model_architecture == "llamafile" and counter > 1:
        output_filename = f"{candidate}.llamafile"
        
    output_filename = output_model_id

    # Figure out plugin and model output directories
    script_directory = dirs.plugin_dir_by_name(plugin_name)

    output_model_id = secure_filename(output_model_id)

    output_path = os.path.join(dirs.MODELS_DIR, output_model_id)
    
    # Create the output directory if it doesn't exist
    # Use exist_ok=True to avoid errors if directory already exists
    os.makedirs(output_path, exist_ok=True)

    # Create a job in the DB with the details of this export
    job_data = dict(
        exporter_name=plugin_name,
        input_model_id=input_model_id,
        input_model_path=input_model_path,
        input_model_architecture=input_model_architecture,
        output_model_id=output_model_id,
        output_model_architecture=output_model_architecture,
        output_model_name=output_model_name,
        output_model_path=output_path,
        params=params,
    )
    job_data_json = json.dumps(job_data)
    job_id = await db.export_job_create(experiment_id=id, job_data_json=job_data_json)

    # Setup arguments to pass to plugin
    args = [
        "--plugin_dir",
        script_directory,
        "--model_name",
        input_model_id,
        "--model_path",
        input_model_path,
        "--model_architecture",
        input_model_architecture,
        "--output_dir",
        output_path,
        "--output_model_id",
        output_model_id,
    ]

    # Add additional parameters that are unique to the plugin (defined in info.json and passed in via plugin_params)
    for key in params:
        new_param = [f"--{key}", params[key]]
        args.extend(new_param)

    # Run the export plugin
    # This calls the training plugin harness, which calls the actual training plugin
    subprocess_command = [dirs.PLUGIN_HARNESS] + args
    try:
        process = await shared.async_run_python_script_and_update_status(
            python_script=subprocess_command, job_id=job_id, begin_string="Exporting"
        )
    except Exception as e:
        import logging

        logging.error(f"Failed to export model. Exception: {e}")
        await db.job_update_status(job_id=job_id, status="FAILED")
        return {"message": "Failed to export model due to an internal error."}

    if process.returncode != 0:
        fail_msg = f"Failed to export model. Return code: {process.returncode}"
        await db.job_update_status(job_id=job_id, status="FAILED")
        print(fail_msg)
        return {"message": fail_msg}

    # Model create was successful!
    # Create an info.json file so this can be read by the system
    # TODO: Add parameters to json_data
    output_model_full_id = f"TransformerLab/{output_model_id}"
    model_description = [
        {
            "model_id": output_model_full_id,
            "model_filename": output_filename,
            "name": output_model_name,
            "local_model": True,
            "json_data": {
                "uniqueID": output_model_full_id,
                "name": output_model_name,
                "model_filename": output_filename,
                "description": f"{output_model_architecture} model generated by Transformer Lab based on {input_model_id}",
                "source": "transformerlab",
                "architecture": output_model_architecture,
                "huggingface_repo": "",
                "params": plugin_params,
            },
        }
    ]
    model_description_file = open(os.path.join(output_path, "info.json"), "w")
    json.dump(model_description, model_description_file)
    model_description_file.close()

    return {"message": "success", "job_id": job_id}


@router.get("/jobs")
async def get_export_jobs(id: int):
    jobs = await db.jobs_get_all_by_experiment_and_type(id, "EXPORT_MODEL")
    return jobs


@router.get("/job")
async def get_export_job(id: int, jobId: str):
    job = await db.job_get(jobId)
    return job
