import json
import os
import time
import asyncio
import logging
import subprocess
import sys

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

import transformerlab.db as db
from transformerlab.shared import shared
from transformerlab.shared import dirs
from transformerlab.routers.serverinfo import watch_file

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

    # GGUF is special: it generates a different format with only one file
    # For everything to work we need the model ID and output filename to match
    if output_model_architecture == "GGUF":
        
        output_model_id = f"{input_model_id_without_author}-{conversion_time}.gguf"
        if len(q_type) > 0:
            output_model_id = f"{input_model_id_without_author}-{conversion_time}-{q_type}.gguf"

        output_filename = output_model_id

    # Figure out plugin and model output directories
    script_directory = dirs.plugin_dir_by_name(plugin_name)

    output_model_id = secure_filename(output_model_id)

    output_path = os.path.join(dirs.MODELS_DIR, output_model_id)

    # Create a job in the DB with the details of this export
    job_data = dict(
        plugin=plugin_name,
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
        "--job_id",
        str(job_id),
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
    subprocess_command = [sys.executable, dirs.PLUGIN_HARNESS] + args
    try:
        # Get the output file path
        job_output_file = await get_output_file_name(job_id)
        
        # Create the output file and run the process with output redirection
        with open(job_output_file, "w") as f:
            process = await asyncio.create_subprocess_exec(
                *subprocess_command, 
                stdout=f, 
                stderr=subprocess.PIPE,
                cwd=script_directory
            )
            _, stderr = await process.communicate()

            try:
                stderr_str = stderr.decode("utf-8", errors="replace") 
            except Exception as e:
                stderr_str = f"[stderr decode error]: {e}"

            if stderr_str.strip():
                if "Traceback" in stderr_str or "Error" in stderr_str or "Exception" in stderr_str:
                    print(f"Error output: {stderr_str}")
                    f.write(f"\nError output:\n{stderr_str}")
                else:
                    print(f"Standard error stream:\n{stderr_str}")
                    f.write(f"\nStandard error stream:\n{stderr_str}")

            if process.returncode != 0:
                fail_msg = f"Failed to export model. Return code: {process.returncode}"
                await db.job_update_status(job_id=job_id, status="FAILED")
                print(fail_msg)
                return {"message": fail_msg}
                
    except Exception as e:
        import logging

        logging.error(f"Failed to export model. Exception: {e}")
        await db.job_update_status(job_id=job_id, status="FAILED")
        return {"message": "Failed to export model due to an internal error."}

    # Model create was successful!
    # Create an info.json file so this can be read by the system
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

    await db.job_update_status(job_id=job_id, status="COMPLETE")
    return {"message": "success", "job_id": job_id}


@router.get("/jobs")
async def get_export_jobs(id: int):
    jobs = await db.jobs_get_all_by_experiment_and_type(id, "EXPORT_MODEL")
    return jobs


@router.get("/job")
async def get_export_job(id: int, jobId: str):
    job = await db.job_get(jobId)
    return job


async def get_output_file_name(job_id: str):
    try:
        # Ensure job_id is a string
        job_id = str(job_id)
        
        # Get job data
        job = await db.job_get(job_id)
        job_data = job["job_data"]
        
        # Check if it has a custom output file path
        if job_data.get("output_file_path") is not None:
            return job_data["output_file_path"]
        
        # Get the plugin name from the job data
        plugin_name = job_data.get("plugin")
        if not plugin_name:
            raise ValueError("Exporter name not found in job data")
        
        # Get the plugin directory
        plugin_dir = dirs.plugin_dir_by_name(plugin_name)
        
        job_id = secure_filename(job_id)
        
        # Check for output file with job id
        if os.path.exists(os.path.join(plugin_dir, f"output_{job_id}.txt")):
            output_file = os.path.join(plugin_dir, f"output_{job_id}.txt")
        else:
            # Create the output file path even if it doesn't exist yet
            output_file = os.path.join(plugin_dir, f"output_{job_id}.txt")
        
        return output_file
    except Exception as e:
        raise e
    

@router.get("/job/{job_id}/stream_output")
async def watch_export_log(job_id: str):
    try:
        job_id = secure_filename(job_id)
        output_file_name = await get_output_file_name(job_id)
    except ValueError as e:
        # if the value error starts with "No output file found for job" then wait 4 seconds and try again
        # because the file might not have been created yet
        if str(e).startswith("No output file found for job"):
            await asyncio.sleep(4)
            print("Retrying to get output file in 4 seconds...")
            output_file_name = await get_output_file_name(job_id)
        else:
            logging.error(f"ValueError: {e}")
            return "An internal error has occurred!"

    return StreamingResponse(
        # we force polling because i can't get this to work otherwise -- changes aren't detected
        watch_file(output_file_name, start_from_beginning=True, force_polling=True),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "Access-Control-Allow-Origin": "*"},
    )