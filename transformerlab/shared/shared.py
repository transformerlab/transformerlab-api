import asyncio
import json
import os
import re
import shutil
import psutil
import subprocess
import sys
import threading
import time
import unicodedata

from anyio import open_process
from anyio.streams.text import TextReceiveStream
from werkzeug.utils import secure_filename

from transformerlab.db.db import experiment_get, experiment_get_by_name
from transformerlab.db.sync import job_mark_as_complete_if_running, job_update_sync
import transformerlab.db.jobs as db_jobs
from transformerlab.routers.experiment.evals import run_evaluation_script
from transformerlab.routers.experiment.generations import run_generation_script
from transformerlab.shared import dirs
from transformerlab.shared.dirs import GLOBAL_LOG_PATH


def popen_and_call(onExit, input="", output_file=None, *popenArgs, **popenKWArgs):
    """
    Runs a subprocess.Popen, and then calls the function onExit when the
    subprocess completes.

    Use it exactly the way you'd normally use subprocess.Popen, except include a
    callable to execute as the first argument. onExit is a callable object, and
    *popenArgs and **popenKWArgs are simply passed up to subprocess.Popen.

    from https://stackoverflow.com/questions/2581817/python-subprocess-callback-when-cmd-exits

    #TODO: There is an async IO way of doing this instead:
    https://docs.python.org/3/library/asyncio-subprocess.html#asyncio.create_subprocess_exec
    If we use the above then we can probably make onExit a coroutine and await it
    but when I tried to implement it as above, it would not work. The subprocess
    wouldn't work concurrently as expected.
    """

    def runInThread(onExit, popenArgs, popenKWArgs):
        if output_file is not None:
            log = open(output_file, "a")
            # get the current date and time as a string:
            current_time = time.strftime("%Y-%m-%d %H:%M:%S")
            print("Printing to file: " + output_file)
            log.write(f"\n\n-- RUN {current_time}--\n")
            log.flush()
        else:
            print("No output file specified, printing to stdout")
            log = subprocess.PIPE

        proc = subprocess.Popen(*popenArgs, **popenKWArgs, stdin=subprocess.PIPE, stdout=log, stderr=log)
        proc.communicate(input=input.encode("utf-8"))
        proc.wait()
        onExit()
        return

    thread = threading.Thread(target=runInThread, args=(onExit, popenArgs, popenKWArgs))
    thread.start()

    return thread  # returns immediately after the thread starts


def slugify(value, allow_unicode=False):
    """
    Copied from https://github.com/django/django/blob/master/django/utils/text.py
    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize("NFKC", value)
    else:
        value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    value = re.sub(r"[^\w\s-]", "", value.lower())
    return re.sub(r"[-\s]+", "-", value).strip("-_")


async def async_run_python_script_and_update_status(python_script: list[str], job_id: str, begin_string: str):
    """
    Use this script for one time, long running scripts that have a definite end. For example
    downloading a model.

    This function runs a python script and updates the status of the job in the database
    to RUNNING when the python script prints begin_string to stderr

    The FastAPI worker uses stderr, not stdout"""

    print(f"Job {job_id} Running async python script: " + str(python_script))
    # Extract plugin location from the python_script list
    plugin_location = None
    if "--plugin_dir" in python_script:
        for i, arg in enumerate(python_script):
            if arg == "--plugin_dir" and i + 1 < len(python_script):
                plugin_location = python_script[i + 1]
                break

    # Check if plugin has a venv directory
    if plugin_location:
        plugin_location = os.path.normpath(plugin_location)
        if not plugin_location.startswith(dirs.PLUGIN_DIR):
            print(f"Plugin location {plugin_location} is not in {dirs.PLUGIN_DIR}")
            raise Exception(f"Plugin location {plugin_location} is not in {dirs.PLUGIN_DIR}")
        if os.path.exists(os.path.join(plugin_location, "venv")) and os.path.isdir(
            os.path.join(plugin_location, "venv")
        ):
            venv_path = os.path.join(plugin_location, "venv")
            print(f">Plugin has virtual environment, activating venv from {venv_path}")
            venv_python = os.path.join(venv_path, "bin", "python")
            command = [venv_python, *python_script]
        else:
            print(">Using system Python interpreter")
            command = [sys.executable, *python_script]

    else:
        print(">Using system Python interpreter")
        command = [sys.executable, *python_script]  # Skip the original Python interpreter

    process = await open_process(command=command, stderr=subprocess.STDOUT, stdout=subprocess.PIPE)

    # read stderr and print:
    if process.stdout:
        async for text in TextReceiveStream(process.stdout):
            print(">> " + text)
            if begin_string in text:
                print(f"Job {job_id} now in progress!")
                await db_jobs.job_update_status(job_id=job_id, status="RUNNING")

            # Check the job_data column for the stop flag:
            job_row = await db_jobs.job_get(job_id)
            job_data = job_row.get("job_data", None)
            if job_data and job_data.get("stop", False):
                print(f"Job {job_id}: 'stop' flag detected. Cancelling job.")
                raise asyncio.CancelledError()

    try:
        await process.wait()

        if process.returncode == 0:
            print(f"Job {job_id} completed successfully")
            await db_jobs.job_update_status(job_id=job_id, status="COMPLETE")
        else:
            print(f"ERROR: Job {job_id} failed with exit code {process.returncode}.")
            await db_jobs.job_update_status(job_id=job_id, status="FAILED")

        return process

    except asyncio.CancelledError:
        process.kill()
        await process.wait()

        print(f"Job {job_id} cancelled.")

        raise asyncio.CancelledError()


async def read_process_output(process, job_id):
    await process.wait()
    returncode = process.returncode
    if returncode == 0:
        print("Worker Process completed successfully")
    else:
        print(f"ERROR: Worker Process ended with exit code {returncode}.")
    with open(GLOBAL_LOG_PATH, "a") as log:
        log.write(f"Inference Server Terminated with {returncode}.\n")
        log.flush()
    # so we should delete the pid file:
    pid_file = os.path.join(dirs.TEMP_DIR, f"worker_job_{job_id}.pid")
    if os.path.exists(pid_file):
        os.remove(pid_file)


async def async_run_python_daemon_and_update_status(
    python_script: list[str], job_id: str, begin_string: str, set_process_id_function=None
):
    """Use this function for daemon processes, for example setting up a model for inference.
    This function is helpful when the start of the daemon process takes a while. So you can
    wait for "begin_string" to be mentioned in stderr in order to let the caller know that
    the daemon is ready to accept input.

    This function runs a python script and updates the status of the job in the database
    to RUNNING when the python script prints begin_string to stderr

    The FastAPI worker uses stderr, not stdout"""

    print("🏃‍♂️ Running python script: " + str(python_script))

    # Extract plugin location from the python_script list
    plugin_location = None
    for i, arg in enumerate(python_script):
        if arg == "--plugin_dir" and i + 1 < len(python_script):
            plugin_location = python_script[i + 1]
            break

    # Open a file to write the output to:
    log = open(GLOBAL_LOG_PATH, "a")

    # Check if plugin has a venv directory
    if plugin_location:
        plugin_location = os.path.normpath(plugin_location)
        if not plugin_location.startswith(dirs.PLUGIN_DIR):
            print(f"Plugin location {plugin_location} is not in {dirs.PLUGIN_DIR}")
            raise Exception(f"Plugin location {plugin_location} is not in {dirs.PLUGIN_DIR}")
        if os.path.exists(os.path.join(plugin_location, "venv")) and os.path.isdir(
            os.path.join(plugin_location, "venv")
        ):
            venv_path = os.path.join(plugin_location, "venv")
            print(f">Plugin has virtual environment, activating venv from {venv_path}")
            venv_python = os.path.join(venv_path, "bin", "python")
            command = [venv_python, *python_script]
        else:
            print(">Using system Python interpreter")
            command = [sys.executable, *python_script]

    else:
        print(">Using system Python interpreter")
        command = [sys.executable, *python_script]  # Skip the original Python interpreter

    process = await asyncio.create_subprocess_exec(
        *command, stdin=None, stderr=subprocess.STDOUT, stdout=subprocess.PIPE
    )

    pid = process.pid
    pid_file = os.path.join(dirs.TEMP_DIR, f"worker_job_{job_id}.pid")
    with open(pid_file, "w") as f:
        f.write(str(pid))

    line = await process.stdout.readline()
    error_msg = None
    while line:
        decoded = line.decode()

        # If we hit the begin_string then the daemon is started and we can return!
        if begin_string in decoded:
            if set_process_id_function is not None:
                if set_process_id_function:
                    set_process_id_function(process)
            print(f"Worker job {job_id} started successfully")
            await db_jobs.job_update_status(job_id=job_id, status="COMPLETE")

            # Schedule the read_process_output coroutine in the current event
            # so we can keep watching this process, but return back to the caller
            # so that the REST call can complete
            asyncio.create_task(read_process_output(process, job_id))

            return process

        # Watch the output for any errors and store the latest error
        elif ("stderr" in decoded) and ("ERROR" in decoded):
            error_msg = decoded.split("| ")[-1]

        if log:
            log.write(decoded)
            log.flush()
        log.flush()
        line = await process.stdout.readline()

    # If we're here then stdout didn't return and we didn't start the daemon
    # Wait on the process and return the error
    await process.wait()
    returncode = process.returncode
    if not error_msg:
        error_msg = f"Process terminated prematurely with exit code {returncode}"

    print(f"ERROR: Worker job {job_id} failed with exit code {returncode}.")
    print(error_msg)
    await db_jobs.job_update_status(job_id=job_id, status="FAILED", error_msg=error_msg)
    return process


async def run_job(job_id: str, job_config, experiment_name: str = "default", job_details: dict = None):
    # This runs a specified job number defined
    # by template_id
    print("Running job: " + str(job_id))

    print("Job Config: " + str(job_config))
    print("Job Details: " + str(job_details))
    master_job_type = job_details["type"]
    print(master_job_type)

    if master_job_type == "TASK":
        """we define a TASK job as a job where we just ask
        the worker to run the related python script, passing in the parameters
        that are defined in job_config"""
        # plugin = job_config["plugin"]
        # update task to be marked as COMPLETE:
        await db_jobs.job_update_status(job_id, "COMPLETE")
        # implement rest later
        return {"status": "complete", "job_id": job_id, "message": "Task job completed successfully"}
    elif master_job_type == "EVAL":
        experiment = await experiment_get_by_name(experiment_name)
        experiment_id = experiment["id"]
        plugin_name = job_config["plugin"]
        eval_name = job_config.get("evaluator", "")
        await db_jobs.job_update_status(job_id, "RUNNING")
        print("Running evaluation script")
        WORKSPACE_DIR = dirs.WORKSPACE_DIR
        plugin_location = dirs.plugin_dir_by_name(plugin_name)
        if not os.path.exists(plugin_location):
            await db_jobs.job_update_status(job_id, "FAILED")
            return {"status": "error", "job_id": job_id, "message": "Evaluation job failed: No plugin found"}

        output_temp_file_dir = os.path.join(WORKSPACE_DIR, "jobs", str(job_id))
        if not os.path.exists(output_temp_file_dir):
            os.makedirs(output_temp_file_dir)
        evals_output_file = os.path.join(output_temp_file_dir, f"output_{job_id}.txt")
        # Create output file if it doesn't exist
        if not os.path.exists(evals_output_file):
            with open(evals_output_file, "w") as f:
                f.write("")
        await run_evaluation_script(experiment_id, plugin_name, eval_name, job_id)
        # Check if stop button was clicked and update status accordingly
        job_row = await db_jobs.job_get(job_id)
        job_data = job_row.get("job_data", None)
        if job_data is None:
            await db_jobs.job_update_status(job_id, "FAILED")
            return {"status": "error", "job_id": job_id, "message": "Evaluation job failed: No job data found"}

        if job_data.get("stop", False):
            await db_jobs.job_update_status(job_id, "STOPPED")
            return {"status": "stopped", "job_id": job_id, "message": "Evaluation job was stopped by user"}
        else:
            # Only set to COMPLETE if not already FAILED
            current_status = await db_jobs.job_get_status(job_id)
            if current_status != "FAILED":
                await db_jobs.job_update_status(job_id, "COMPLETE")
            return {"status": "complete", "job_id": job_id, "message": "Evaluation job completed successfully"}
    elif master_job_type == "GENERATE":
        experiment = await experiment_get_by_name(experiment_name)
        experiment_id = experiment["id"]
        plugin_name = job_config["plugin"]
        generation_name = job_config["generator"]
        await db_jobs.job_update_status(job_id, "RUNNING")
        print("Running generation script")
        WORKSPACE_DIR = dirs.WORKSPACE_DIR
        plugin_location = dirs.plugin_dir_by_name(plugin_name)
        if not os.path.exists(plugin_location):
            await db_jobs.job_update_status(job_id, "FAILED")
            return {"status": "error", "job_id": job_id, "message": "Generation job failed: No plugin found"}
        output_temp_file_dir = os.path.join(WORKSPACE_DIR, "jobs", str(job_id))
        if not os.path.exists(output_temp_file_dir):
            os.makedirs(output_temp_file_dir)
        gen_output_file = os.path.join(output_temp_file_dir, f"output_{job_id}.txt")
        # Create output file if it doesn't exist
        if not os.path.exists(gen_output_file):
            with open(gen_output_file, "w") as f:
                f.write("")

        await run_generation_script(experiment_id, plugin_name, generation_name, job_id)

        # Check should_stop flag and update status accordingly
        job_row = await db_jobs.job_get(job_id)
        job_data = job_row.get("job_data", None)
        if job_data is None:
            await db_jobs.job_update_status(job_id, "FAILED")
            return {"status": "error", "job_id": job_id, "message": "Generation job failed: No job data found"}

        if job_data.get("stop", False):
            await db_jobs.job_update_status(job_id, "STOPPED")
            return {"status": "stopped", "job_id": job_id, "message": "Generation job was stopped by user"}
        else:
            # Only set to COMPLETE if not already FAILED
            current_status = await db_jobs.job_get_status(job_id)
            if current_status != "FAILED":
                await db_jobs.job_update_status(job_id, "COMPLETE")
            return {"status": "complete", "job_id": job_id, "message": "Generation job completed successfully"}
    elif master_job_type == "EXPORT":
        plugin_name = job_config["plugin"]
        await db_jobs.job_update_status(job_id, "RUNNING")
        print("Running export script")
        WORKSPACE_DIR = dirs.WORKSPACE_DIR
        output_temp_file_dir = os.path.join(WORKSPACE_DIR, "jobs", str(job_id))
        if not os.path.exists(output_temp_file_dir):
            os.makedirs(output_temp_file_dir)
        export_output_file = os.path.join(output_temp_file_dir, f"output_{job_id}.txt")
        # Create output file if it doesn't exist
        if not os.path.exists(export_output_file):
            with open(export_output_file, "w") as f:
                f.write("")

        # Run the export script using the existing run_exporter_script function
        from transformerlab.routers.experiment.export import run_exporter_script

        config = job_config["config"]
        # Extract parameters from the job config
        experiment_id = int(job_details["experiment_id"])
        plugin_name = config["plugin_name"]
        plugin_architecture = config["output_model_architecture"]
        plugin_params = json.dumps(config["params"])
        plugin_location = dirs.plugin_dir_by_name(plugin_name)
        if not os.path.exists(plugin_location):
            await db_jobs.job_update_status(job_id, "FAILED")
            return {"status": "error", "job_id": job_id, "message": "Evaluation job failed: No plugin found"}

        # Call the existing run_exporter_script function with the existing job_id
        result = await run_exporter_script(
            id=experiment_id,
            plugin_name=plugin_name,
            plugin_architecture=plugin_architecture,
            plugin_params=plugin_params,
            job_id=job_id,
        )

        # Check the result and update job status accordingly
        if result.get("status") == "success":
            # Only set to COMPLETE if not already FAILED
            current_status = await db_jobs.job_get_status(job_id)
            if current_status != "FAILED":
                await db_jobs.job_update_status(job_id, "COMPLETE")
                print(f"Export job {job_id} completed successfully")
            return {"status": "complete", "job_id": job_id, "message": "Export job completed successfully"}

        else:
            await db_jobs.job_update_status(job_id, "FAILED")
            print(f"Export job {job_id} failed")
            return {"status": "error", "job_id": job_id, "message": result.get("message", "Export job failed")}

    job_type = job_config["config"]["type"]

    # Get the plugin script name:
    template_config = job_config["config"]
    plugin_name = str(template_config["plugin_name"])

    # Get the job details from the database:
    job_details = await db_jobs.job_get(job_id)
    experiment_id = job_details["experiment_id"]
    # Get the experiment details from the database:
    experiment_details = await experiment_get(experiment_id)
    print("Experiment Details: ", experiment_details)
    experiment_details_as_string = json.dumps(experiment_details)
    experiment_name = experiment_details["name"]
    experiment_dir = dirs.experiment_dir_by_name(experiment_name)

    # The script is in workspace/experiments/plugins/<plugin_name>/main.py so we need to
    # form that string:
    WORKSPACE_DIR = dirs.WORKSPACE_DIR
    plugin_location = dirs.plugin_dir_by_name(plugin_name)
    if not os.path.exists(plugin_location):
        await db_jobs.job_update_status(job_id, "FAILED")
        return {"status": "error", "job_id": job_id, "message": "Evaluation job failed: No plugin found"}
    output_temp_file_dir = os.path.join(WORKSPACE_DIR, "jobs", str(job_id))
    if not os.path.exists(output_temp_file_dir):
        os.makedirs(output_temp_file_dir)
    output_file = os.path.join(output_temp_file_dir, f"output_{job_id}.txt")

    def on_train_complete():
        print("Training Job: The process has finished")
        job_mark_as_complete_if_running(job_id)
        end_time = time.strftime("%Y-%m-%d %H:%M:%S")
        asyncio.run(db_jobs.job_update_job_data_insert_key_value(job_id, "end_time", end_time))

    def on_job_complete():
        job_update_sync(job_id, "COMPLETE")
        end_time = time.strftime("%Y-%m-%d %H:%M:%S")
        asyncio.run(db_jobs.job_update_job_data_insert_key_value(job_id, "end_time", end_time))

    if job_type == "LoRA":
        job_config = job_config["config"]
        model_name = job_config["model_name"]
        model_name = secure_filename(model_name)
        template_config = job_config
        adaptor_name = job_config.get("adaptor_name", "adaptor")
        template_config["job_id"] = job_id
        template_config["adaptor_output_dir"] = os.path.join(dirs.WORKSPACE_DIR, "adaptors", model_name, adaptor_name)
        template_config["output_dir"] = os.path.join(
            experiment_dir,
            "tensorboards",
            template_config["template_name"],
        )
        # Check if plugin has a venv directory
        venv_path = os.path.join(plugin_location, "venv")
        await db_jobs.job_update_status(job_id, "RUNNING")
        start_time = time.strftime("%Y-%m-%d %H:%M:%S")
        await db_jobs.job_update_job_data_insert_key_value(job_id, "start_time", start_time)

        if os.path.exists(venv_path) and os.path.isdir(venv_path):
            venv_python = os.path.join(venv_path, "bin", "python")

        tempdir = os.path.join(dirs.WORKSPACE_DIR, "temp")
        if not os.path.exists(tempdir):
            os.makedirs(tempdir)
        # Check if hyperparameter sweep is requested
        run_sweeps = template_config.get("run_sweeps", False)
        # if run_sweeps in ["on", "true", "yes"]:
        if run_sweeps:
            print(f"Hyperparameter sweep requested for job {job_id}")

            # Get sweep configuration
            sweep_config = template_config.get("sweep_config", {})
            if isinstance(sweep_config, str):
                try:
                    sweep_config = json.loads(sweep_config)
                except json.JSONDecodeError:
                    print(f"Error decoding sweep config JSON: {sweep_config}. Using default sweep configuration.")
                    sweep_config = {
                        "learning_rate": ["1e-5", "3e-5", "5e-5"],
                        "lora_rank": ["8", "16", "32"],
                        "lora_alpha": ["16", "32", "64"],
                        "batch_size": ["4", "8"],
                    }

            if not sweep_config:
                print("No sweep configuration provided. Using default sweep parameters.")
                sweep_config = {
                    "learning_rate": ["1e-5", "3e-5", "5e-5"],
                    # "lora_rank": ["8", "16", "32"],
                    # "lora_alpha": ["16", "32", "64"],
                    # "batch_size": ["4", "8"],
                }

            print(f"Sweep configuration: {json.dumps(sweep_config, indent=2)}")

            # Create sweep directory to store results
            sweep_dir = os.path.join(template_config["output_dir"], f"sweep_{job_id}")
            os.makedirs(sweep_dir, exist_ok=True)

            # Generate all configurations
            from itertools import product

            # Get all parameter names and their possible values
            param_names = list(sweep_config.keys())
            param_values = [sweep_config[name] for name in param_names]

            # Generate all combinations using product
            configs = []
            for values in product(*param_values):
                config = dict(zip(param_names, values))
                configs.append(config)

            total_configs = len(configs)
            print(f"Generated {total_configs} configurations for sweep")

            # Initialize sweep tracking
            await db_jobs.job_update_job_data_insert_key_value(job_id, "sweep_total", str(total_configs))
            await db_jobs.job_update_job_data_insert_key_value(job_id, "sweep_current", "0")

            # Get metrics configuration
            metric_name = template_config.get("sweep_metric", "eval/loss")
            lower_is_better = template_config.get("lower_is_better", "true").lower() in ["true", "yes", "on"]
            best_metric = float("inf") if lower_is_better else float("-inf")
            best_config = None

            # Store results for each run
            results = []

            # Run each configuration sequentially
            for i, config_params in enumerate(configs):
                print(f"\n--- Running configuration {i + 1}/{total_configs} ---")
                print(f"Parameters: {json.dumps(config_params, indent=2)}")

                # Create a unique run directory
                run_dir = os.path.join(sweep_dir, f"run_{i + 1}")
                os.makedirs(run_dir, exist_ok=True)

                # Create a unique adaptor directory for this run
                run_adaptor_dir = os.path.join(
                    dirs.WORKSPACE_DIR, "adaptors", secure_filename(model_name), f"{adaptor_name}_sweep_{i + 1}"
                )
                os.makedirs(run_adaptor_dir, exist_ok=True)

                # Create a copy of the template config for this run
                run_config = template_config.copy()

                # Update with the specific parameter values for this run
                for param_name, param_value in config_params.items():
                    run_config[param_name] = param_value

                # Set unique directories for this run
                run_config["output_dir"] = run_dir
                run_config["adaptor_output_dir"] = run_adaptor_dir

                # Create input file for this run
                run_input_file = os.path.join(tempdir, f"plugin_input_{job_id}_run_{i + 1}.json")
                run_input_contents = {"experiment": experiment_details, "config": run_config}
                with open(run_input_file, "w") as outfile:
                    json.dump(run_input_contents, outfile, indent=4)

                # Update job progress
                await db_jobs.job_update_sweep_progress(job_id, int((i / total_configs) * 100))
                await db_jobs.job_update_job_data_insert_key_value(job_id, "sweep_current", str(i + 1))
                await db_jobs.job_update_job_data_insert_key_value(
                    job_id, "sweep_running_config", json.dumps(config_params)
                )

                # Run the training job with this configuration
                run_output_file = os.path.join(sweep_dir, f"output_sweep_{job_id}.txt")
                await db_jobs.job_update_job_data_insert_key_value(
                    job_id, "sweep_output_file", os.path.join(sweep_dir, f"output_sweep_{job_id}.txt")
                )

                # Create command for this run
                if os.path.exists(venv_path) and os.path.isdir(venv_path):
                    print(f">Plugin has virtual environment, activating venv from {venv_path}")
                    venv_python = os.path.join(venv_path, "bin", "python")
                    run_command = [
                        venv_python,
                        dirs.PLUGIN_HARNESS,
                        "--plugin_dir",
                        plugin_location,
                        "--input_file",
                        run_input_file,
                        "--experiment_name",
                        experiment_name,
                    ]
                else:
                    print(">Using system Python interpreter")
                    run_command = [
                        sys.executable,
                        dirs.PLUGIN_HARNESS,
                        "--plugin_dir",
                        plugin_location,
                        "--input_file",
                        run_input_file,
                        "--experiment_name",
                        experiment_name,
                    ]

                # Replace synchronous subprocess.run with asyncio
                async def run_process_async(cmd, output_file):
                    # Open file for writing
                    with open(output_file, "a") as f:
                        # Create subprocess with piped stdout
                        process = await asyncio.create_subprocess_exec(
                            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT
                        )

                        # Process output in real-time
                        while True:
                            line = await process.stdout.readline()
                            if not line:
                                break

                            # Decode and write to file
                            decoded_line = line.decode("utf-8")
                            f.write(f"\n[Run {i + 1}/{total_configs}]: {decoded_line.strip()}")
                            f.flush()

                        # Wait for process to complete
                        await process.wait()
                        return process.returncode

                # Run the process asynchronously
                await run_process_async(run_command, run_output_file)

                # Delete the output adaptor directory if it exists
                if os.path.exists(run_adaptor_dir) and os.path.isdir(run_adaptor_dir):
                    print(f"Deleting adaptor directory: {run_adaptor_dir}")
                    shutil.rmtree(run_adaptor_dir, ignore_errors=True)

                # Check job data for training metrics
                try:
                    # Get latest metrics from job_data (assuming plugin saved metrics there)
                    metrics_path = os.path.join(run_dir, "metrics.json")
                    if os.path.exists(metrics_path):
                        with open(metrics_path, "r") as f:
                            run_metrics = json.load(f)
                    else:
                        # Fallback to a default metric value if no metrics found
                        run_metrics = {metric_name: 0.0}

                    # Track results
                    results.append(
                        {
                            "config": config_params,
                            "metrics": run_metrics,
                            "run_dir": run_dir,
                            "adaptor_dir": run_adaptor_dir,
                        }
                    )

                    # Check if this is the best result so far
                    if metric_name in run_metrics:
                        metric_value = run_metrics[metric_name]
                        is_better = (lower_is_better and metric_value < best_metric) or (
                            not lower_is_better and metric_value > best_metric
                        )

                        if best_config is None or is_better:
                            best_metric = metric_value
                            best_config = config_params.copy()

                            # Update job data with current best
                            await db_jobs.job_update_job_data_insert_key_value(
                                job_id, "sweep_best_config", json.dumps(best_config)
                            )
                            await db_jobs.job_update_job_data_insert_key_value(
                                job_id, "sweep_best_metric", json.dumps({metric_name: best_metric})
                            )
                except Exception as e:
                    print(f"Error processing metrics for run {i + 1}: {str(e)}")
                    results.append(
                        {"config": config_params, "error": str(e), "run_dir": run_dir, "adaptor_dir": run_adaptor_dir}
                    )

            # Save all results
            sweep_results = {
                "sweep_config": sweep_config,
                "results": results,
                "best_config": best_config,
                "best_metric": {metric_name: best_metric},
                "metric_name": metric_name,
                "lower_is_better": lower_is_better,
            }

            sweep_results_file = os.path.join(sweep_dir, "sweep_results.json")
            with open(sweep_results_file, "w") as f:
                json.dump(sweep_results, f, indent=2)

            await db_jobs.job_update_job_data_insert_key_value(job_id, "sweep_results_file", sweep_results_file)

            print("\n--- Sweep completed ---")
            print(f"Best configuration: {json.dumps(best_config, indent=2)}")
            print(f"Best {metric_name}: {best_metric}")
            await db_jobs.job_update_sweep_progress(job_id, 100)

            # Optionally train final model with best configuration
            train_final_model = template_config.get("train_final_model", True)
            if train_final_model and best_config:
                print("\n--- Training final model with best configuration ---")

                # Use the original output and adaptor directories for the final model
                final_config = template_config.copy()

                # Update with best parameters
                for param_name, param_value in best_config.items():
                    final_config[param_name] = param_value

                # Create input file for final run
                final_input_file = os.path.join(tempdir, f"plugin_input_{job_id}_final.json")
                final_input_contents = {"experiment": experiment_details, "config": final_config}
                with open(final_input_file, "w") as outfile:
                    json.dump(final_input_contents, outfile, indent=4)

                # Use the appropriate python interpreter
                if os.path.exists(venv_path) and os.path.isdir(venv_path):
                    venv_python = os.path.join(venv_path, "bin", "python")
                    final_command = [
                        venv_python,
                        dirs.PLUGIN_HARNESS,
                        "--plugin_dir",
                        plugin_location,
                        "--input_file",
                        final_input_file,
                        "--experiment_name",
                        experiment_name,
                    ]
                else:
                    final_command = [
                        sys.executable,
                        dirs.PLUGIN_HARNESS,
                        "--plugin_dir",
                        plugin_location,
                        "--input_file",
                        final_input_file,
                        "--experiment_name",
                        experiment_name,
                    ]

                # Run the final training synchronously
                popen_and_call(on_train_complete, experiment_details_as_string, output_file, final_command)
                return

            return

        else:
            # Create a file in the temp directory to store the inputs:
            tempdir = os.path.join(dirs.WORKSPACE_DIR, "temp")
            if not os.path.exists(tempdir):
                os.makedirs(tempdir)
            input_file = os.path.join(tempdir, f"plugin_input_{job_id}.json")
            # The following two ifs convert nested JSON strings to JSON objects -- this is a hack
            # and should be done in the API itself
            if "config" in experiment_details:
                experiment_details["config"] = json.loads(experiment_details["config"])
                if "inferenceParams" in experiment_details["config"]:
                    experiment_details["config"]["inferenceParams"] = json.loads(
                        experiment_details["config"]["inferenceParams"]
                    )
            input_contents = {"experiment": experiment_details, "config": template_config}
            with open(input_file, "w") as outfile:
                json.dump(input_contents, outfile, indent=4)

            start_time = time.strftime("%Y-%m-%d %H:%M:%S")
            await db_jobs.job_update_job_data_insert_key_value(job_id, "start_time", start_time)

            # Check if plugin has a venv directory
            venv_path = os.path.join(plugin_location, "venv")
            print("No hyperparameter sweep requested, running single job")
            if os.path.exists(venv_path) and os.path.isdir(venv_path):
                print(f">Plugin has virtual environment, activating venv from {venv_path}")
                venv_python = os.path.join(venv_path, "bin", "python")
                # Construct command that first activates venv then runs script
                training_popen_command = [
                    venv_python,
                    dirs.PLUGIN_HARNESS,
                    "--plugin_dir",
                    plugin_location,
                    "--input_file",
                    input_file,
                    "--experiment_name",
                    experiment_name,
                ]

            else:
                print(">Using system Python interpreter")
                training_popen_command = [
                    sys.executable,
                    dirs.PLUGIN_HARNESS,
                    "--plugin_dir",
                    plugin_location,
                    "--input_file",
                    input_file,
                    "--experiment_name",
                    experiment_name,
                ]

        popen_and_call(on_train_complete, experiment_details_as_string, output_file, training_popen_command)

    elif job_type == "pretraining":
        template_config = job_config["config"]
        template_config["job_id"] = job_id
        template_config["output_dir"] = os.path.join(
            experiment_dir,
            "tensorboards",
            template_config["template_name"],
        )

        # Create a file in the temp directory to store the inputs:
        tempdir = os.path.join(dirs.WORKSPACE_DIR, "temp")
        if not os.path.exists(tempdir):
            os.makedirs(tempdir)
        input_file = os.path.join(tempdir, f"plugin_input_{job_id}.json")
        # The following two ifs convert nested JSON strings to JSON objects -- this is a hack
        # and should be done in the API itself
        if "config" in experiment_details:
            experiment_details["config"] = json.loads(experiment_details["config"])
            if "inferenceParams" in experiment_details["config"]:
                experiment_details["config"]["inferenceParams"] = json.loads(
                    experiment_details["config"]["inferenceParams"]
                )
        input_contents = {"experiment": experiment_details, "config": template_config}
        with open(input_file, "w") as outfile:
            json.dump(input_contents, outfile, indent=4)

        start_time = time.strftime("%Y-%m-%d %H:%M:%S")
        await db_jobs.job_update_job_data_insert_key_value(job_id, "start_time", start_time)

        # Check if plugin has a venv directory
        venv_path = os.path.join(plugin_location, "venv")
        if os.path.exists(venv_path) and os.path.isdir(venv_path):
            print(f">Plugin has virtual environment, activating venv from {venv_path}")
            venv_python = os.path.join(venv_path, "bin", "python")
            # Construct command that first activates venv then runs script
            training_popen_command = [
                venv_python,
                dirs.PLUGIN_HARNESS,
                "--plugin_dir",
                plugin_location,
                "--input_file",
                input_file,
                "--experiment_name",
                experiment_name,
            ]
        else:
            print(">Using system Python interpreter")
            training_popen_command = [
                sys.executable,
                dirs.PLUGIN_HARNESS,
                "--plugin_dir",
                plugin_location,
                "--input_file",
                input_file,
                "--experiment_name",
                experiment_name,
            ]

        popen_and_call(on_train_complete, experiment_details_as_string, output_file, training_popen_command)

    elif job_type == "embedding":
        template_config = job_config["config"]
        template_config["job_id"] = job_id
        template_config["output_dir"] = os.path.join(
            experiment_dir,
            "tensorboards",
            template_config["template_name"],
        )

        if not os.path.exists(output_file):
            with open(output_file, "w") as f:
                f.write("")

        # Create a file in the temp directory to store the inputs:
        tempdir = os.path.join(dirs.WORKSPACE_DIR, "temp")
        if not os.path.exists(tempdir):
            os.makedirs(tempdir)
        input_file = os.path.join(tempdir, f"plugin_input_{job_id}.json")
        # The following two ifs convert nested JSON strings to JSON objects -- this is a hack
        # and should be done in the API itself
        if "config" in experiment_details:
            experiment_details["config"] = json.loads(experiment_details["config"])
            if "inferenceParams" in experiment_details["config"]:
                experiment_details["config"]["inferenceParams"] = json.loads(
                    experiment_details["config"]["inferenceParams"]
                )
        input_contents = {"experiment": experiment_details, "config": template_config}
        with open(input_file, "w") as outfile:
            json.dump(input_contents, outfile, indent=4)

        start_time = time.strftime("%Y-%m-%d %H:%M:%S")
        await db_jobs.job_update_job_data_insert_key_value(job_id, "start_time", start_time)

        # Check if plugin has a venv directory
        venv_path = os.path.join(plugin_location, "venv")
        if os.path.exists(venv_path) and os.path.isdir(venv_path):
            print(f">Plugin has virtual environment, activating venv from {venv_path}")
            venv_python = os.path.join(venv_path, "bin", "python")
            # Construct command that first activates venv then runs script
            training_popen_command = [
                venv_python,
                dirs.PLUGIN_HARNESS,
                "--plugin_dir",
                plugin_location,
                "--input_file",
                input_file,
                "--experiment_name",
                experiment_name,
            ]
        else:
            print(">Using system Python interpreter")
            training_popen_command = [
                sys.executable,
                dirs.PLUGIN_HARNESS,
                "--plugin_dir",
                plugin_location,
                "--input_file",
                input_file,
                "--experiment_name",
                experiment_name,
            ]

        popen_and_call(on_train_complete, experiment_details_as_string, output_file, training_popen_command)

    else:
        print("I don't know what to do with this job type: " + job_type)
        on_job_complete()

    await db_jobs.job_update_status(job_id, "RUNNING")
    return


rainbow = [
    "\033[38;5;196m",
    "\033[38;5;202m",
    "\033[38;5;226m",
    "\033[38;5;082m",
    "\033[38;5;021m",
    "\033[38;5;093m",
    "\033[38;5;163m",
]
reset = "\033[0m"


def print_in_rainbow(text):
    for i, line in enumerate(text.split("\n")):
        chunks = [line[i : i + 6] for i in range(0, len(line), 6)]
        for j, chunk in enumerate(chunks):
            print(rainbow[j % len(rainbow)], end="")
            print(chunk, end="")
            print(reset, end="")
        print("", flush=True)


def kill_sglang_subprocesses():
    current_pid = os.getpid()
    for proc in psutil.process_iter(attrs=["pid", "name", "cmdline"]):
        try:
            if proc.pid == current_pid:
                continue  # Skip self

            cmdline_list = proc.info.get("cmdline")
            if not cmdline_list:  # Handles None or empty list
                continue

            cmdline = " ".join(cmdline_list)
            if "sglang" in cmdline or "sglang::scheduler" in cmdline:
                proc.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
