import asyncio
import json
import os
import subprocess
import sys
import threading
import re
import time
import unicodedata
from transformerlab.routers.experiment.evals import run_evaluation_script
from transformerlab.routers.experiment.generations import run_generation_script
from transformerlab.shared import dirs
from werkzeug.utils import secure_filename


from anyio import open_process
from anyio.streams.text import TextReceiveStream

import transformerlab.db as db

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

    command = [sys.executable, "-u", *python_script]

    process = await open_process(command=command, stderr=subprocess.STDOUT, stdout=subprocess.PIPE)

    # read stderr and print:
    if process.stdout:
        async for text in TextReceiveStream(process.stdout):
            print(">> " + text)
            if begin_string in text:
                print(f"Job {job_id} now in progress!")
                await db.job_update_status(job_id=job_id, status="RUNNING")

            # Check the job_data column for the stop flag:
            job_row = await db.job_get(job_id)
            job_data = job_row.get("job_data", None)
            if job_data and job_data.get("stop", False):
                print(f"Job {job_id}: 'stop' flag detected. Cancelling job.")
                raise asyncio.CancelledError()

    try:
        await process.wait()

        if process.returncode == 0:
            print(f"Job {job_id} completed successfully")
            await db.job_update_status(job_id=job_id, status="COMPLETE")
        else:
            print(f"ERROR: Job {job_id} failed with exit code {process.returncode}.")
            await db.job_update_status(job_id=job_id, status="FAILED")

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

    command = [sys.executable, *python_script]
    print(command)

    # open a file to write the output to:
    log = open(GLOBAL_LOG_PATH, "a")

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
            await db.job_update_status(job_id=job_id, status="COMPLETE")

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
    await db.job_update_status(job_id=job_id, status="FAILED", error_msg=error_msg)
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
        await db.job_update_status(job_id, "COMPLETE")
        # implement rest later
        return
    elif master_job_type == "EVAL":
        experiment = await db.experiment_get_by_name(experiment_name)
        experiment_id = experiment["id"]
        plugin_name = job_config["plugin"]
        eval_name = job_config.get("evaluator", "")
        await db.job_update_status(job_id, "RUNNING")
        print("Running evaluation script")
        plugin_location = dirs.plugin_dir_by_name(plugin_name)
        evals_output_file = os.path.join(plugin_location, f"output_{job_id}.txt")
        # Create output file if it doesn't exist
        if not os.path.exists(evals_output_file):
            with open(evals_output_file, "w") as f:
                f.write("")
        await run_evaluation_script(experiment_id, plugin_name, eval_name, job_id)
        await db.job_update_status(job_id, "COMPLETE")
        return
    elif master_job_type == "GENERATE":
        experiment = await db.experiment_get_by_name(experiment_name)
        experiment_id = experiment["id"]
        plugin_name = job_config["plugin"]
        generation_name = job_config["generator"]
        await db.job_update_status(job_id, "RUNNING")
        print("Running generation script")
        plugin_location = dirs.plugin_dir_by_name(plugin_name)
        gen_output_file = os.path.join(plugin_location, f"output_{job_id}.txt")
        # Create output file if it doesn't exist
        if not os.path.exists(gen_output_file):
            with open(gen_output_file, "w") as f:
                f.write("")
        await run_generation_script(experiment_id, plugin_name, generation_name, job_id)
        await db.job_update_status(job_id, "COMPLETE")
        return

    job_type = job_config["config"]["type"]

    # Get the plugin script name:
    template_config = job_config["config"]
    plugin_name = str(template_config["plugin_name"])

    # Get the job details from the database:
    job_details = await db.job_get(job_id)
    experiment_id = job_details["experiment_id"]
    # Get the experiment details from the database:
    experiment_details = await db.experiment_get(experiment_id)
    experiment_details_as_string = json.dumps(experiment_details)
    experiment_name = experiment_details["name"]
    experiment_dir = dirs.experiment_dir_by_name(experiment_name)

    # The script is in workspace/experiments/plugins/<plugin_name>/main.py so we need to
    # form that string:
    plugin_location = dirs.plugin_dir_by_name(plugin_name)
    output_file = os.path.join(plugin_location, f"output_{job_id}.txt")

    def on_train_complete():
        print("Training Job: The process has finished")
        db.job_mark_as_complete_if_running(job_id)
        end_time = time.strftime("%Y-%m-%d %H:%M:%S")
        asyncio.run(db.job_update_job_data_insert_key_value(job_id, "end_time", end_time))

    def on_job_complete():
        db.job_update_sync(job_id, "COMPLETE")
        end_time = time.strftime("%Y-%m-%d %H:%M:%S")
        asyncio.run(db.job_update_job_data_insert_key_value(job_id, "end_time", end_time))

    if job_type == "LoRA":
        job_config = job_config["config"]
        model_name = job_config["model_name"]
        model_name = secure_filename(model_name)
        template_config = job_config
        adaptor_name = job_config["adaptor_name"]
        template_config["job_id"] = job_id
        template_config["adaptor_output_dir"] = os.path.join(dirs.WORKSPACE_DIR, "adaptors", model_name, adaptor_name)
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
        await db.job_update_job_data_insert_key_value(job_id, "start_time", start_time)

        # Check if plugin has a venv directory
        venv_path = os.path.join(plugin_location, "venv")
        if os.path.exists(venv_path) and os.path.isdir(venv_path):
            print(f">Plugin has virtual environment, activating venv from {venv_path}")
            venv_python = os.path.join(venv_path, "bin", "python")
            # Construct command that first activates venv then runs script
            training_popen_command = [
                "/bin/bash",
                "-c",
                f"source {os.path.join(venv_path, 'bin', 'activate')} && {venv_python} {dirs.PLUGIN_HARNESS} "
                + f'--plugin_dir "{plugin_location}" --input_file "{input_file}" --experiment_name "{experiment_name}"',
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
        await db.job_update_job_data_insert_key_value(job_id, "start_time", start_time)

        # Check if plugin has a venv directory
        venv_path = os.path.join(plugin_location, "venv")
        if os.path.exists(venv_path) and os.path.isdir(venv_path):
            print(f">Plugin has virtual environment, activating venv from {venv_path}")
            venv_python = os.path.join(venv_path, "bin", "python")
            # Construct command that first activates venv then runs script
            training_popen_command = [
                "/bin/bash",
                "-c",
                f"source {os.path.join(venv_path, 'bin', 'activate')} && {venv_python} {dirs.PLUGIN_HARNESS} "
                + f'--plugin_dir "{plugin_location}" --input_file "{input_file}" --experiment_name "{experiment_name}"',
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
        await db.job_update_job_data_insert_key_value(job_id, "start_time", start_time)

        # Check if plugin has a venv directory
        venv_path = os.path.join(plugin_location, "venv")
        if os.path.exists(venv_path) and os.path.isdir(venv_path):
            print(f">Plugin has virtual environment, activating venv from {venv_path}")
            venv_python = os.path.join(venv_path, "bin", "python")
            # Construct command that first activates venv then runs script
            training_popen_command = [
                "/bin/bash",
                "-c",
                f"source {os.path.join(venv_path, 'bin', 'activate')} && {venv_python} {dirs.PLUGIN_HARNESS} "
                + f'--plugin_dir "{plugin_location}" --input_file "{input_file}" --experiment_name "{experiment_name}"',
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

    await db.job_update_status(job_id, "RUNNING")
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
