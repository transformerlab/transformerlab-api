from asyncio import create_subprocess_exec
import asyncio
import json
import os
import subprocess
import sys
import threading
import re
import time
import unicodedata
from pathlib import Path
from transformerlab.shared import dirs


from anyio import open_process
from anyio.streams.text import TextReceiveStream

import transformerlab.db as db

from transformerlab.shared.dirs import (
    GLOBAL_LOG_PATH)


def popen_and_call(onExit, input='', output_file=None, *popenArgs, **popenKWArgs):
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
        if (output_file != None):
            log = open(output_file, 'a')
            # get the current date and time as a string:
            current_time = time.strftime("%Y-%m-%d %H:%M:%S")
            print("Printing to file: " + output_file)
            log.write(f'\n\n-- RUN {current_time}--\n')
            log.flush()
        else:
            print("No output file specified, printing to stdout")
            log = subprocess.PIPE

        proc = subprocess.Popen(
            *popenArgs, **popenKWArgs, stdin=subprocess.PIPE, stdout=log, stderr=log)
        proc.communicate(input=input.encode("utf-8"))
        proc.wait()
        onExit()
        return

    thread = threading.Thread(
        target=runInThread, args=(onExit, popenArgs, popenKWArgs))
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
        value = (
            unicodedata.normalize("NFKD", value)
            .encode("ascii", "ignore")
            .decode("ascii")
        )
    value = re.sub(r"[^\w\s-]", "", value.lower())
    return re.sub(r"[-\s]+", "-", value).strip("-_")


async def async_run_python_script_and_update_status(python_script: list[str], job_id: str, begin_string: str):
    """
    Use this script for one time, long running scripts that have a definite end. For example
    downloading a model.

    This function runs a python script and updates the status of the job in the database
    to IN_PROGRESS when the python script prints begin_string to stderr

    The FastAPI worker uses stderr, not stdout"""

    print(f"Job {job_id} Running async python script: " + str(python_script))

    command = [sys.executable, '-u', *python_script]

    process = await open_process(command=command, stderr=subprocess.STDOUT, stdout=subprocess.PIPE)

    # read stderr and print:
    if process.stdout:
        async for text in TextReceiveStream(process.stdout):
            print("**SUB PROCESS:( "+text+" )**")
            if begin_string in text:
                print(f"Job {job_id} now in progress!")
                await db.job_update_status(job_id=job_id, status="IN_PROGRESS")

    await process.wait()

    if process.returncode == 0:
        print(f"Job {job_id} completed successfully")
        await db.job_update_status(job_id=job_id, status="COMPLETE")
    else:
        print(
            f"ERROR: Job {job_id} failed with exit code {process.returncode}.")
        await db.job_update_status(job_id=job_id, status="FAILED")

    return process


async def async_run_python_daemon_and_update_status(python_script: list[str], job_id: str, begin_string: str, set_process_id_function=None):
    """Use this function for daemon processes, for example setting up a model for inference.
    This function is helpful when the start of the daemon process takes a while. So you can
    wait for "begin_string" to be mentioned in stderr in order to let the caller know that
    the daemon is ready to accept input.

    This function runs a python script and updates the status of the job in the database
    to IN_PROGRESS when the python script prints begin_string to stderr

    The FastAPI worker uses stderr, not stdout"""

    print("🥼 Running python script: " + str(python_script))

    command = [sys.executable, *python_script]
    print(command)

    # open a file to write the output to:
    log = open(GLOBAL_LOG_PATH, 'a')

    process = await asyncio.create_subprocess_exec(*command, stdin=None, stderr=subprocess.STDOUT, stdout=subprocess.PIPE)

    line = await process.stdout.readline()
    error_msg = None
    while line:
        decoded = line.decode()

        # If we hit the begin_string then the daemon is started and we can return!
        if begin_string in decoded:
            if set_process_id_function != None:
                set_process_id_function(process)
            print(f"Worker job {job_id} started successfully")
            await db.job_update_status(job_id=job_id, status="COMPLETE")
            return process

        # Watch the output for any errors and store the latest error
        elif ("stderr" in decoded) and ("ERROR" in decoded):
            error_msg = decoded.split("| ")[-1]

        print(decoded)
        log.write(decoded)
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


async def run_job(job_id: str, job_config, experiment_name: str = "default"):
    # This runs a specified job number defined
    # by template_id
    print("Running job: " + str(job_id))

    # A job is a specific run of a job_template.
    # So first we pull up the specified job_template id
    template_id = job_config["template_id"]

    # Get the template
    template: dict[_KT, _VT] = await db.get_training_template(template_id)

    print("Template: " + str(template))
    job_type = str(template['type'])

    # Get the plugin script name:
    template_config = json.loads(template['config'])
    plugin_name = str(template_config["plugin_name"])

    # Get the job details from the database:
    job_details = await db.job_get(job_id)
    experiment_id = job_details["experiment_id"]
    # Get the experiment details from the database:
    experiment_details = await db.experiment_get(experiment_id)
    experiment_details_as_string = json.dumps(experiment_details)
    experiment_name = experiment_details["name"]

    # The script is in workspace/experiments/plugins/<plugin_name>/main.py so we need to
    # form that string:
    plugin_location = dirs.plugin_dir_by_name(plugin_name)
    output_file = os.path.join(plugin_location, f"output_{job_id}.txt")

    def on_train_complete():
        print('Training Job is Complete')
        db.job_update_sync(job_id, "COMPLETE")

    def on_job_complete():
        db.job_update_sync(job_id, "COMPLETE")

    if job_type == "LoRA":
        model_name = template_config["model_name"]
        template_config = json.loads(template['config'])
        adaptor_name = template_config["adaptor_name"]
        template_config["job_id"] = job_id
        template_config["adaptor_output_dir"] = os.path.join(
            dirs.WORKSPACE_DIR, "adaptors", model_name, adaptor_name)
        template_config["output_dir"] = os.path.join(
            dirs.WORKSPACE_DIR, "tensorboards", f"job{job_id}")

        # Create a file in the temp directory to store the inputs:
        tempdir = os.path.join(dirs.WORKSPACE_DIR, "temp")
        if not os.path.exists(tempdir):
            os.makedirs(tempdir)
        input_file = os.path.join(tempdir, f"plugin_input_{job_id}.json")
        # The following two ifs convert nested JSON strings to JSON objects -- this is a hack
        # and should be done in the API itself
        if "config" in experiment_details:
            experiment_details["config"] = json.loads(
                experiment_details["config"])
            if "inferenceParams" in experiment_details["config"]:
                experiment_details["config"]["inferenceParams"] = json.loads(
                    experiment_details["config"]["inferenceParams"])
        input_contents = {"experiment": experiment_details,
                          "config": template_config}
        with open(input_file, 'w') as outfile:
            json.dump(input_contents, outfile, indent=4)

        # This calls the training plugin harness, which calls the actual training plugin
        plugin_harness = os.path.join(dirs.PLUGIN_SDK_DIR, "training_plugin_harness.py")
        training_popen_command = [
            "python3",
            plugin_harness,
            "--plugin_dir",
            plugin_location,
            "--input_file",
            input_file,
            "--experiment_name",
            experiment_name
        ]
        print("RUNNING: popen command:")
        print(training_popen_command)
        popen_and_call(on_train_complete,
                       experiment_details_as_string, output_file, training_popen_command)
    else:
        print("I don't know what to do with this job type: " + job_type)
        on_job_complete()

    await db.job_update_status(job_id, "RUNNING")
    return
