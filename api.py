"""
The Entrypoint File for Transformer Lab's API Server.
"""

import os
import argparse
import asyncio

import json
import signal
import subprocess
from contextlib import asynccontextmanager
import sys
from werkzeug.utils import secure_filename

import fastapi
import httpx

# Using torch to test for CUDA and MPS support.
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from fastchat.constants import (
    ErrorCode,
)
from fastchat.protocol.openai_api_protocol import (
    ErrorResponse,
)

from transformerlab.db.jobs import job_create, job_get_error_msg, job_update_status
from transformerlab.db.db import experiment_get
import transformerlab.db.session as db
from transformerlab.shared.ssl_utils import ensure_persistent_self_signed_cert
from transformerlab.routers import (
    data,
    model,
    serverinfo,
    train,
    plugins,
    evals,
    config,
    tasks,
    prompts,
    tools,
    batched_prompts,
    diffusion,
    recipes,
    users,
)

import torch

try:
    from pynvml import nvmlShutdown

    HAS_AMD = False
except Exception:
    from pyrsmi import rocml

    HAS_AMD = True
from transformerlab import fastchat_openai_api
from transformerlab.routers.experiment import experiment
from transformerlab.routers.experiment import workflows
from transformerlab.routers.experiment import jobs
from transformerlab.shared import dirs
from transformerlab.shared import shared
from transformerlab.shared import galleries

from dotenv import load_dotenv

load_dotenv()

# The following environment variable can be used by other scripts
# who need to connect to the root DB, for example
os.environ["LLM_LAB_ROOT_PATH"] = dirs.ROOT_DIR
# environment variables that start with _ are
# used internally to set constants that are shared between separate processes. They are not meant to be
# to be overriden by the user.
os.environ["_TFL_WORKSPACE_DIR"] = dirs.WORKSPACE_DIR
os.environ["_TFL_SOURCE_CODE_DIR"] = dirs.TFL_SOURCE_CODE_DIR
# The temporary image directory for transformerlab
temp_image_dir = os.path.join(dirs.WORKSPACE_DIR, "temp", "images")
os.environ["TLAB_TEMP_IMAGE_DIR"] = str(temp_image_dir)

from transformerlab.routers.job_sdk import get_xmlrpc_router, get_trainer_xmlrpc_router  # noqa: E402


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Docs on lifespan events: https://fastapi.tiangolo.com/advanced/events/"""
    # Do the following at API Startup:
    print_launch_message()
    galleries.update_gallery_cache()
    spawn_fastchat_controller_subprocess()
    await db.init()
    if "--reload" in sys.argv:
        await install_all_plugins()
    # run the migration
    asyncio.create_task(migrate())
    asyncio.create_task(run_over_and_over())
    print("FastAPI LIFESPAN: 🏁 🏁 🏁 Begin API Server 🏁 🏁 🏁", flush=True)
    yield
    # Do the following at API Shutdown:
    await db.close()
    # Run the clean up function
    cleanup_at_exit()
    print("FastAPI LIFESPAN: Complete")


# the migrate function only runs the conversion function if no tasks are already present
async def migrate():
    if len(await tasks.tasks_get_all()) == 0:
        for exp in await experiment.experiments_get_all():
            await tasks.convert_all_to_tasks(exp["id"])


async def run_over_and_over():
    """Every three seconds, check for new jobs to run."""
    while True:
        await asyncio.sleep(3)
        await jobs.start_next_job()
        await workflows.start_next_step_in_workflow()


description = "Transformerlab API helps you do awesome stuff. 🚀"

tags_metadata = [
    {
        "name": "datasets",
        "description": "Actions used to manage the datasets used by Transformer Lab.",
    },
    {"name": "train", "description": "Actions for training models."},
    {"name": "experiment", "descriptions": "Actions for managinging experiments."},
    {
        "name": "model",
        "description": "Actions for interacting with huggingface models",  # TODO: is this true?
    },
    {
        "name": "serverinfo",
        "description": "Actions for interacting with the Transformer Lab server.",
    },
]

app = fastapi.FastAPI(
    title="Transformerlab API",
    description=description,
    summary="An API for working with LLMs.",
    version="0.0.1",
    terms_of_service="http://example.com/terms/",
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
    },
    lifespan=lifespan,
    openapi_tags=tags_metadata,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def create_error_response(code: int, message: str) -> JSONResponse:
    return JSONResponse(ErrorResponse(message=message, code=code).model_dump(), status_code=400)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return create_error_response(ErrorCode.VALIDATION_TYPE_ERROR, str(exc))


### END GENERAL API - NOT OPENAI COMPATIBLE ###


app.include_router(model.router)
app.include_router(serverinfo.router)
app.include_router(train.router)
app.include_router(data.router)
app.include_router(experiment.router)
app.include_router(plugins.router)
app.include_router(evals.router)
app.include_router(jobs.router)
app.include_router(tasks.router)
app.include_router(config.router)
app.include_router(prompts.router)
app.include_router(tools.router)
app.include_router(recipes.router)
app.include_router(batched_prompts.router)
app.include_router(fastchat_openai_api.router)
app.include_router(diffusion.router)
app.include_router(get_xmlrpc_router())
app.include_router(get_trainer_xmlrpc_router())

# This includes the FastAPI Users routers
app.include_router(users.router)


controller_process = None
worker_process = None


def spawn_fastchat_controller_subprocess():
    global controller_process
    logfile = open(os.path.join(dirs.FASTCHAT_LOGS_DIR, "controller.log"), "w")
    port = "21001"

    controller_process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "fastchat.serve.controller",
            "--port",
            port,
            "--log-file",
            os.path.join(dirs.FASTCHAT_LOGS_DIR, "controller.log"),
        ],
        stdout=logfile,
        stderr=logfile,
    )
    print(f"Started fastchat controller on port {port}")


async def install_all_plugins():
    all_plugins = await plugins.list_plugins()
    print("Re-copying all plugin files from source to workspace")
    for plugin in all_plugins:
        plugin_id = plugin["uniqueId"]
        print(f"Refreshing workspace plugin: {plugin_id}")
        await plugins.copy_plugin_files_to_workspace(plugin_id)


# @app.get("/")
# async def home():
#     return {"msg": "Welcome to Transformer Lab!"}


@app.get("/server/controller_start", tags=["serverinfo"])
async def server_controler_start():
    spawn_fastchat_controller_subprocess()
    return {"message": "OK"}


@app.get("/server/controller_stop", tags=["serverinfo"])
async def server_controller_stop():
    controller_process.terminate()
    return {"message": "OK"}


def set_worker_process_id(process):
    global worker_process
    worker_process = process


@app.get("/server/worker_start", tags=["serverinfo"])
async def server_worker_start(
    model_name: str,
    adaptor: str = "",
    model_filename: str | None = None,
    model_architecture: str = "",
    eight_bit: bool = False,
    cpu_offload: bool = False,
    inference_engine: str = "default",
    experiment_id: int = None,
    inference_params: str = "",
):
    global worker_process

    # the first priority for inference params should be the inference params passed in, then the inference parameters in the experiment
    # first we check to see if any inference params were passed in
    if inference_params != "":
        try:
            inference_params = json.loads(inference_params)
        except json.JSONDecodeError:
            return {"status": "error", "message": "malformed inference params passed"}
    # then we check to see if we are an experiment
    elif experiment_id is not None:
        try:
            experiment = await experiment_get(experiment_id)
            experiment_config = experiment["config"]
            if not isinstance(experiment_config, dict):
                experiment_config = json.loads(experiment_config)
            try:
                inference_params = experiment_config["inferenceParams"]
            except KeyError:
                print("No inference params found in experiment config, using empty dict")
                inference_params = {}
            if not isinstance(inference_params, dict):
                # if inference_params is a string, we need to parse it as JSON
                inference_params = json.loads(inference_params)
        except json.JSONDecodeError:
            return {"status": "error", "message": "malformed inference params passed"}
    # if neither are true, then we have an issue
    else:
        return {"status": "error", "message": "malformed inference params passed"}

    engine = inference_engine
    if "inferenceEngine" in inference_params and engine == "default":
        engine = inference_params.get("inferenceEngine")

    if engine == "default":
        return {"status": "error", "message": "no inference engine specified"}

    inference_engine = engine

    model_architecture = model_architecture

    plugin_name = inference_engine
    plugin_location = dirs.plugin_dir_by_name(plugin_name)

    model = model_name
    if model_filename is not None and model_filename != "":
        model = model_filename

    if adaptor != "":
        adaptor = f"{dirs.WORKSPACE_DIR}/adaptors/{secure_filename(model)}/{adaptor}"

    params = [
        dirs.PLUGIN_HARNESS,
        "--plugin_dir",
        plugin_location,
        "--model-path",
        model,
        "--model-architecture",
        model_architecture,
        "--adaptor-path",
        adaptor,
        "--parameters",
        json.dumps(inference_params),
    ]

    job_id = await job_create(type="LOAD_MODEL", status="STARTED", job_data="{}", experiment_id=experiment_id)

    print("Loading plugin loader instead of default worker")

    with open(dirs.GLOBAL_LOG_PATH, "a") as global_log:
        global_log.write(f"🏃 Loading Inference Server for {model_name} with {inference_params}\n")

    worker_process = await shared.async_run_python_daemon_and_update_status(
        python_script=params,
        job_id=job_id,
        begin_string="Application startup complete.",
        set_process_id_function=set_worker_process_id,
    )
    exitcode = worker_process.returncode
    if exitcode == 99:
        with open(dirs.GLOBAL_LOG_PATH, "a") as global_log:
            global_log.write(
                "GPU (CUDA) Out of Memory: Please try a smaller model or a different inference engine. Restarting the server may free up resources.\n"
            )
        return {
            "status": "error",
            "message": "GPU (CUDA) Out of Memory: Please try a smaller model or a different inference engine. Restarting the server may free up resources.",
        }
    if exitcode is not None and exitcode != 0:
        with open(dirs.GLOBAL_LOG_PATH, "a") as global_log:
            global_log.write(f"Error loading model: {model_name} with exit code {exitcode}\n")
        error_msg = await job_get_error_msg(job_id, experiment_id)
        if not error_msg:
            error_msg = f"Exit code {exitcode}"
            await job_update_status(job_id, "FAILED", experiment_id=experiment_id, error_msg=error_msg)
        return {"status": "error", "message": error_msg}
    with open(dirs.GLOBAL_LOG_PATH, "a") as global_log:
        global_log.write(f"Model loaded successfully: {model_name}\n")
    return {"status": "success", "job_id": job_id}


@app.get("/server/worker_stop", tags=["serverinfo"])
async def server_worker_stop():
    global worker_process
    print(f"Stopping worker process: {worker_process}")
    if worker_process is not None:
        from transformerlab.shared.shared import kill_sglang_subprocesses

        try:
            worker_process.terminate()
            kill_sglang_subprocesses()
            worker_process = None
        except Exception as e:
            print(f"Error stopping worker process: {e}")
    # check if there is a file called worker.pid, if so kill the related process:
    if os.path.isfile("worker.pid"):
        with open("worker.pid", "r") as f:
            pids = [line.strip() for line in f if line.strip()]
            for pid in pids:
                print(f"Killing worker process with PID: {pid}")
                try:
                    os.kill(int(pid), signal.SIGTERM)
                except ProcessLookupError:
                    print(f"Process {pid} no longer exists, skipping")
                except Exception as e:
                    print(f"Error killing process {pid}: {e}")
        # delete the worker.pid file:
        os.remove("worker.pid")
    return {"message": "OK"}


@app.get("/server/worker_healthz", tags=["serverinfo"])
async def server_worker_health(request: Request):
    models = []
    result = []
    try:
        models = await fastchat_openai_api.show_available_models()
    except httpx.HTTPError as exc:
        print(f"HTTP Exception for {exc.request.url} - {exc}")
        raise HTTPException(status_code=503, detail="No worker")

    # We create a new object with JUST the id of the models
    # we do this so that we get a clean object that can be used
    # by react to see if the object changed. If we returned the whole
    # model object, you would see some changes in the object that are
    # not relevant to the user -- triggering renders in React
    for model_data in models.data:
        result.append({"id": model_data.id})

    return result


# Add an endpoint that serves the static files in the ~/.transformerlab/webapp directory:
app.mount("/", StaticFiles(directory=dirs.STATIC_FILES_DIR, html=True), name="application")


def cleanup_at_exit():
    if controller_process is not None:
        print("🔴 Quitting spawned controller.")
        controller_process.kill()
    if worker_process is not None:
        print("🔴 Quitting spawned workers.")
        try:
            worker_process.kill()
        except ProcessLookupError:
            print(f"Process {worker_process.pid} doesn't exist so nothing to kill")
    if os.path.isfile("worker.pid"):
        with open("worker.pid", "r") as f:
            pids = [line.strip() for line in f if line.strip()]
            for pid in pids:
                os.kill(int(pid), signal.SIGTERM)
            os.remove("worker.pid")
    # Perform NVML Shutdown if CUDA is available
    if torch.cuda.is_available():
        try:
            print("🔴 Releasing allocated GPU Resources")
            if not HAS_AMD:
                nvmlShutdown()
            else:
                rocml.smi_shutdown()
        except Exception as e:
            print(f"Error shutting down NVML: {e}")
    print("🔴 Quitting Transformer Lab API server.")


def parse_args():
    parser = argparse.ArgumentParser(description="FastChat ChatGPT-Compatible RESTful API server.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="host name")
    parser.add_argument("--port", type=int, default=8338, help="port number")
    parser.add_argument("--allow-credentials", action="store_true", help="allow credentials")
    parser.add_argument("--allowed-origins", type=json.loads, default=["*"], help="allowed origins")
    parser.add_argument("--allowed-methods", type=json.loads, default=["*"], help="allowed methods")
    parser.add_argument("--allowed-headers", type=json.loads, default=["*"], help="allowed headers")
    parser.add_argument("--auto_reinstall_plugins", type=bool, default=False, help="auto reinstall plugins")
    parser.add_argument("--https", action="store_true", help="Serve the API over HTTPS with a self-signed cert.")

    return parser.parse_args()


def print_launch_message():
    # Print the welcome message to the CLI
    with open(os.path.join(os.path.dirname(__file__), "transformerlab/launch_header_text.txt"), "r") as f:
        text = f.read()
        shared.print_in_rainbow(text)
    print("http://www.transformerlab.ai\nhttps://github.com/transformerlab/transformerlab-api\n")


def run():
    args = parse_args()

    print(f"args: {args}")
    if args.allowed_origins == ["*"]:
        args.allowed_credentials = False

    app.add_middleware(
        CORSMiddleware,
        allow_origins=args.allowed_origins,
        allow_credentials=args.allow_credentials,
        allow_methods=args.allowed_methods,
        allow_headers=args.allowed_headers,
    )

    if args.https:
        cert_path, key_path = ensure_persistent_self_signed_cert()
        uvicorn.run(
            "api:app", host=args.host, port=args.port, log_level="warning", ssl_certfile=cert_path, ssl_keyfile=key_path
        )
    else:
        uvicorn.run("api:app", host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    run()
