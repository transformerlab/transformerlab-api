"""
The Entrypoint File for Transformer Lab's API Server.
"""

import os
import argparse
import asyncio
import shutil

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

from transformerlab.db.jobs import job_create, job_update_status
from transformerlab.db import jobs as db_jobs
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
    recipes,
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
from transformerlab.shared import shared
from transformerlab.shared import galleries
from lab.dirs import get_workspace_dir, get_jobs_dir
from lab import dirs as lab_dirs, Experiment, Job
from transformerlab.shared import dirs
from transformerlab.db.filesystem_migrations import migrate_datasets_table_to_filesystem, migrate_models_table_to_filesystem
from transformerlab.shared.request_context import set_current_org_id
from lab.dirs import set_organization_id as lab_set_org_id

from dotenv import load_dotenv
from datetime import datetime

JOBS_DIR = get_jobs_dir()
load_dotenv()

# The following environment variable can be used by other scripts
# who need to connect to the root DB, for example
os.environ["LLM_LAB_ROOT_PATH"] = dirs.ROOT_DIR
# environment variables that start with _ are
# used internally to set constants that are shared between separate processes. They are not meant to be
# to be overriden by the user.
os.environ["_TFL_SOURCE_CODE_DIR"] = dirs.TFL_SOURCE_CODE_DIR
# The temporary image directory for transformerlab (default; per-request overrides computed in routes)
temp_image_dir = os.path.join(get_workspace_dir(), "temp", "images")
os.environ["TLAB_TEMP_IMAGE_DIR"] = str(temp_image_dir)

async def migrate_jobs():
    """Migrate jobs from DB to filesystem"""
    try:
        # Late import to avoid hard dependency during tests without DB
        from transformerlab.db.session import async_session
        from sqlalchemy import text as sqlalchemy_text

        print("Migrating jobs...")
        
        # Read existing job rows from DB using raw SQL (like dataset migration)
        jobs_rows = []
        experiments_map = {}  # Map experiment_id to experiment_name
        try:
            # First check if the jobs table exists
            async with async_session() as session:
                result = await session.execute(
                    sqlalchemy_text("SELECT name FROM sqlite_master WHERE type='table' AND name='job'")
                )
                jobs_table_exists = result.fetchone() is not None
                
                # Also check if experiments table exists to get the name mapping
                result = await session.execute(
                    sqlalchemy_text("SELECT name FROM sqlite_master WHERE type='table' AND name='experiment'")
                )
                experiments_table_exists = result.fetchone() is not None
            
            if not jobs_table_exists:
                print("No jobs table found, skipping jobs migration.")
                return
                
            # Get experiments mapping first (can't use experiment_get_all() as it might be deleted)
            if experiments_table_exists:
                async with async_session() as session:
                    result = await session.execute(sqlalchemy_text("SELECT * FROM experiment"))
                    experiments = result.mappings().all()
                    for exp in experiments:
                        # Ensure consistent string keys for mapping
                        experiments_map[str(exp['id'])] = exp['name']            
            
            # Get all jobs using raw SQL (can't use jobs_get_by_experiment() as it might be deleted)
            async with async_session() as session:
                result = await session.execute(sqlalchemy_text("SELECT * FROM job"))
                jobs = result.mappings().all()
                dict_jobs = [dict(job) for job in jobs]
                for job in dict_jobs:
                    # Handle job_data JSON inconsistency (like dataset migration)
                    if "job_data" in job and job["job_data"]:
                        if isinstance(job["job_data"], str):
                            try:
                                job["job_data"] = json.loads(job["job_data"])
                            except json.JSONDecodeError:
                                job["job_data"] = {}
                    jobs_rows.append(job)
        except Exception as e:
            print(f"Failed to read jobs for migration: {e}")
            jobs_rows = []

        if not jobs_rows:
            print("No jobs found in DB to migrate.")
            return

        # Move existing jobs directory to temp if it exists
        # We do this because the SDK's create() method will fail if directories already exist, so we temporarily move
        # the existing directories aside, let the SDK create clean directories with proper structure,
        # then copy back all the existing files (preserving user data like logs, configs, etc.)
        temp_jobs_dir = None
        if os.path.exists(JOBS_DIR):
            temp_jobs_dir = f"{JOBS_DIR}_migration_temp"
            print(f"Moving existing jobs directory to: {temp_jobs_dir}")
            os.rename(JOBS_DIR, temp_jobs_dir)

        migrated = 0
        for job in jobs_rows:
            # Get experiment name from mapping
            experiment_id = job.get('experiment_id')
            experiment_name = experiments_map.get(str(experiment_id), 'unknown')
            
            try:
                # Create SDK Job
                job_obj = Job.create(job['id'])
                # Update the JSON data with DB data
                job_obj._update_json_data_field(key="id", value=job["id"])
                job_obj._update_json_data_field(key="experiment_id", value=experiment_name)  # Use name instead of numeric ID
                job_obj._update_json_data_field(key="job_data", value=job.get("job_data", {}))
                job_obj._update_json_data_field(key="status", value=job["status"])
                job_obj._update_json_data_field(key="type", value=job["type"])
                job_obj._update_json_data_field(key="progress", value=job.get("progress"))
                job_obj._update_json_data_field(key="created_at", value=job.get("created_at"))
                job_obj._update_json_data_field(key="updated_at", value=job.get("updated_at"))
                
                # Copy existing files from temp directory if they exist
                # This preserves all user data (logs, configs, outputs, etc.) that was in the
                # original job directories while maintaining the new SDK structure
                if temp_jobs_dir:
                    old_job_dir = os.path.join(temp_jobs_dir, str(job['id']))
                    if os.path.exists(old_job_dir):
                        new_job_dir = job_obj.get_dir()
                        # Copy all files except index.json (which we just created)
                        for item in os.listdir(old_job_dir):
                            src = os.path.join(old_job_dir, item)
                            dst = os.path.join(new_job_dir, item)
                            if os.path.isdir(src):
                                shutil.copytree(src, dst, dirs_exist_ok=True)
                            else:
                                shutil.copy2(src, dst)
                    else:
                        # Job not found in jobs directory, check if it's in the wrong place
                        # (experiments/{experiment_name}/jobs/{job_id}) from the last month
                        temp_experiments_dir = f"{lab_dirs.get_experiments_dir()}_migration_temp"
                        if os.path.exists(temp_experiments_dir):
                            wrong_place_job_dir = os.path.join(temp_experiments_dir, str(experiment_name), "jobs", str(job['id']))
                            if os.path.exists(wrong_place_job_dir):
                                new_job_dir = job_obj.get_dir()
                                # Copy all files except index.json (which we just created)
                                for item in os.listdir(wrong_place_job_dir):
                                    src = os.path.join(wrong_place_job_dir, item)
                                    dst = os.path.join(new_job_dir, item)
                                    if os.path.isdir(src):
                                        shutil.copytree(src, dst, dirs_exist_ok=True)
                                    else:
                                        shutil.copy2(src, dst)
                
                migrated += 1
            except Exception:
                # Best-effort migration; continue
                continue

        # Clean up temp directory
        if temp_jobs_dir and os.path.exists(temp_jobs_dir):
            print(f"Cleaning up temp jobs directory: {temp_jobs_dir}")
            shutil.rmtree(temp_jobs_dir)
        
        # Clean up temp experiments directory if it was used for job migration
        temp_experiments_dir = f"{lab_dirs.get_experiments_dir()}_migration_temp"
        if os.path.exists(temp_experiments_dir):
            print(f"Cleaning up temp experiments directory after job migration: {temp_experiments_dir}")
            shutil.rmtree(temp_experiments_dir)

        # Archive the legacy jobs table if present (like dataset migration)
        try:
            async with async_session() as session:
                await session.execute(sqlalchemy_text("ALTER TABLE job RENAME TO zzz_archived_job"))
                await session.commit()
        except Exception:
            pass

        if migrated:
            print(f"Jobs migration completed: {migrated} entries migrated to filesystem store.")
            
    except Exception as e:
        # Do not block startup on migration issues
        print(f"Jobs migration skipped due to error: {e}")

async def migrate_experiments():
    """Migrate experiments from DB to filesystem following the dataset migration pattern."""
    try:
        # Late import to avoid hard dependency during tests without DB
        from transformerlab.db.session import async_session
        from sqlalchemy import text as sqlalchemy_text

        print("Migrating experiments...")
        
        # Read existing experiment rows from DB using raw SQL (like dataset migration)
        experiments_rows = []
        try:
            # First check if the experiments table exists
            async with async_session() as session:
                result = await session.execute(
                    sqlalchemy_text("SELECT name FROM sqlite_master WHERE type='table' AND name='experiment'")
                )
                exists = result.fetchone() is not None
            if not exists:
                print("No experiments table found, skipping experiments migration.")
                return
                
            # Get all experiments using raw SQL (can't use experiment_get_all() as it might be deleted)
            async with async_session() as session:
                result = await session.execute(sqlalchemy_text("SELECT * FROM experiment"))
                experiments = result.mappings().all()
                dict_experiments = [dict(experiment) for experiment in experiments]
                for exp in dict_experiments:
                    # Handle config JSON inconsistency (like dataset migration)
                    if "config" in exp and exp["config"]:
                        if isinstance(exp["config"], str):
                            try:
                                exp["config"] = json.loads(exp["config"])
                            except json.JSONDecodeError:
                                exp["config"] = {}
                    experiments_rows.append(exp)
        except Exception as e:
            print(f"Failed to read experiments for migration: {e}")
            experiments_rows = []

        if not experiments_rows:
            print("No experiments found in DB to migrate.")
            return

        # Move existing experiments directory to temp if it exists
        # We do this because the SDK's create() method will fail if 
        # directories already exist, so we temporarily move the existing directories aside, let the 
        # SDK create clean directories with proper structure, then copy back all the existing files 
        # (preserving user data like models, datasets, configs, etc.)
        temp_experiments_dir = None
        experiments_dir = lab_dirs.get_experiments_dir()
        if os.path.exists(experiments_dir):
            temp_experiments_dir = f"{experiments_dir}_migration_temp"
            print(f"Moving existing experiments directory to: {temp_experiments_dir}")
            os.rename(experiments_dir, temp_experiments_dir)

        migrated = 0
        for exp in experiments_rows:
            try:
                # Create SDK Experiment
                experiment = Experiment.create(exp['name'])
                # Update the JSON data with DB data
                experiment._update_json_data_field(key="id", value=exp["name"])
                experiment._update_json_data_field(key="db_experiment_id", value=exp["id"])
                experiment._update_json_data_field(key="config", value=exp.get("config", {}))
                experiment._update_json_data_field(key="created_at", value=exp.get("created_at", datetime.now().isoformat()))
                experiment._update_json_data_field(key="updated_at", value=exp.get("updated_at", datetime.now().isoformat()))
                
                # Copy existing files from temp directory if they exist
                # This preserves all user data (models, datasets, configs, etc.) that was in the
                # original experiment directories while maintaining the new SDK structure
                if temp_experiments_dir:
                    old_experiment_dir = os.path.join(temp_experiments_dir, exp['name'])
                    if os.path.exists(old_experiment_dir):
                        new_experiment_dir = experiment.get_dir()
                        for item in os.listdir(old_experiment_dir):
                            src = os.path.join(old_experiment_dir, item)
                            dst = os.path.join(new_experiment_dir, item)
                            if os.path.isdir(src):
                                shutil.copytree(src, dst, dirs_exist_ok=True)
                            else:
                                shutil.copy2(src, dst)
                
                migrated += 1
            except Exception:
                # Best-effort migration; continue
                continue

        # Clean up temp directory
        if temp_experiments_dir and os.path.exists(temp_experiments_dir):
            print(f"Cleaning up temp experiments directory: {temp_experiments_dir}")
            shutil.rmtree(temp_experiments_dir)

        # Archive the legacy experiments table if present (like dataset migration)
        try:
            async with async_session() as session:
                await session.execute(sqlalchemy_text("ALTER TABLE experiment RENAME TO zzz_archived_experiment"))
                await session.commit()
        except Exception:
            pass

        if migrated:
            print(f"Experiments migration completed: {migrated} entries migrated to filesystem store.")
            
    except Exception as e:
        # Do not block startup on migration issues
        print(f"Experiments migration skipped due to error: {e}")

async def migrate_job_and_experiment_to_filesystem():
    """Migrate data from DB to filesystem if not already migrated."""
    
    try: 
        # Late import to avoid hard dependency during tests without DB
        from transformerlab.db.session import async_session
        from sqlalchemy import text as sqlalchemy_text

        # Check if migration is needed by looking for the existence of old database tables
        experiments_need_migration = False
        jobs_need_migration = False
        
        try:
            async with async_session() as session:
                # Check if experiments table exists
                result = await session.execute(
                    sqlalchemy_text("SELECT name FROM sqlite_master WHERE type='table' AND name='experiment'")
                )
                experiments_need_migration = result.fetchone() is not None
                
                # Check if jobs table exists  
                result = await session.execute(
                    sqlalchemy_text("SELECT name FROM sqlite_master WHERE type='table' AND name='job'")
                )
                jobs_need_migration = result.fetchone() is not None
        except Exception as e:
            print(f"Failed to check for migration tables: {e}")
            return
        
        # If neither needs migration, skip entirely
        if not experiments_need_migration and not jobs_need_migration:
            print("No migration needed - tables not found.")
            return
        else:
            if experiments_need_migration:
                await migrate_experiments()
            if jobs_need_migration:
                await migrate_jobs()
                
    except Exception as e:
        print(f"Error during migration: {e}")
        # Do not block startup on migration issues
        pass


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
    # run the migrations
    asyncio.create_task(migrate())
    asyncio.create_task(migrate_models_table_to_filesystem())
    asyncio.create_task(migrate_datasets_table_to_filesystem())
    asyncio.create_task(migrate_job_and_experiment_to_filesystem())
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
    # Restrict origins so credentialed requests (cookies) are allowed
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Middleware to set context var for organization id per request (multitenant)
@app.middleware("http")
async def set_org_context(request: Request, call_next):
    try:
        org_id = None
        if os.getenv("TFL_MULTITENANT") == "true":
            org_cookie_name = os.getenv("AUTH_ORGANIZATION_COOKIE_NAME", "tlab_org_id")
            org_id = request.cookies.get(org_cookie_name)
        set_current_org_id(org_id)
        if lab_set_org_id is not None:
            lab_set_org_id(org_id)
        response = await call_next(request)
        return response
    finally:
        # Clear at end of request
        set_current_org_id(None)
        if lab_set_org_id is not None:
            lab_set_org_id(None)


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

# Authentication and session management routes
if os.getenv("TFL_MULTITENANT") == "true":
    from transformerlab.routers import auth  # noqa: E402

    app.include_router(auth.router)


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
    experiment_id: str = None,
    inference_params: str = "",
    request: Request = None,
):
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
    plugin_location = lab_dirs.plugin_dir_by_name(plugin_name)

    model = model_name
    if model_filename is not None and model_filename != "":
        model = model_filename

    if adaptor != "":
        # Resolve per-request workspace if multitenant
        workspace_dir = get_workspace_dir()
        adaptor = f"{workspace_dir}/adaptors/{secure_filename(model)}/{adaptor}"

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

    with open(lab_dirs.GLOBAL_LOG_PATH, "a") as global_log:
        global_log.write(f"🏃 Loading Inference Server for {model_name} with {inference_params}\n")

    process = await shared.async_run_python_daemon_and_update_status(
        python_script=params,
        job_id=job_id,
        begin_string="Application startup complete.",
        set_process_id_function=set_worker_process_id,
    )
    exitcode = process.returncode
    if exitcode == 99:
        with open(lab_dirs.GLOBAL_LOG_PATH, "a") as global_log:
            global_log.write(
                "GPU (CUDA) Out of Memory: Please try a smaller model or a different inference engine. Restarting the server may free up resources.\n"
            )
        return {
            "status": "error",
            "message": "GPU (CUDA) Out of Memory: Please try a smaller model or a different inference engine. Restarting the server may free up resources.",
        }
    if exitcode is not None and exitcode != 0:
        with open(lab_dirs.GLOBAL_LOG_PATH, "a") as global_log:
            global_log.write(f"Error loading model: {model_name} with exit code {exitcode}\n")
        job = await db_jobs.job_get(job_id)
        error_msg = None
        if job and job.get("job_data"):
            error_msg = job["job_data"].get("error_msg")
        if not error_msg:
            error_msg = f"Exit code {exitcode}"
            await job_update_status(job_id, "FAILED", experiment_id=experiment_id, error_msg=error_msg)
        return {"status": "error", "message": error_msg}
    with open(lab_dirs.GLOBAL_LOG_PATH, "a") as global_log:
        global_log.write(f"Model loaded successfully: {model_name}\n")
    return {"status": "success", "job_id": job_id}


@app.get("/server/worker_stop", tags=["serverinfo"])
async def server_worker_stop():
    global worker_process
    print(f"Stopping worker process: {worker_process}")
    if worker_process is not None:
        try:
            os.kill(worker_process.pid, signal.SIGTERM)
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

    # Wait a bit for the worker to fully terminate
    await asyncio.sleep(1)

    # Refresh the controller to remove the stopped worker immediately
    try:
        async with httpx.AsyncClient() as client:
            await client.post(fastchat_openai_api.app_settings.controller_address + "/refresh_all_workers")
    except Exception as e:
        print(f"Error refreshing controller after stopping worker: {e}")

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
                try:
                    os.kill(int(pid), signal.SIGTERM)
                except ProcessLookupError:
                    print(f"Process {pid} doesn't exist so nothing to kill")
                except Exception as e:
                    print(f"Error killing process {pid}: {e}")
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
