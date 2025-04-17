import asyncio
from distutils.dir_util import copy_tree
import json
import os
import platform

import aiofiles
import subprocess
import shutil

from transformerlab.shared import dirs

from werkzeug.utils import secure_filename

from fastapi import APIRouter


router = APIRouter(prefix="/plugins", tags=["plugins"])


@router.get("/gallery", summary="Display the plugins available for LLMLab to download.")
async def plugin_gallery():
    """Get list of plugins that we can access"""

    local_workspace_gallery_directory = dirs.PLUGIN_PRELOADED_GALLERY
    # today the remote gallery is a local file, we will move it remote later
    remote_gallery_file = os.path.join(dirs.TFL_SOURCE_CODE_DIR, "transformerlab/galleries/plugin-gallery.json")

    # first get the remote gallery from the remote gallery file:
    with open(remote_gallery_file) as f:
        remote_gallery = json.load(f)

    # now get the local workspace gallery
    workspace_gallery = []
    if os.path.exists(local_workspace_gallery_directory):
        for plugin in os.listdir(local_workspace_gallery_directory):
            if os.path.isdir(os.path.join(local_workspace_gallery_directory, plugin)):
                try:
                    info = json.load(open(os.path.join(local_workspace_gallery_directory, plugin, "index.json")))
                except Exception as e:
                    print(f"Error loading {plugin} index.json: {e}")

                # These are fields we expect:
                # details = {
                #     "name": info["name"],
                #     "uniqueId": info["uniqueId"],
                #     "description": info["description"],
                #     "type": info["type"],
                #     "version": info["version"],
                #     "url": info["url"],
                #     "icon": info["icon"],
                #     "tags": info["tags"],
                # }
                workspace_gallery.append(info)

    # now combine all the galleries into one
    gallery = workspace_gallery + remote_gallery

    # Now get a list of the plugins that are already installed:
    local_workspace_gallery_directory = dirs.PLUGIN_DIR
    installed_plugins = []
    if os.path.exists(local_workspace_gallery_directory):
        for lp in os.listdir(local_workspace_gallery_directory):
            installed_plugins.append(lp)

    # Now add a field to each plugin in the gallery to indicate if it is installed:
    for plugin in gallery:
        if plugin["uniqueId"] in installed_plugins:
            plugin["installed"] = True
        else:
            plugin["installed"] = False

    # Sort the gallery alphabetically by plugin["name"]
    gallery = sorted(gallery, key=lambda x: x["name"])

    return gallery


async def copy_plugin_files_to_workspace(plugin_id: str):
    plugin_id = secure_filename(plugin_id)

    plugin_path = os.path.join(dirs.PLUGIN_PRELOADED_GALLERY, plugin_id)
    # create the directory if it doesn't exist
    new_directory = os.path.join(dirs.PLUGIN_DIR, plugin_id)
    if not os.path.exists(plugin_path):
        print(f"Plugin {plugin_path} not found in gallery.")
        return
    if not os.path.exists(new_directory):
        os.makedirs(new_directory)
    # Now copy it to the workspace:
    copy_tree(plugin_path, dirs.plugin_dir_by_name(plugin_id))


@router.get("/gallery/{plugin_id}/install", summary="Install a plugin from the gallery.")
async def install_plugin(plugin_id: str):
    """Install a plugin from the gallery"""
    # For now we assume all gallery plugins are stored at transformerlab/plugins and installed ones go to
    # workspace/plugins

    plugin_id = secure_filename(plugin_id)

    plugin_path = os.path.join(dirs.PLUGIN_PRELOADED_GALLERY, plugin_id)

    # Check if plugin exists at the location:
    if not os.path.exists(plugin_path):
        print(f"Plugin {plugin_path} not found in gallery.")
        return {"error": "Plugin not found in gallery."}

    # Open the Plugin index.json:
    plugin_index_json = open(f"{plugin_path}/index.json", "r")
    plugin_index = json.load(plugin_index_json)
    plugin_index_json.close()

    await copy_plugin_files_to_workspace(plugin_id)

    new_directory = os.path.join(dirs.PLUGIN_DIR, plugin_id)
    venv_path = os.path.join(new_directory, "venv")

    global_log_file_name = dirs.GLOBAL_LOG_PATH
    async with aiofiles.open(global_log_file_name, "a") as log_file:
        # Create virtual environment using uv
        print("Creating virtual environment for plugin...")
        await log_file.write(f"## Creating virtual environment for {plugin_id}...\n")

        proc = await asyncio.create_subprocess_exec(
            "uv", "venv", venv_path, "--python", "3.11", cwd=new_directory, stdout=log_file, stderr=log_file
        )
        await proc.wait()

        # Run uv sync after setup script, also with environment activated
        print("Running uv sync to install dependencies...")
        await log_file.write(f"## Running uv sync for {plugin_id}...\n")

        # Use a similar logic
        if check_nvidia_gpu():
            # If we have a GPU, use the requirements file for GPU
            requirements_file_path = os.path.join(os.environ["_TFL_SOURCE_CODE_DIR"], "requirements-uv.txt")
        else:
            # If we don't have a GPU, use the requirements file for CPU
            print("No NVIDIA GPU detected, using CPU requirements file.")
            requirements_file_path = os.path.join(os.environ["_TFL_SOURCE_CODE_DIR"], "requirements-no-gpu-uv.txt")

        proc = await asyncio.create_subprocess_exec(
            "/bin/bash",
            "-c",
            f"source {venv_path}/bin/activate && uv pip sync {requirements_file_path}",
            cwd=new_directory,
            stdout=log_file,
            stderr=log_file,
        )
        await proc.wait()

        # If index object contains a key called setup-script, run it:
        if "setup-script" in plugin_index:
            # Run shell script with virtual environment activated
            print("Running Plugin Install script in virtual environment...")
            await log_file.write(f"## Running setup script for {plugin_id} in virtual environment...\n")

            setup_script_name = plugin_index["setup-script"]
            # Use bash -c to properly source the activation script before running setup script
            proc = await asyncio.create_subprocess_exec(
                "/bin/bash",
                "-c",
                f"source {venv_path}/bin/activate && bash {setup_script_name}",
                cwd=new_directory,
                stdout=log_file,
                stderr=log_file,
            )
            await proc.wait()
        else:
            print("No install script found")
            await log_file.write(f"## No setup script found for {plugin_id}.\n")

        await log_file.write(f"## Plugin Install for {plugin_id} completed.\n")

    print("Plugin installation completed.")

    return {"status": "success", "message": f"Plugin {plugin_id} installed successfully."}


@router.get("/list", summary="List the plugins that are currently installed.")
async def list_plugins() -> list[object]:
    """Get list of plugins that are currently installed"""

    local_workspace_gallery_directory = dirs.PLUGIN_DIR

    # now get the local workspace gallery
    workspace_gallery = []
    if os.path.exists(local_workspace_gallery_directory):
        for plugin in os.listdir(local_workspace_gallery_directory):
            if os.path.isdir(os.path.join(local_workspace_gallery_directory, plugin)):
                index_file = os.path.join(local_workspace_gallery_directory, plugin, "index.json")
                if os.path.isfile(index_file):
                    with open(index_file, "r") as f:
                        info = json.load(f)
                    workspace_gallery.append(info)

    return workspace_gallery


def check_nvidia_gpu() -> bool:
    """
    Check if NVIDIA GPU is available

    Returns:
        tuple: (has_gpu, gpu_info)
            has_gpu: True if NVIDIA GPU is detected, False otherwise
            gpu_info: String with GPU name if detected, empty string otherwise
    """
    has_gpu = False
    gpu_info = ""

    # Check if nvidia-smi is available
    if shutil.which("nvidia-smi") is not None:
        try:
            # Run nvidia-smi to get GPU information
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                check=True,
            )
            gpu_info = result.stdout.strip()

            if gpu_info:
                has_gpu = True
            else:
                print("Nvidia SMI exists, No NVIDIA GPU detected. Perhaps you need to re-install NVIDIA drivers.")
        except subprocess.SubprocessError:
            print("Issue with NVIDIA SMI")

    return has_gpu


async def missing_platform_plugins() -> list[str]:
    system = platform.system()
    cpu = platform.machine()

    installed_plugins = await list_plugins()
    installed_plugins_names = [plugin["uniqueId"] for plugin in installed_plugins]
    missing_plugins = []

    if system == "Darwin" and cpu == "arm64":
        # This is an OSX Machine with Apple Silicon
        mlx_plugins = ["mlx_server", "mlx_exporter", "mlx_lora_trainer"]

        for plugin in mlx_plugins:
            if plugin not in installed_plugins_names:
                missing_plugins.append(plugin)

    if system == "Darwin" and cpu == "x86_64":
        # This is an OSX Machine with x86_64
        osx_plugins = ["ollama_server", "gguf_exporter"]

        for plugin in osx_plugins:
            if plugin not in installed_plugins_names:
                missing_plugins.append(plugin)

    if system == "Linux":
        # This is an Linux Machine, hopefully with a GPU but we could
        # test for that further
        linux_plugins = ["fastchat_server", "llama_trainer", "eleuther-ai-lm-evaluation-harness"]

        for plugin in linux_plugins:
            if plugin not in installed_plugins_names:
                missing_plugins.append(plugin)

    return missing_plugins


@router.get(
    "/list_missing_plugins_for_current_platform",
    summary="Returns a list of plugins that are recommended for the current platform.",
)
async def list_missing_plugins_for_current_platform():
    missing_plugins = await missing_platform_plugins()
    return missing_plugins


@router.get("/install_missing_plugins_for_current_platform", summary="Install the default platform plugins.")
async def install_missing_plugins_for_current_platform():
    missing_plugins = await missing_platform_plugins()

    for plugin in missing_plugins:
        print(f"Installing missing plugin: {plugin}")
        await install_plugin(plugin)
    return missing_plugins


# on startup, copy all plugins from the transformerlab/plugins directory to the workspace/plugins directory
# print("Copying plugins from transformerlab/plugins to workspace/plugins")
# copy_tree("transformerlab/plugins", "workspace/plugins", update=1)
