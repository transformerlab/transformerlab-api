import asyncio
from distutils.dir_util import copy_tree
import json
import os
import platform


from fastapi import APIRouter


router = APIRouter(prefix="/plugins", tags=["plugins"])


@router.get("/gallery", summary="Display the plugins available for LLMLab to download.")
async def plugin_gallery():
    """Get list of plugins that we can access"""

    # There are three places that we can get plugins from (to add to an experiment):
    #  1. The remote gallery on the Transformer Lab website
    #  2. The pre-installed local gallery in the transformerlab/plugins folder. These are hadcoded
    #     plugins that are included with LLMLab.
    #  3. The local gallery in the workspace/plugins folder. These are plugins
    #     that the user can create and are made accessible to all experiments

    # For location 2., we will copy all of these plugins to the workspace/plugins folder
    # on startup (see bottom of this file). So we do not need to check this directory here.

    local_workspace_gallery_directory = "workspace/plugins"
    # today the remote gallery is a local file, we will move it remote later
    remote_gallery_file = "transformerlab/galleries/plugin-gallery.json"

    # first get the remote gallery from the remote gallery file:
    with open(remote_gallery_file) as f:
        remote_gallery = json.load(f)

    # now get the local workspace gallery
    workspace_gallery = []
    if os.path.exists(local_workspace_gallery_directory):
        for plugin in os.listdir(local_workspace_gallery_directory):
            if os.path.isdir(os.path.join(local_workspace_gallery_directory, plugin)):
                info = json.load(
                    open(os.path.join(local_workspace_gallery_directory, plugin, "index.json")))

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

    return gallery


@router.get("/gallery/{plugin_id}/install", summary="Install a plugin from the gallery.")
async def install_plugin(plugin_id: str):
    """Install a plugin from the gallery"""
    # For now we assume all gallery plugins are stored at transformerlab/plugins and installed ones go to
    # workspace/plugins

    plugin_path = os.path.join("transformerlab", "plugins", plugin_id)

    # Check if plugin exists at the location:
    if not os.path.exists(plugin_path):
        return {"error": "Plugin not found in gallery."}

    # # Check if plugin is already installed:
    # if os.path.exists(os.path.join("workspace", "plugins", plugin_id)):
    #     return {"error": "Plugin already installed."}

    # Open the Plugin index.json:
    plugin_index_json = open(f"{plugin_path}/index.json", "r")
    plugin_index = json.load(plugin_index_json)

    # create the directory if it doesn't exist
    new_directory = os.path.join("workspace", "plugins", plugin_id)
    if not os.path.exists(new_directory):
        os.makedirs(new_directory)
    # Now copy it to the workspace:
    copy_tree(plugin_path, os.path.join("workspace", "plugins", plugin_id))

    # If index object contains a key called setup-script, run it:
    if "setup-script" in plugin_index:
        # Run shell script
        print("Running Plugin Install script...")
        setup_script_name = plugin_index["setup-script"]
        proc = await asyncio.create_subprocess_exec('/bin/bash', f"{setup_script_name}", cwd=new_directory)
        await proc.wait()
    else:
        print("No install script found")


@router.get("/list", summary="List the plugins that are currently installed.")
async def list_plugins() -> list[object]:
    """Get list of plugins that are currently installed"""

    local_workspace_gallery_directory = "workspace/plugins"

    # now get the local workspace gallery
    workspace_gallery = []
    if os.path.exists(local_workspace_gallery_directory):
        for plugin in os.listdir(local_workspace_gallery_directory):
            if os.path.isdir(os.path.join(local_workspace_gallery_directory, plugin)):
                info = json.load(
                    open(os.path.join(local_workspace_gallery_directory, plugin, "index.json")))
                workspace_gallery.append(info)

    return workspace_gallery


async def missing_platform_plugins() -> list[str]:
    system = platform.system()
    cpu = platform.machine()

    installed_plugins = await list_plugins()
    installed_plugins_names = [plugin["uniqueId"]
                               for plugin in installed_plugins]
    missing_plugins = []

    if (system == "Darwin" and cpu == "arm64"):
        # This is an OSX Machine with Apple Silicon
        mlx_plugins = ["mlx_server", "mlx_exporter", "mlx_lora_trainer"]

        for plugin in mlx_plugins:
            if plugin not in installed_plugins_names:
                missing_plugins.append(plugin)

    if (system == "Linux" and cpu == "x86_64"):
        # This is an Linux Machine with x86_64
        # @TODO fill in soon
        linux_plugins = []

        for plugin in linux_plugins:
            if plugin not in installed_plugins_names:
                missing_plugins.append(plugin)

    return missing_plugins


@router.get("/list_missing_plugins_for_current_platform", summary="Returns true if the default platform plugins are installed.")
async def list_missing_plugins_for_current_platform():
    missing_plugins = await missing_platform_plugins()
    return missing_plugins


@router.get("/install_missing_plugins_for_current_platform", summary="Install the default platform plugins.")
async def install_missing_plugins_for_current_platform():
    missing_plugins = await missing_platform_plugins()

    for plugin in missing_plugins:
        await install_plugin(plugin)
    return missing_plugins

# on startup, copy all plugins from the transformerlab/plugins directory to the workspace/plugins directory
# print("Copying plugins from transformerlab/plugins to workspace/plugins")
# copy_tree("transformerlab/plugins", "workspace/plugins", update=1)
