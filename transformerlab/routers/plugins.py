from distutils.dir_util import copy_tree
import json
import os


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


# on startup, copy all plugins from the transformerlab/plugins directory to the workspace/plugins directory
print("Copying plugins from transformerlab/plugins to workspace/plugins")
copy_tree("transformerlab/plugins", "workspace/plugins", update=1)
