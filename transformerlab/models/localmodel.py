"""
Local models are anything in the workspace/models directory.

This package defines:
LocalModelStore: for listing and getting local models
LocalFilesystemModel and LocalFilesystemGGUFModel classes
"""

import os
import json

from transformerlab.models import basemodel
from transformerlab.models import modelstore
import transformerlab.db as db
from transformerlab.shared import dirs


class LocalModelStore(modelstore.ModelStore):
    """
    Remember the main functions to call are:
    async def list_models()
    async def has_model(model_id):
    """

    def __init__(self):
        super().__init__()

    async def fetch_model_list(self):
        """
        Check both the database and workspace for models.
        """

        # start with the list of downloaded models which is stored in the db
        models = await db.model_local_list()

        # now generate a list of local models by reading the filesystem
        models_dir = dirs.MODELS_DIR

        # now iterate through all the subdirectories in the models directory
        with os.scandir(models_dir) as dirlist:
            for entry in dirlist:
                if entry.is_dir():

                    # Look for model information in info.json
                    info_file = os.path.join(models_dir, entry, "info.json")
                    try:
                        with open(info_file, "r") as f:
                            filedata = json.load(f)
                            f.close()

                            # NOTE: In some places info.json may be a list and in others not
                            # Once info.json format is finalized we can remove this
                            if isinstance(filedata, list):
                                filedata = filedata[0]

                            # tells the app this model was loaded from workspace directory
                            filedata["stored_in_filesystem"] = True

                            # Set local_path to the filesystem location
                            # this will tell Hugging Face to not try downloading
                            filedata["local_path"] = os.path.join(
                                models_dir, entry)

                            # Some models are a single file (possibly of many in a directory, e.g. GGUF)
                            # For models that have model_filename set we should link directly to that specific file
                            if ("model_filename" in filedata and filedata["model_filename"]):
                                filedata["local_path"] = os.path.join(
                                    filedata["local_path"], filedata["model_filename"])

                            models.append(filedata)

                    except FileNotFoundError:
                        # do nothing: just ignore this directory
                        pass

        return models


async def list_models(path: str, uninstalled_only: bool = True):
    """
    This function recursively calls itself to generate a list of models under path.
    First try to determine if this directory is itself a model (and then check
    to see if we can support it). Then search subdirectories for models.
    NOTE: If you pass this a directory with a large tree under it, this can take 
    a long time to run!
    """
    if not os.path.isdir(path):
        return []

    # First decide if this directory is a model
    # Trivially decide this based purely on presence of a configuration file.
    config_file = os.path.join(path, "config.json")
    if os.path.isfile(config_file):
        # TODO Verify that this is something we can support
        model = LocalFilesystemModel(path)

        # Save ourselves some time if we're only looking for uninstalled models
        installed = await model.is_installed()
        if not uninstalled_only or not installed:
            return [model]

    # Otherwise scan this directory for single-file models
    # And then scan subdirectories recursively
    models = []
    with os.scandir(path) as dirlist:
        for entry in dirlist:
            if entry.is_dir():
                models.extend(await list_models(entry.path, uninstalled_only))

            # Use file extension to decide if this is a GGUF model
            if entry.is_file():
                _, fileext = os.path.splitext(entry.path)
                if fileext.lower() == ".gguf" or fileext.lower() == ".ggml":
                    model = LocalFilesystemGGUFModel(entry.path)
                    installed = await model.is_installed()
                    if not uninstalled_only or not installed:
                        models.append(model)

        dirlist.close()

    return models


class LocalFilesystemModel(basemodel.BaseModel):
    def __init__(self, model_path):

        # The ID for this model will be the directory name without path
        model_id = os.path.basename(model_path)

        super().__init__(model_id)

        # model_path is the key piece of data for local models
        self.json_data["source"] = "local"
        self.json_data["source_id_or_path"] = model_path
        self.json_data["model_filename"] = model_path

        # Get model details from configuration file
        config_file = os.path.join(model_path, "config.json")
        try:
            with open(config_file, "r") as f:
                filedata = json.load(f)
                f.close()

                architecture_list = filedata.get("architectures", [])
                self.json_data["architecture"] = architecture_list[0] if architecture_list else ""
                self.json_data["context_size"] = filedata.get(
                    "max_position_embeddings", "")
                self.json_data["quantization"] = filedata.get(
                    "quantization", {})

                # TODO: Check formats to make sure this is a valid model?

        except FileNotFoundError:
            self.status = "Missing configuration file"
            print(f"WARNING: {self.id} missing configuration")

        except json.JSONDecodeError:
            # Invalid JSON means invlalid model
            self.status = "{self.id} has invalid JSON for configuration"
            print(f"ERROR: Invalid config.json in {model_path}")


class LocalFilesystemGGUFModel(basemodel.BaseModel):
    def __init__(self, model_path):

        # The ID for this model will be the filename without path
        model_id = os.path.basename(model_path)

        super().__init__(model_id)

        # Get model details from configuration file
        if os.path.isfile(model_path):
            architecture = "GGUF"
            formats = ["GGUF"]
        else:
            self.status = f"Invalid GGUF model: {model_path}"
            architecture = "unknown"
            formats = []

        # TODO: This is a HACK! Need to not have two sources for these fields
        self.json_data["architecture"] = architecture
        self.json_data["formats"] = formats
        self.json_data["source"] = "local"
        self.json_data["source_id_or_path"] = model_path
        self.json_data["model_filename"] = model_path
