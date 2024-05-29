import os
import json

from transformerlab.models import basemodel

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
        if uninstalled_only and not installed:
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
                    if uninstalled_only and not installed:
                        models.append(model)


        dirlist.close()

    return models


class LocalFilesystemModel(basemodel.BaseModel):
    def __init__(self, model_path):

        # The ID for this model will be the file or directory name without path
        model_id = os.path.basename(model_path)

        super().__init__(model_id)

        self.id = model_id
        self.name = model_id

        self.model_source = "local"
        self.source_id_or_path = model_path

        architecture = "unknown"
        context_size = ""
        formats = []
        quantization = {}

        # Get model details from configuration file
        config_file = os.path.join(model_path, "config.json")
        try:
            with open(config_file, "r") as f:
                filedata = json.load(f)
                f.close()

                architecture_list = filedata.get("architectures", [])
                architecture = architecture_list[0] if architecture_list else ""
                context_size = filedata.get("max_position_embeddings", "")
                quantization = filedata.get("quantization", {})

                # TODO: Check formats to make sure this is a valid model
                # formats = self._detect_model_formats()

        except FileNotFoundError:
            self.status = "Missing configuration file"
            print(f"WARNING: {self.id} missing configuration")

        except json.JSONDecodeError:
            # Invalid JSON means invlalid model
            self.status = "{self.id} has invalid JSON for configuration"
            print(f"ERROR: Invalid config.json in {model_path}")

        self.architecture = architecture
        self.formats = formats

        self.json_data["context_size"] = context_size
        self.json_data["quantization"] = quantization

        # TODO: This is a HACK! Need to not have two sources for these fields
        self.json_data["architecture"] = self.architecture
        self.json_data["formats"] = self.formats
        self.json_data["source"] = self.model_source
        self.json_data["source_id_or_path"] = self.source_id_or_path
        self.json_data["model_filename"] = self.model_filename


class LocalFilesystemGGUFModel(basemodel.BaseModel):
    def __init__(self, model_path):

        # The ID for this model will be the file or directory name without path
        model_id = os.path.basename(model_path)

        super().__init__(model_id)

        self.id = model_id
        self.name = model_id

        self.model_source = "local"
        self.source_id_or_path = model_path

        # TODO: Pull data from model metadata?
        architecture = "unknown"
        context_size = ""
        formats = []
        quantization = {}

        # Get model details from configuration file
        if os.path.isfile(model_path):
            architecture = "GGUF"
            formats = ["GGUF"]

        self.architecture = architecture
        self.formats = formats
        self.model_filename = model_path

        # TODO: This is a HACK! Need to not have two sources for these fields
        self.json_data["architecture"] = self.architecture
        self.json_data["formats"] = self.formats
        self.json_data["context_size"] = context_size
        self.json_data["quantization"] = quantization
        self.json_data["source"] = self.model_source
        self.json_data["source_id_or_path"] = self.source_id_or_path
        self.json_data["model_filename"] = self.model_filename