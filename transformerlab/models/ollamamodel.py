"""
This package defines OllamaModel which represents a model that is stored
in the Ollama cache and can be imported into Transformer Lab.
"""

from transformerlab.models import basemodel

import os
import json


async def list_models():
    ollama_model_library = ollama_models_library_dir()
    if ollama_model_library is None:
        return []

    models = []
    with os.scandir(ollama_model_library) as dirlist:
        # Scan the ollama cache repos for cached models
        for entry in dirlist:
            if entry.is_dir():
                ollama_model = OllamaModel(entry.name)
                models.append(ollama_model)

    return models


class OllamaModel(basemodel.BaseModel):
    """
    Wrapper for models imported from Ollama.
    These models are kept in the ollama cache (usually ~/.ollama)
    """

    def __init__(self, ollama_id):
        super().__init__(ollama_id)

        # A convention in Transformer Lab is that GGUF models are named
        # modelname.gguf. Most models in ollama will not have the gguf part.
        file_name, file_extension = os.path.splitext(ollama_id)
        if file_extension == ".gguf":
            self.id = f"{ollama_id}"
        else:
            self.id = f"{ollama_id}.gguf"
        self.name = f"{ollama_id} (ollama)"

        # inherit json_data from the parent and only update specific fields
        self.json_data["uniqueID"] = self.id
        self.json_data["name"] = self.name

        # Assume all models from ollama are GGUF
        self.json_data["architecture"] = "GGUF"
        self.json_data["formats"] = ["GGUF"]
        self.json_data["source"] = "ollama"
        self.json_data["source_id_or_path"] = ollama_id

        # NOTE: This can change self.status if there's an error
        self.json_data["model_filename"] = self.get_model_path()

    def _get_model_blob_filename(self):
        """
        This returns just the filename of the blob containing the actual model
        If anything goes wrong along the way this returns None
        """

        # Get the path to the manifest file
        library_dir = ollama_models_library_dir()
        ollamaid = self.json_data.get("source_id_or_path", self.id)

        if not library_dir:
            self.status = "failed to find ollama library"
            return None

        # Read in the manifest file
        manifestfile = os.path.join(library_dir, ollamaid, "latest")
        try:
            with open(manifestfile, "r") as f:
                filedata = json.load(f)

        except FileNotFoundError:
            print("ollama manifest file not found")
            return None

        # The format of v2 schema is that there is a list called "layers"
        # Objects in layers have data on the files in the blobs directory
        # those files can be model, license, template, params
        # we are after the model file
        schemaVersion = filedata.get("schemaVersion", None)
        if schemaVersion == 2:
            layers = filedata.get("layers", [])
            for layer in layers:
                # Each layer has a mediaType field describing what the file contains
                # and a digest field with the name of the file
                if layer.get("mediaType", None) == "application/vnd.ollama.image.model":
                    # Check if the specified file exists or not!
                    digestvalue = layer.get("digest", None)
                    models_dir = ollama_models_dir()
                    blobs_dir = os.path.join(models_dir, "blobs")

                    # ollama lists the file with a ":" that needs to be converted to a "-"
                    modelfile = digestvalue.replace(":", "-")
                    model_path = os.path.join(blobs_dir, modelfile)
                    try:
                        with open(model_path, "r") as f:
                            return model_path
                    except FileNotFoundError:
                        self.status = f"model file does not exist {modelfile}"
                        return None

            # If we get here it means schemaVersion is 2 but there was no listed model
            self.status = "no valid ollama.image.model attribute"

        # schemaVersion is not 2. We only support 2.
        self.status = f"unsupported ollama schemaVersion {schemaVersion}"
        return None

    def get_model_path(self):
        model_filename = self.json_data.get("model_filename", "")
        if model_filename:
            return model_filename
        else:
            blobfile = self._get_model_blob_filename()
            if blobfile:
                return blobfile
            else:
                return None


#########################
#  DIRECTORY STRUCTURE  #
#########################


def ollama_models_dir():
    try:
        ollama_dir = os.environ["OLLAMA_MODELS"]
    except KeyError:
        ollama_dir = os.path.join(os.path.expanduser("~"), ".ollama", "models")

    # Check that the directory actually exists
    if not os.path.isdir(ollama_dir):
        return None

    return ollama_dir


def ollama_models_library_dir():
    models_dir = ollama_models_dir()

    if not models_dir:
        return None

    library_dir = os.path.join(
        models_dir, "manifests", "registry.ollama.ai", "library")

    if not os.path.isdir(library_dir):
        return None

    return library_dir
