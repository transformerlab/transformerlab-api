from transformerlab.models import basemodel

import os
import json


async def list_models(uninstalled_only: bool = True):

    ollama_model_library = ollama_models_library_dir()
    if ollama_model_library is None:
        return []

    models = []
    with os.scandir(ollama_model_library) as dirlist:
        # Scan the ollama cache repos for cached models
        # If uninstalled_only is True then skip any models TLab has already
        for entry in dirlist:
            if entry.is_dir():
                ollama_model = OllamaModel(entry.name)

                # TODO: Create a function to check if this is installed
                model_installed = await ollama_model.is_installed()
                if (not uninstalled_only or not model_installed):
                    models.append(ollama_model)

    return models


class OllamaModel(basemodel.BaseModel):
        
    def __init__(self, ollama_id):
        super().__init__(ollama_id)

        self.id = f"ollama/{ollama_id}"
        self.name = f"{ollama_id} - GGUF"

        # Assume all models from Ollama are GGUF
        self.architecture = "GGUF"

        self.model_source = "ollama"
        self.source_id_or_path = ollama_id

        # TODO: Figure out the localtion of this blob
        self.model_filename = self._get_model_blob_filename()

        # inherit json_data from the parent and only update specific fields
        self.json_data["uniqueID"] = self.id
        self.json_data["model_filename"] = self.get_path_to_model()
        self.json_data["name"] = self.name
        self.json_data["architecture"] = self.architecture


    # This returns just the filename of the blob containing the actual model
    # If anything goes wrong along the way this returns None
    def _get_model_blob_filename(self):
        # Get the path to the manifest file
        library_dir = ollama_models_library_dir()
        if not library_dir or not self.source_id_or_path:
            return None

        # Read in the manifest file
        manifestfile = os.path.join(library_dir, self.source_id_or_path, "latest")
        try:
            with open(manifestfile, "r") as f:
                filedata = json.load(f)
                f.close()
        except FileNotFoundError:
            return None

        # The format of v2 schema is that there is a list called "layers"
        # Objects in layers have data on the files in the blobs directory
        # those files can be model, license, template, params
        # we are after the model file
        if filedata.get("schemaVersion", None) == 2:
            layers = filedata.get("layers", [])
            for layer in layers:

                # Each layer has a mediaType field describing what the file contains
                # and a digest field with the name of the file
                if layer.get("mediaType", None) == "application/vnd.ollama.image.model":
                    return layer.get("digest", None)

        return None


    def get_path_to_model(self):
        models_dir = ollama_models_dir()
        blobs_dir = os.path.join(models_dir, "blobs")
        return os.path.join(blobs_dir, self.model_filename)
    

#########################
#  DIRECTORY STRUCTURE  #
#########################

def ollama_models_dir():
    try:
        ollama_dir = os.environ['OLLAMA_MODELS']
    except:
        ollama_dir = os.path.join(os.path.expanduser("~"), ".ollama", "models")

    # Check that the directory actually exists
    if not os.path.isdir(ollama_dir):
        return None

    return ollama_dir


def ollama_models_library_dir():
    models_dir = ollama_models_dir()

    if not models_dir:
        return None

    library_dir = os.path.join(models_dir, "manifests", "registry.ollama.ai", "library")

    if not os.path.isdir(library_dir):
        return None

    return library_dir