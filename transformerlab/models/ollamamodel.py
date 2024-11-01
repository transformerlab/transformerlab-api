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

                model_installed = await ollama_model.is_installed()
                if (not uninstalled_only or not model_installed):
                    models.append(ollama_model)

    return models


class OllamaModel(basemodel.BaseModel):
        
    def __init__(self, ollama_id):
        super().__init__(ollama_id)

        self.id = f"ollama:{ollama_id}"
        self.name = f"{ollama_id} - GGUF"

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
        if self.status != "OK":
            print(f"Error reading {self.id}: {self.status}")


    # This returns just the filename of the blob containing the actual model
    # If anything goes wrong along the way this returns None
    def _get_model_blob_filename(self):
        # Get the path to the manifest file
        library_dir = ollama_models_library_dir()
        ollamaid = self.json_data.get("source_id_or_path", self.id)

        if not library_dir:
            self.status = f"failed to find ollama library"
            return None

        # Read in the manifest file
        manifestfile = os.path.join(library_dir, ollamaid, "latest")
        try:
            with open(manifestfile, "r") as f:
                filedata = json.load(f)

        except FileNotFoundError:
            print(f"ollama manifest file not found")
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
                    modelfile = layer.get("digest", None)
                    try :
                        with open(modelfile, "r") as f:
                            return modelfile
                    except FileNotFoundError:
                        self.status = f"model file does not exist {modelfile}"
                        return None

            # If we get here it means schemaVersion is 2 but there was no listed model
            self.status = f"no valid ollama.image.model attribute"

        # schemaVersion is not 2. We only support 2.
        self.status = f"unsupported ollama schemaVersion {schemaVersion}"
        return None


    def get_model_path(self):
        model_filename = self.json_data.get("model_filename", "")
        if model_filename:
            return model_filename
        else:
            models_dir = ollama_models_dir()
            blobs_dir = os.path.join(models_dir, "blobs")
            blobfile = self._get_model_blob_filename()
            if blobfile:
                return os.path.join(blobs_dir, blobfile)
            else:
                return None
    

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
