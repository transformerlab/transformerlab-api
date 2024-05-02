from transformerlab.models import basemodel

import os


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
        self.json_data["name"] = self.name
        self.json_data["architecture"] = self.architecture


    def _get_model_blob_filename(self):
        #TODO: Pull this from the digest field of the manifest
        return "sha256-00e1317cbf74d901080d7100f57580ba8dd8de57203072dc6f668324ba545f29"


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