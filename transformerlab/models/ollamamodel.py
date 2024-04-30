from transformerlab.models import basemodel

import os


def list_models(uninstalled_only: bool = True):

    ollama_model_library = _get_ollama_models_library_dir()
    if ollama_model_library is None:
        return []

    models = []
    with os.scandir(ollama_model_library) as dirlist:
        # Scan the ollama cache repos for cached models
        # If uninstalled_only is True then skip any models TLab has already
        for entry in dirlist:
            if entry.is_dir():
                ollama_model = OllamaModel(entry.name)

                model_installed = ollama_model.installed
                if (not uninstalled_only or not model_installed):
                    models.append(ollama_model.json_data)

    return models


class OllamaModel(basemodel.BaseModel):
        
    def __init__(self, ollama_id):
        super().__init__(ollama_id)

        self.id = f"ollama/{ollama_id}"
        self.name = f"{ollama_id} - GGUF"
        self.model_store = "ollama"

        # TODO: Figure all of this out
        self.installed = False
        self.model_path = ollama_id

        # TODO: Figure out the localtion of this blob
        self.model_filename = self._get_model_blob_filename()

        # Assume all models from Ollama are GGUF
        self.architecture = "GGUF"
        self.supported = True

        # inherit json_data from the parent and only update specific fields
        self.json_data["uniqueID"] = self.id
        self.json_data["name"] = self.name
        self.json_data["architecture"] = self.architecture


    def _get_model_blob_filename(self):
        return self.id


    def get_path_to_model(self):
        models_dir = _get_ollama_models_dir()
        blobs_dir = os.path.join(models_dir, "blobs")
        return os.path.join(blobs_dir, self.model_filename)
    

def _get_ollama_models_dir():
    try:
        ollama_dir = os.environ['OLLAMA_MODELS']
    except:
        ollama_dir = os.path.join(os.path.expanduser("~"), ".ollama", "models")

    # Check that the directory actually exists
    if not os.path.isdir(ollama_dir):
        return None

    return ollama_dir


def _get_ollama_models_library_dir():
    ollama_models_dir = _get_ollama_models_dir()

    if not ollama_models_dir:
        return None

    library_dir = os.path.join(ollama_models_dir, "manifests", "registry.ollama.ai", "library")

    if not os.path.isdir(library_dir):
        return None

    return library_dir