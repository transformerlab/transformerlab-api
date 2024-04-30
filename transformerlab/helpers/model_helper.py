import json
import os

from transformerlab.shared import shared
from transformerlab.shared import dirs


def get_default_model_jsondata(model_id: str):
    # This defines the base structure of the json_data 
    # stored in the model DB and used in the TLab code

    return {
        "uniqueID": model_id,
        "name": model_id,
        "description": "",
        "huggingface_repo": "",
        "parameters": "",
        "context": "",
        "architecture": "",
        "license": "",
        "logo": "",

        # The following are from huggingface_hu.hf_api.ModelInfo
        "private": False, 
        "gated": False, # Literal["auto", "manual", False]
        "model_type": "",
        "library_name": "", 
        "transformers_version": ""
    }


##############################
## OLLAMA MODELS
##############################


def get_ollama_models_dir():
    try:
        ollama_dir = os.environ['OLLAMA_MODELS']
    except:
        ollama_dir = os.path.join(os.path.expanduser("~"), ".ollama", "models")

    # Check that the directory actually exists
    if not os.path.isdir(ollama_dir):
        return None

    return ollama_dir


def get_ollama_models_library_dir():
    ollama_models_dir = get_ollama_models_dir()

    if not ollama_models_dir:
        return None

    library_dir = os.path.join(ollama_models_dir, "manifests", "registry.ollama.ai", "library")

    if not os.path.isdir(library_dir):
        return None

    return library_dir


def get_ollama_model(model_id: str):
    model = get_default_model_jsondata(model_id)
    model["supported"] = True
    model["source"] = "ollama"
    model["architecture"] = "GGUF"

    # TODO
    model["installed"] = False
    model["model_path"] = ""
    model["model_filename"] = ""
    model["size_on_disk"] = 0
    return model
