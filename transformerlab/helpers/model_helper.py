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
        ollama_dir = os.path.join(os.getenv("HOME"), ".ollama", "models")
    return ollama_dir


def get_ollama_models_library_dir():
    ollama_models_dir = get_ollama_models_dir()
    return os.path.join(ollama_models_dir, "manifests", "registry.ollama.ai", "library")


def get_ollama_model(model_id: str):
    return get_default_model_jsondata(model_id)