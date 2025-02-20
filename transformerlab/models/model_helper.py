# model_helper - common functions for using models from various sources

from transformerlab.models import ollamamodel
from transformerlab.models import huggingfacemodel
from transformerlab.models import localmodelstore

import traceback


local_model_store = localmodelstore.LocalModelStore()

###
# SHARED MODEL FUNCTIONS
###


async def list_installed_models():
    """
    This function checks both the DB and the workspace models directory
    and returns a list of models in the format that models are stored in the DB.
    """
    global local_model_store
    return await local_model_store.list_models()


async def is_model_installed(model_id: str):
    """
    Return true is a model with the unique ID model_id
    is downloaded to the DB or Transfomrmer Lab workspace.
    """
    global local_model_store
    local_models = local_model_store.list_models()
    for model in local_models:
        if model["model_id"] == model_id:
            model_downloaded = True
    return model_downloaded


###
# MODEL SOURCE WRAPPER FUNCTIONS
#
# These could probably be rearchitected to use a plugin style model.
# But for now we will wrap the various model sources for convenience.
###


def list_model_sources():
    """
    Supported strings that can be passsed as model_source 
    to the functons that follow.
    """
    return [
        "huggingface",
        "ollama"
    ]


def get_model_by_source_id(model_source: str, model_source_id: str):
    """
    Get a model from a model_source.
    model_source needs to be one of the strings returned by list_model_sources.
    model_source_id is the ID for that model internal to the model_source.
    """

    try:
        match model_source:
            case "ollama":
                return ollamamodel.OllamaModel(model_source_id)
            case "huggingface":
                return huggingfacemodel.HuggingFaceModel(model_source_id)
    except Exception:
        print(
            f"Caught exception getting model {model_source_id} from {model_source}:")
        traceback.print_exc()
    return None


async def list_models_from_source(model_source: str, uninstalled_only: bool = True):
    """
    Get a list of models available at model_source.
    model_source needs to be one of the strings returned by list_model_sources.
    """
    try:
        match model_source:
            case "ollama":
                return await ollamamodel.list_models(uninstalled_only)
            case "huggingface":
                return await huggingfacemodel.list_models(uninstalled_only)
    except Exception:
        print(f"Caught exception listing models from {model_source}:")
        traceback.print_exc()
    return []
