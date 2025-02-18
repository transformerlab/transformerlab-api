# model_helper - common functions for using models from various sources

import os
import json

import transformerlab.db as db
from transformerlab.shared import dirs
from transformerlab.models import ollamamodel
from transformerlab.models import huggingfacemodel

import traceback

def model_architecture_is_supported(model_architecture: str):
    # Return true if the passed string is a supported model architecture
    # This is a hack and shouldn't be here. We use this to decide if we can import.
    # Just tell the user what the model architecture is and let them import.
    supported_architectures = [
        "GGUF",
        "MLX",
        "CohereForCausalLM",
        "FalconForCausalLM",
        "GemmaForCausalLM",
        "Gemma2ForCausalLM",
        "GPTBigCodeForCausalLM",
        "LlamaForCausalLM",
        "LlavaForConditionalGeneration",
        "MistralForCausalLM",
        "MixtralForCausalLM",
        "PhiForCausalLM",
        "Phi3ForCausalLM",
        "Qwen2ForCausalLM",
        "T5ForConditionalGeneration"

    ]
    return model_architecture in supported_architectures


###
# SHARED MODEL FUNCTIONS


async def list_installed_models():
    """
    This function checks both the DB and the workspace models directory
    and returns a list of models in the format that models are stored in the DB.
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
        print(f"Caught exception getting model {model_source_id} from {model_source}:")
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
