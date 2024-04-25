import json

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

