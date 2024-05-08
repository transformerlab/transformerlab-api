import os
import json

from transformerlab.models import basemodel

import huggingface_hub


async def list_models(uninstalled_only: bool = True):
    """
    NOTE: This is only listing locally cached Hugging Face models.
    """

    # Get a list of repos cached in the hugging face hub
    hf_cache_info = huggingface_hub.scan_cache_dir()
    repos=hf_cache_info.repos

    # Cycle through the hugging face repos and add them to the list
    # if they are valid models
    models = []
    for repo in repos:

        # Filter out anything that isn't a model
        if (repo.repo_type != "model"):
            continue

        model = HuggingFaceModel(repo.repo_id)

        # Save ourselves some time if we're only looking for uninstalled models
        installed = await model.is_installed()
        if uninstalled_only and installed:
            continue

        models.append(model)

    return models


class HuggingFaceModel(basemodel.BaseModel):
        
    def __init__(self, hugging_face_id):
        super().__init__(hugging_face_id)

        self.id = hugging_face_id
        self.name = hugging_face_id

        self.model_source = "huggingface"
        self.source_id_or_path = hugging_face_id

        # We need to access the huggingface_hub to figure out more model details
        model_details = {}
        architecture = "unknown"
        private = False
        gated = False

        # Calling huggingface_hub functions can throw a number of exceptions
        try:
            model_details = get_model_details_from_huggingface(hugging_face_id)
            architecture = model_details.get("architecture", "unknown")
            gated = model_details.get("private", False)
            private = model_details.get("gated", False)

        except huggingface_hub.utils.GatedRepoError:
            # Model exists but this user is not on the authorized list
            architecture = "Not Authenticated"
            gated = True

        except huggingface_hub.utils.RepositoryNotFoundError:
            # invalid model ID or private repo without access
            architecture = "Not Authenticated"
            gated = True
            private = True
 
        except huggingface_hub.utils.EntryNotFoundError:
            # This model is missing key configuration information
            print(f"WARNING: {hugging_face_id} missing configuration ")
            print(f"{type(e).__name__}: {e}")

        except Exception as e:
            # Something unexpected happened
            print(f"{type(e).__name__}: {e}")

        # Use the huggingface details to extend json_data
        if (model_details):
            self.json_data.update(model_details)
        else:
            self.json_data["uniqueID"] = hugging_face_id
            self.json_data["name"] = hugging_face_id
            self.json_data["architecture"] = architecture
            self.json_data["private"] = private
            self.json_data["gated"] = gated

        self.architecture = architecture

        # Additional json_data
        # self.json_data["size_on_disk"] = repo.size_on_disk

        # TODO: filename needed for GGUF
        self.model_filename = None




def get_model_details_from_huggingface(hugging_face_id):
    """
    Gets model config details from huggingface_hub
    and return in the format of BaseModel's json_data.
    This is just a helper function for the constructor to make things more readable.

    This function can raise several Exceptions from HuggingFace
    """

    # Use the Hugging Face Hub API to download the config.json file for this model
    # This may throw an exception if the model doesn't exist or we don't have access rights
    huggingface_hub.hf_hub_download(repo_id=hugging_face_id, filename="config.json")

    # Also get model info for metadata and license details
    # Similar to hf_hub_download this can throw exceptions
    # Some models don't have a model card (mostly models that have been deprecated)
    # In that case just set model_card_data to an empty object
    hf_model_info = huggingface_hub.model_info(hugging_face_id)
    try:
        model_card = hf_model_info.card_data
        model_card_data = model_card.data.to_dict()
    except AttributeError:
        model_card_data = {}

    # Use Hugging Face file system API and let it figure out which file we should be reading
    fs = huggingface_hub.HfFileSystem()
    filename = os.path.join(hugging_face_id, "config.json")
    with fs.open(filename) as f:
        filedata = json.load(f)

        # config.json stores a list of architectures but we only store one so just take the first!
        architecture_list = filedata.get("architectures", [])
        architecture = architecture_list[0] if architecture_list else ""

        # Oh except that GGUF and MLX aren't listed as architectures, we have to look in library_name
        library_name = getattr(hf_model_info, "library", "")
        if (library_name == "MLX" or library_name == "GGUF"):
            architecture = library_name

        # TODO: Context length definition seems to vary by architecture. May need conditional logic here.
        context_size = filedata.get("max_position_embeddings", "")

        # TODO: Figure out description, paramters, model size
        newmodel = basemodel.BaseModel(hugging_face_id)
        config = newmodel.json_data
        config = {
            "uniqueID": hugging_face_id,
            "name": filedata.get("name", hugging_face_id),
            "context": context_size,
            "private": getattr(hf_model_info, "private", False),
            "gated": getattr(hf_model_info, "gated", False),
            "architecture": architecture,
            "huggingface_repo": hugging_face_id,
            "model_type": filedata.get("model_type", ""),
            "library_name": library_name,
            "transformers_version": filedata.get("transformers_version", ""),
            "license": model_card_data.get("license", "")
        }
        return config

    # Something did not go to plan
    return None
