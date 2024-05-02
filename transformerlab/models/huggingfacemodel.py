import os

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

        # TODO: How to figure out architecture?
        self.architecture = "GGUF"

        self.model_source = "huggingface"
        self.source_id_or_path = hugging_face_id

        # TODO: This function is currently in the model router.
        # Move/copy it here. Deal with the auth stuff
        #
        # model_details = {}
        #    try:
        #        model_details = get_model_details_from_huggingface(model_id)
        #        self.architecture = model_details.get("architecture", "unknown")
        #    except GatedRepoError:
        #        architecture = "Not Authenticated"
        #    except Exception as e:
        #        print(f"{type(e).__name__}: {e}")
        #        architecture = "unknown"
        # Additional json_data
        # self.json_data["size_on_disk"] = repo.size_on_disk

        # TODO: filename needed for GGUF
        self.model_filename = None

        # inherit json_data from the parent and only update specific fields
        self.json_data["uniqueID"] = self.id
        self.json_data["name"] = self.name
        self.json_data["architecture"] = self.architecture
