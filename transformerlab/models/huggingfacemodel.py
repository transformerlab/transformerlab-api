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

        # Filter out anything that hasn't actually been downloaded
        # Minor hack: Check repo size and if it's under 10K it's probably just config
        if (repo.size_on_disk < 10000):
            continue

        model = HuggingFaceModel(repo.repo_id)

        # Check if this model is only GGUF files, in which case handle those separately
        gguf_only = (len(model.formats) == 1) and (model.formats[0] == "GGUF")
        if not gguf_only:

            # Regular (i.e. not GGUF only) model
            # Check if it's installed already if we are filtering on that
            installed = await model.is_installed()
            if not uninstalled_only or not installed:
                models.append(model)

        # If this repo is tagged GGUF then it might contain multiple
        # GGUF files each of which is a potential model to import
        if "GGUF" in model.formats:
            # TODO: This requires making a new Model class or using LocalGGUFModel
            # Not trivial given how we currently download GGUF in to workspace/models
            pass

    return models


class HuggingFaceModel(basemodel.BaseModel):
        
    def __init__(self, hugging_face_id):
        super().__init__(hugging_face_id)

        # We need to access the huggingface_hub to figure out more model details
        model_details = {}
        architecture = "unknown"
        formats = []
        private = False
        gated = False

        # Calling huggingface_hub functions can throw a number of exceptions
        try:
            model_details = get_model_details_from_huggingface(hugging_face_id)
            architecture = model_details.get("architecture", "unknown")
            gated = model_details.get("private", False)
            private = model_details.get("gated", False)
            formats = self._detect_model_formats()

        except huggingface_hub.utils.GatedRepoError:
            # Model exists but this user is not on the authorized list
            self.status = "Authentication Required"
            gated = True

        except huggingface_hub.utils.RepositoryNotFoundError:
            # invalid model ID or private repo without access
            self.status = "Model not found"
            gated = True
            private = True
 
        except huggingface_hub.utils.EntryNotFoundError:
            # This model is missing key configuration information
            self.status = "Missing configuration file"
            print(f"WARNING: {hugging_face_id} missing configuration")
            print(f"{type(e).__name__}: {e}")

        except Exception as e:
            # Something unexpected happened
            self.status = str(e)
            print(f"{type(e).__name__}: {e}")

        # Use the huggingface details to extend json_data
        if (model_details):
            self.json_data.update(model_details)
        else:
            self.json_data["uniqueID"] = hugging_face_id
            self.json_data["name"] = hugging_face_id
            self.json_data["private"] = private
            self.json_data["gated"] = gated

        self.architecture = architecture
        self.formats = formats

        # TODO: This is a HACK! Need to not have two sources for these fields
        self.json_data["architecture"] = self.architecture
        self.json_data["formats"] = self.formats
        self.json_data["source"] = "huggingface"
        self.json_data["source_id_or_path"] = hugging_face_id
        self.json_data["model_filename"] = None # TODO: What about GGUF?


    def _detect_model_formats(self):
        """
        Scans the files in the HuggingFace repo to try to determine the format
        of the model.
        """
        # Get a list of files in this model and iterate over them
        source_id_or_path = self.json_data.get("source_id_or_path", self.id)
        try:
            repo_files = huggingface_hub.list_repo_files(source_id_or_path)
        except:
            return []

        detected_formats = []
        for repo_file in repo_files:
            format = basemodel.get_model_file_format(repo_file)

            # If this format isn't in the list already then add it!
            if format and (format not in detected_formats):
                detected_formats.append(format)

        return detected_formats


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
        model_card_data = model_card.to_dict()
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

        # Oh except we list MLX as an architecture but HuggingFace doesn't
        # For MLX it is sometimes stored in library_name
        library_name = getattr(hf_model_info, "library_name", "")
        if (library_name.lower() == "mlx"):
            architecture = "MLX"

        # And sometimes it is stored in the tags for the repo
        model_tags = getattr(hf_model_info, "tags", [])
        print(hugging_face_id)
        if ("mlx" in model_tags):
            architecture = "MLX"

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
            "tags": model_tags,
            "transformers_version": filedata.get("transformers_version", ""),
            "quantization": filedata.get("quantization", ""),
            "license": model_card_data.get("license", "")
        }
        return config

    # Something did not go to plan
    return None
