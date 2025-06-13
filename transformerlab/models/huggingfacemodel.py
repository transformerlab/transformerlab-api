"""
This package defines HuggingFaceModel and functions for interacting with models
in the Hugging Face hub local cache.
"""

import os
import json
import fnmatch
import shutil

from transformerlab.models import basemodel

import huggingface_hub
from huggingface_hub.hf_api import RepoFile
from huggingface_hub import scan_cache_dir


async def list_models():
    """
    NOTE: This is only listing locally cached Hugging Face models.
    """

    # Get a list of repos cached in the hugging face hub
    hf_cache_info = huggingface_hub.scan_cache_dir()
    repos = hf_cache_info.repos

    # Cycle through the hugging face repos and add them to the list
    # if they are valid models
    models = []
    for repo in repos:
        # Filter out anything that isn't a model
        if repo.repo_type != "model":
            continue

        # Filter out anything that hasn't actually been downloaded
        # Minor hack: Check repo size and if it's under 10K it's probably just config
        if repo.size_on_disk < 10000:
            continue

        model = HuggingFaceModel(repo.repo_id)

        # Check if this model is only GGUF files, in which case handle those separately
        # TODO: Need to handle GGUF Repos separately. But DO NOT read in the full JSON
        # for this repo or it will be too slow.
        formats = []
        gguf_only = (len(formats) == 1) and (formats[0] == "GGUF")
        if not gguf_only:
            # Regular (i.e. not GGUF only) model
            models.append(model)

        # If this repo is tagged GGUF then it might contain multiple
        # GGUF files each of which is a potential model to import
        if "GGUF" in formats:
            # TODO: This requires making a new Model class or using LocalGGUFModel
            # Not trivial given how we currently download GGUF in to workspace/models
            print("Skipping GGUF repo", repo.repo_id)
            pass

    return models


class HuggingFaceModel(basemodel.BaseModel):
    def __init__(self, hugging_face_id):
        super().__init__(hugging_face_id)

        self.source = "huggingface"
        self.source_id_or_path = hugging_face_id

    async def get_json_data(self):
        json_data = await super().get_json_data()

        # We need to access the huggingface_hub to figure out more model details
        # We'll get details and merge them with our json_data
        # Calling huggingface_hub functions can throw a number of exceptions
        model_details = {}
        private = False
        gated = False
        try:
            model_details = await get_model_details_from_huggingface(self.id)
            json_data["formats"] = self._detect_model_formats()

        except huggingface_hub.utils.GatedRepoError:
            # Model exists but this user is not on the authorized list
            self.status = "Authentication Required"
            gated = True

        except huggingface_hub.utils.RepositoryNotFoundError:
            # invalid model ID or private repo without access
            self.status = "Model not found"
            gated = True
            private = True

        except huggingface_hub.utils.EntryNotFoundError as e:
            # This model is missing key configuration information
            self.status = "Missing configuration file"
            print(f"WARNING: {self.id} missing configuration")
            print(f"{type(e).__name__}: {e}")

        except Exception as e:
            # Something unexpected happened
            self.status = str(e)
            print(f"{type(e).__name__}: {e}")

        # Use the huggingface details to extend json_data
        if model_details:
            json_data.update(model_details)
        else:
            json_data["uniqueID"] = self.id
            json_data["name"] = self.id
            json_data["private"] = private
            json_data["gated"] = gated

        return json_data

    def _detect_model_formats(self):
        """
        Scans the files in the HuggingFace repo to try to determine the format
        of the model.
        """
        # Get a list of files in this model and iterate over them
        try:
            repo_files = huggingface_hub.list_repo_files(self.id)
        except Exception:
            return []

        detected_formats = []
        for repo_file in repo_files:
            format = basemodel.get_model_file_format(repo_file)

            # If this format isn't in the list already then add it!
            if format and (format not in detected_formats):
                detected_formats.append(format)

        return detected_formats


async def get_model_details_from_huggingface(hugging_face_id: str):
    """
    Gets model config details from huggingface_hub
    and return in the format of BaseModel's json_data.
    This is just a helper function for the constructor to make things more readable.

    This function can raise several Exceptions from HuggingFace
    """

    # Get model info for metadata and license details
    # Similar to hf_hub_download this can throw exceptions
    # Some models don't have a model card (mostly models that have been deprecated)
    # In that case just set model_card_data to an empty object
    hf_model_info = huggingface_hub.model_info(hugging_face_id)
    try:
        model_card = hf_model_info.card_data
        model_card_data = model_card.to_dict()
    except AttributeError:
        model_card_data = {}

    # Detect SD model by tags or by presence of model_index.json
    model_tags = getattr(hf_model_info, "tags", [])
    print("Model tags:", model_tags)
    is_sd = False
    if any("stable-diffusion" in t or "diffusers" in t for t in model_tags):
        is_sd = True
    try:
        repo_files = huggingface_hub.list_repo_files(hugging_face_id)
        if any(f.endswith("model_index.json") for f in repo_files):
            is_sd = True
    except Exception:
        repo_files = []

    sd_patterns = [
        "*.ckpt",
        "*.safetensors",
        "*.pt",
        "*.bin",
        "config.json",
        "model_index.json",
        "vocab.json",
        "merges.txt",
        "tokenizer.json",
        "*.yaml",
        "*.yml",
    ]

    if is_sd:
        # Try to read model_index.json for metadata, else just return minimal config
        model_index_path = os.path.join(hugging_face_id, "model_index.json")
        fs = huggingface_hub.HfFileSystem()
        model_index = None
        try:
            with fs.open(model_index_path) as f:
                model_index = json.load(f)
        except Exception:
            model_index = None
        config = {
            "uniqueID": hugging_face_id,
            "name": getattr(hf_model_info, "modelId", hugging_face_id),
            "private": getattr(hf_model_info, "private", False),
            "gated": getattr(hf_model_info, "gated", False),
            "architecture": getattr(model_index, "_class_name", "StableDiffusionPipeline"),
            "huggingface_repo": hugging_face_id,
            "model_type": "stable-diffusion",
            "size_of_model_in_mb": get_huggingface_download_size(hugging_face_id, sd_patterns) / (1024 * 1024),
            "tags": model_tags,
            "license": model_card_data.get("license", ""),
            "allow_patterns": sd_patterns,
        }
        if model_index:
            config["model_index"] = model_index
        return config

    # Non-SD models: require config.json
    huggingface_hub.hf_hub_download(repo_id=hugging_face_id, filename="config.json")
    fs = huggingface_hub.HfFileSystem()
    filename = os.path.join(hugging_face_id, "config.json")
    with fs.open(filename) as f:
        filedata = json.load(f)

        # config.json stores a list of architectures but we only store one so just take the first!
        architecture_list = filedata.get("architectures", [])
        architecture = architecture_list[0] if architecture_list else ""

        # Oh except we list GGUF and MLX as architectures, but HuggingFace sometimes doesn't
        # It is usually stored in library, or sometimes in tags
        library_name = getattr(hf_model_info, "library_name", "")
        if library_name:
            if library_name.lower() == "mlx":
                architecture = "MLX"
            if library_name.lower() == "gguf":
                architecture = "GGUF"

        # And sometimes it is stored in the tags for the repo
        model_tags = getattr(hf_model_info, "tags", [])
        if "mlx" in model_tags:
            architecture = "MLX"

        # calculate model size
        model_size = get_huggingface_download_size(hugging_face_id) / (1024 * 1024)

        # TODO: Context length definition seems to vary by architecture. May need conditional logic here.
        context_size = filedata.get("max_position_embeddings", "")

        # --- Stable Diffusion detection and allow_patterns logic ---
        sd_patterns = [
            "*.ckpt",
            "*.safetensors",
            "*.pt",
            "*.bin",
            "config.json",
            "model_index.json",
            "vocab.json",
            "merges.txt",
            "tokenizer.json",
            "*.yaml",
            "*.yml",
        ]
        is_sd = False
        # Heuristic: check tags or config for 'stable-diffusion' or 'diffusers' or common SD files
        if any("stable-diffusion" in t or "diffusers" in t for t in model_tags):
            is_sd = True
        # Or check for model_index.json in repo files
        try:
            repo_files = huggingface_hub.list_repo_files(hugging_face_id)
            if any(f.endswith("model_index.json") for f in repo_files):
                is_sd = True
        except Exception:
            pass

        # TODO: Figure out description, paramters, model size
        newmodel = basemodel.BaseModel(hugging_face_id)
        config = await newmodel.get_json_data()
        config = {
            "uniqueID": hugging_face_id,
            "name": filedata.get("name", hugging_face_id),
            "context": context_size,
            "private": getattr(hf_model_info, "private", False),
            "gated": getattr(hf_model_info, "gated", False),
            "architecture": architecture,
            "huggingface_repo": hugging_face_id,
            "model_type": filedata.get("model_type", ""),
            "size_of_model_in_mb": model_size,
            "library_name": library_name,
            "tags": model_tags,
            "transformers_version": filedata.get("transformers_version", ""),
            "quantization": filedata.get("quantization", ""),
            "license": model_card_data.get("license", ""),
        }
        return config

    # Something did not go to plan
    return None


def get_huggingface_download_size(model_id: str, allow_patterns: list = []):
    """
    Get the size in bytes of all files to be downloaded from Hugging Face.

    Raises: RepositoryNotFoundError if model_id doesn't exist on huggingface (or can't be accessed)
    """

    # This can throw Exceptions: RepositoryNotFoundError
    hf_model_info = huggingface_hub.list_repo_tree(model_id)

    # Iterate over files in the model repo and add up size if they are included in download
    download_size = 0
    total_size = 0
    for file in hf_model_info:
        if isinstance(file, RepoFile):
            total_size += file.size

            # if there are no allow_patterns to filter on then add every file
            if len(allow_patterns) == 0:
                download_size += file.size

            # If there is an array of allow_patterns then only add this file
            # if it matches one of the allow_patterns
            else:
                for pattern in allow_patterns:
                    if fnmatch.fnmatch(file.path, pattern):
                        download_size += file.size
                        break

    return download_size


def delete_model_from_hf_cache(model_id: str, cache_dir: str = None) -> None:
    """
    Delete a model from the Hugging Face cache by scanning the cache to locate
    the model repository and then deleting its folder.

    If cache_dir is provided, it will be used as the cache location; otherwise,
    the default cache directory is used (which respects HF_HOME or HF_HUB_CACHE).

    Args:
        model_id (str): The model ID (e.g. "mlx-community/Qwen2.5-7B-Instruct-4bit").
        cache_dir (str, optional): Custom cache directory.
    """

    # Scan the cache using the provided cache_dir if available.
    hf_cache_info = scan_cache_dir(cache_dir=cache_dir) if cache_dir else scan_cache_dir()

    found = False
    # Iterate over all cached repositories.
    for repo in hf_cache_info.repos:
        # Only consider repos of type "model" and match the repo id.
        if repo.repo_type == "model" and repo.repo_id == model_id:
            shutil.rmtree(repo.repo_path)
            print(f"Deleted model cache folder: {repo.repo_path}")
            found = True
            break

    if not found:
        print(f"Model cache folder not found for: {model_id}")
