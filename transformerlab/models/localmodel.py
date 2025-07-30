"""
LocalModelStore manages models in both the database and
.transformerlab/workspace/models directory.

There are functions in model_helper to make it easier to work with.
"""

import os
import json
from huggingface_hub import hf_hub_download
from transformerlab.db.jobs import jobs_get_all
from transformerlab.models import modelstore
import transformerlab.db.db as db
from transformerlab.shared import dirs


def is_sentence_transformer_model(
    model_name_or_path: str,
    token: bool | str | None = None,
    cache_folder: str | None = None,
    revision: str | None = None,
    local_files_only: bool = False,
) -> bool:
    """
    Checks if the given model name or path corresponds to a SentenceTransformer model.

    Args:
        model_name_or_path (str): The name or path of the model.
        token (Optional[Union[bool, str]]): The token to be used for authentication. Defaults to None.
        cache_folder (Optional[str]): The folder to cache the model files. Defaults to None.
        revision (Optional[str]): The revision of the model. Defaults to None.
        local_files_only (bool): Whether to only use local files for the model. Defaults to False.

    Returns:
        bool: True if the model is a SentenceTransformer model, False otherwise.
    """
    return bool(
        load_file_path(
            model_name_or_path,
            "modules.json",
            token=token,
            cache_folder=cache_folder,
            revision=revision,
            local_files_only=local_files_only,
        )
    )


def load_file_path(
    model_name_or_path: str,
    filename: str,
    token: bool | str | None = None,
    cache_folder: str | None = None,
    revision: str | None = None,
    local_files_only: bool = True,
) -> str | None:
    """
    Loads a file from a local or remote location.

    Args:
        model_name_or_path (str): The model name or path.
        filename (str): The name of the file to load.
        token (Optional[Union[bool, str]]): The token to access the remote file (if applicable).
        cache_folder (Optional[str]): The folder to cache the downloaded file (if applicable).
        revision (Optional[str], optional): The revision of the file (if applicable). Defaults to None.
        local_files_only (bool, optional): Whether to only consider local files. Defaults to False.

    Returns:
        Optional[str]: The path to the loaded file, or None if the file could not be found or loaded.
    """
    # If file is local
    file_path = os.path.join(model_name_or_path, filename)
    if os.path.exists(file_path):
        return file_path

    # If file is remote
    try:
        return hf_hub_download(
            model_name_or_path,
            filename=filename,
            revision=revision,
            library_name="sentence-transformers",
            token=token,
            cache_dir=cache_folder,
            local_files_only=local_files_only,
        )
    except Exception:
        return None


class LocalModelStore(modelstore.ModelStore):
    def __init__(self):
        super().__init__()

    async def filter_embedding_models(self, models, embedding=False):
        """
        Filter out models based on whether they are embedding models or not.
        """

        embedding_models = []
        non_embedding_models = []

        for model in models:
            if model.get("model_id", None):
                if (
                    model["json_data"].get("model_filename", None)
                    and model["json_data"]["model_filename"].strip() != ""
                ):
                    model_id = model["json_data"]["model_filename"]
                elif model.get("local_path", None) and model["local_path"].strip() != "":
                    model_id = model["local_path"]
                else:
                    model_id = model["model_id"]
            else:
                print("Model ID not found in model data.")
                print(model)
            if is_sentence_transformer_model(model_id):
                embedding_models.append(model)
            else:
                non_embedding_models.append(model)

        return embedding_models if embedding else non_embedding_models

    async def list_models(self, embedding=False):
        """
        Check both the database and workspace for models.
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
                            filedata["local_path"] = os.path.join(models_dir, entry)

                            # Some models are a single file (possibly of many in a directory, e.g. GGUF)
                            # For models that have model_filename set we should link directly to that specific file
                            if "model_filename" in filedata and filedata["model_filename"]:
                                filedata["local_path"] = os.path.join(
                                    filedata["local_path"], filedata["model_filename"]
                                )

                            models.append(filedata)

                    except FileNotFoundError:
                        # do nothing: just ignore this directory
                        pass

        # Filter out models based on whether they are embedding models or not
        models = await self.filter_embedding_models(models, embedding)

        return models

    def compute_output_model(self, input_model, adaptor_name):
        """
        Compute the output model name by taking the last part of the input model
        (in case it is a full path) and appending an underscore and the adaptor name.

        For example:
            input_model: "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            adaptor_name: "ml-qa"
            returns: "TinyLlama-1.1B-Chat-v1.0_ml-qa"
        """
        base_model = input_model.split("/")[-1]
        return f"{base_model}_{adaptor_name}"

    async def build_provenance(self):
        """
        Build a mapping from model names to their provenance data by reading _tlab_provenance.json files
        in each model directory.
        """
        provenance = {}
        models_dir = dirs.MODELS_DIR

        # Load the tlab_complete_provenance.json file if it exists
        complete_provenance_file = os.path.join(models_dir, "_tlab_complete_provenance.json")
        if os.path.exists(complete_provenance_file):
            with open(complete_provenance_file, "r") as f:
                try:
                    provenance = json.load(f)
                except json.JSONDecodeError:
                    print(f"Error loading {complete_provenance_file}: Invalid JSON format.")
                    provenance = {}
                except Exception as e:
                    print(f"Error loading {complete_provenance_file}: {str(e)}")
                    provenance = {}
            # Check if the provenance for any local models was missed
            provenance, local_added_count = await self.check_provenance_for_local_models(provenance)
            if local_added_count != 0:
                # Save new provenance mapping
                with open(complete_provenance_file, "w") as f:
                    json.dump(provenance, f)
            # Check if the provenance mapping is up to date
            # The -1 here indicates that we are not counting the _tlab_complete_provenance.json file in models_dir
            if len(provenance) > 0 and len(os.listdir(models_dir)) + local_added_count - 1 == len(provenance):
                return provenance, False

        # If the provenance mapping is not built or models_dir has changed, we need to rebuild it
        # Scan all model directories
        with os.scandir(models_dir) as dirlist:
            for entry in dirlist:
                if entry.is_dir():
                    # Look for provenance file
                    provenance_file = os.path.join(models_dir, entry.name, "_tlab_provenance.json")
                    try:
                        if os.path.exists(provenance_file):
                            with open(provenance_file, "r") as f:
                                prov_data = json.load(f)
                                if "md5_checksums" in prov_data:
                                    prov_data["parameters"]["md5_checksums"] = prov_data["md5_checksums"]

                                # Compute output model name if needed
                                output_model = prov_data.get("model_name")
                                if not output_model and prov_data.get("input_model") and prov_data.get("adaptor_name"):
                                    output_model = self.compute_output_model(
                                        prov_data["input_model"], prov_data["adaptor_name"]
                                    )
                                prov_data["output_model"] = output_model

                                # Add to provenance mapping
                                provenance[output_model] = prov_data

                    except Exception as e:
                        print(f"Error loading provenance for {entry.name}: {str(e)}")

        # Import from local_models too when building from scratch
        provenance, _ = await self.check_provenance_for_local_models(provenance)

        return provenance, True

    async def check_provenance_for_local_models(self, provenance):
        # Get the list of all local models
        models = await db.model_local_list()
        models_added_to_provenance = 0
        # Iterate through models and check if they have provenance data and if they exist already in provenance
        for model_dict in models:
            if (
                model_dict.get("model_id", "") not in provenance.keys()
                or model_dict.get("model_name", "") not in provenance.keys()
            ):
                # Check if the model_source is local
                if model_dict.get("json_data", {}).get("source", "") == "local" and os.path.exists(
                    model_dict.get("json_data", {}).get("source_id_or_path", "")
                ):
                    # Check if the model has a _tlab_provenance.json file
                    provenance_file = os.path.join(
                        model_dict["json_data"]["source_id_or_path"], "_tlab_provenance.json"
                    )
                    if os.path.exists(provenance_file):
                        # Load the provenance file
                        with open(provenance_file, "r") as f:
                            prov_data = json.load(f)
                            if "md5_checksums" in prov_data:
                                prov_data["parameters"]["md5_checksums"] = prov_data["md5_checksums"]

                            # Compute output model name if needed
                            output_model = prov_data.get("model_name")
                            if not output_model and prov_data.get("input_model") and prov_data.get("adaptor_name"):
                                output_model = self.compute_output_model(
                                    prov_data["input_model"], prov_data["adaptor_name"]
                                )
                            prov_data["output_model"] = output_model

                            # Add to provenance mapping
                            provenance[output_model] = prov_data
                            models_added_to_provenance += 1

        return provenance, models_added_to_provenance

    async def trace_provenance(self, latest_model, provenance_mapping):
        """
        Trace back the chain of training jobs from the final model using provenance files.
        """
        chain = []
        current_model = latest_model.split("/")[-1]

        while current_model in provenance_mapping:
            job_details = provenance_mapping[current_model]
            chain.insert(0, job_details)

            # Get parent model from provenance
            parent_model = job_details.get("input_model")
            if not parent_model or parent_model == current_model:
                break

            # Normalize the parent model by taking the last part if it is a path
            parent_model = parent_model.split("/")[-1]
            current_model = parent_model

        return chain

    async def get_evals_by_model(self):
        """
        Retrieve all completed EVAL jobs and group them by the model_name specified in the job_data.
        For each eval, remove keys we want to ignore (i.e., additional_output_path,
        completion_status, and completion_details) and attach the job_id.
        """
        eval_jobs = await jobs_get_all(experiment_id=None, type="EVAL", status="COMPLETE")
        evals_by_model = {}
        for job in eval_jobs:
            eval_data = job["job_data"]
            # # Remove keys that are not required
            # eval_data.pop("additional_output_path", None)
            # eval_data.pop("completion_status", None)
            # eval_data.pop("completion_details", None)
            # Attach the JOB ID
            eval_data["job_id"] = job.get("id")
            model_name = eval_data.get("model_name")
            adapter_name = eval_data.get("model_adapter")
            if model_name and os.path.exists(model_name):
                # If model_name is a path, take the last part
                model_name = model_name.split("/")[-1]
            if model_name and adapter_name:
                evals_by_model.setdefault(self.compute_output_model(model_name, adapter_name), []).append(eval_data)
            if model_name:
                evals_by_model.setdefault(model_name, []).append(eval_data)
        return evals_by_model

    async def list_model_provenance(self, model_id):
        """
        List model provenance by reading from _tlab_provenance.json files.
        """
        # Build a mapping from model names to their provenance data
        provenance_mapping, provenance_updated = await self.build_provenance()

        if provenance_updated:
            # Save the provenance mapping as a json file
            provenance_file = os.path.join(dirs.MODELS_DIR, "_tlab_complete_provenance.json")
            with open(provenance_file, "w") as f:
                json.dump(provenance_mapping, f)

        # Trace the provenance chain leading to the given model
        chain = await self.trace_provenance(model_id, provenance_mapping)

        # Retrieve eval jobs grouped by model_name
        evals_by_model = await self.get_evals_by_model()

        if len(chain) == 0:
            item = {
                "input_model": model_id,
                "output_model": model_id,
                "dataset": None,
                "parameters": {},
                "start_time": None,
                "end_time": None,
            }
            item["evals"] = evals_by_model.get(model_id, [])
            return {"final_model": model_id, "provenance_chain": [item]}

        # For every training job in the provenance chain, attach evals
        for item in chain:
            output_model = item.get("output_model") or item.get("model_name")
            item["evals"] = evals_by_model.get(output_model, [])

        # Create the final provenance dictionary
        output = {"final_model": model_id, "provenance_chain": chain}

        return output
