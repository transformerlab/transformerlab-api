"""
LocalModelStore manages models in both the database and
.transformerlab/workspace/models directory.

There are functions in model_helper to make it easier to work with.
"""

import os
import json

from transformerlab.models import modelstore
import transformerlab.db as db
from transformerlab.shared import dirs


class LocalModelStore(modelstore.ModelStore):
    def __init__(self):
        super().__init__()

    async def list_models(self):
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

        return models

    async def compute_output_model(self, input_model, adaptor_name):
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

    async def build_provenance(self, jobs):
        """
        Build a mapping from the computed output model name (as produced by a training job)
        to the job details. Each job is assumed to have a 'job_data' field (already converted
        from JSON) containing:
          - 'model_name': the input model used for training.
          - 'dataset': the dataset used.
          - 'config': a JSON string containing additional training parameters (including 'adaptor_name').
          - 'start_time' and 'end_time'.

        The computed output model is based on the input model and the adaptor name.
        """
        provenance = {}
        for job in jobs:
            # Each job is a dict containing a key "job_data"
            job_data = job["job_data"]
            if job_data.get("model_name", None) is not None:
                input_model = job_data.get("model_name").split("/")[-1]
                dataset = job_data.get("dataset")
                start_time = job_data.get("start_time")
                end_time = job_data.get("end_time")

                # The configuration is stored as a JSON string inside job_data["config"]
                config_str = job_data.get("config")
                try:
                    config = json.loads(config_str)
                except Exception as e:
                    print(f"Error parsing config for job id {job.get('id', 'unknown')}: {e}")
                    config = {}

                adaptor_name = config.get("adaptor_name")
                if not adaptor_name:
                    # If no adaptor is specified, we cannot compute an output model.
                    continue

                output_model = await self.compute_output_model(input_model, adaptor_name)

                provenance[output_model] = {
                    "job_id": job.get("id"),
                    "input_model": input_model,
                    "output_model": output_model,
                    "dataset": dataset,
                    "parameters": config,
                    "start_time": start_time,
                    "end_time": end_time,
                }
        return provenance

    async def trace_provenance(self, latest_model, provenance_mapping):
        """
        Trace back the chain of training jobs from the final model.

        Starting with the provided latest model name, the function walks backwards through the
        provenance mapping by using the input model of each job (normalized to its base name).
        It stops when the input model is the same as the current model (indicating a base model).
        The result is a chain (list) of job details in the order from the earliest training step
        to the final training job.
        """
        chain = []
        current_model = latest_model.split("/")[-1]
        print(f"Tracing provenance chain for model {current_model}")

        while current_model in provenance_mapping:
            job_details = provenance_mapping[current_model]
            chain.insert(0, job_details)
            # Normalize the parent model by taking the last part if it is a path.
            parent_model = job_details["input_model"].split("/")[-1]
            if parent_model == current_model:
                break
            current_model = parent_model
        return chain

    async def get_evals_by_model(self):
        """
        Retrieve all completed EVAL jobs and group them by the model_name specified in the job_data.
        For each eval, remove keys we want to ignore (i.e., additional_output_path,
        completion_status, and completion_details) and attach the job_id.
        """
        eval_jobs = await db.jobs_get_all(type="EVAL", status="COMPLETED")
        evals_by_model = {}
        for job in eval_jobs:
            eval_data = job["job_data"]
            # Remove keys that are not required
            eval_data.pop("additional_output_path", None)
            eval_data.pop("completion_status", None)
            eval_data.pop("completion_details", None)
            # Attach the JOB ID
            eval_data["job_id"] = job.get("id")
            model_name = eval_data.get("model_name")
            if model_name:
                evals_by_model.setdefault(model_name, []).append(eval_data)
        return evals_by_model

    async def list_model_provenance(self, model_id):
        """
        List all model journeys in the workspace.
        """
        print(f"Listing model provenance for model {model_id}")

        # Fetch all completed TRAIN jobs using the provided function
        jobs = await db.jobs_get_all(type="TRAIN", status="COMPLETE")

        print(f"Found {len(jobs)} completed training jobs")

        # Build a mapping from computed output model name to job details
        provenance_mapping = await self.build_provenance(jobs)

        print(f"Built provenance mapping with {len(provenance_mapping)} entries")

        # Trace the provenance chain leading to the given final model name
        chain = await self.trace_provenance(model_id, provenance_mapping)

        print(f"Traced provenance chain with {len(chain)} entries")
        # Retrieve eval jobs grouped by model_name using the dedicated function
        evals_by_model = await self.get_evals_by_model()

        print(f"Retrieved evals by model with {len(evals_by_model)} entries")

        # For every training job in the provenance chain, attach evals for the corresponding model
        for item in chain:
            train_model_name = item.get("train_model_name")
            item["evals"] = evals_by_model.get(train_model_name, [])

        # Create the final provenance dictionary output
        output = {"final_model": model_id, "provenance_chain": chain}

        return output
