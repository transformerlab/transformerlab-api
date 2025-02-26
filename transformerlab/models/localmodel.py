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

    async def list_model_journeys(self):
        """
        List all model journeys in the workspace.
        """
        model_list = await self.list_models()
        model_journey = {
            "models": [],
        }
        training_jobs = await db.jobs_get_all(type="TRAIN", status="COMPLETED")
        eval_jobs = await db.jobs_get_all(type="EVAL", status="COMPLETED")
        for model in model_list:
            model_id = model["model_id"]
            for job in training_jobs:
                if job["model_id"] == model_id:
                    pass
        return model_journey


# sample output of list_model_journey:
"""
{
  models: [
    {
      type: 'base_model',
      id: 'meta/llama3.1-8B-instruct',
      name: 'meta/llama3.1-8B-instruct',
      children: [
        {
          type: 'fine_tuning_job',
          jobId: 1,
          metadata: { dataset: 'Dataset A' },
          child: {
            type: 'fine_tuned_model',
            modelId: 'ft_model_1',
            name: 'Fine Tuned Model 1',
            children: [
              {
                type: 'eval_job',
                jobId: 2,
                metadata: { metric: 'accuracy', value: 95.5 }
              },
              {
                type: 'eval_job',
                jobId: 3,
                metadata: { metric: 'accuracy', value: 96.7 }
              },
              {
                type: 'fine_tuning_job',
                jobId: 6,
                metadata: { dataset: 'Dataset B' },
                child: {
                  type: 'fine_tuned_model',
                  modelId: 'ft_model_3',
                  name: 'Fine Tuned Model 3',
                  children: []
                }
              }
            ]
          }
        },
        {
          type: 'fine_tuning_job',
          jobId: 4,
          metadata: { dataset: 'Dataset C' },
          child: {
            type: 'fine_tuned_model',
            modelId: 'ft_model_2',
            name: 'Fine Tuned Model 2',
            children: [
              {
                type: 'eval_job',
                jobId: 5,
                metadata: { metric: 'accuracy', value: 97.0 }
              }
            ]
          }
        }
      ]
    }
  ]
  }
"""
