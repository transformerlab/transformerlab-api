"""
This is the default Transformer Lab model store.
It reads from both the database and the workspace models directory.
"""

import os
import json
import transformerlab.db as db
from transformerlab.models import modelstore
from transformerlab.shared import dirs


class LocalModelStore(modelstore.ModelStore):

    def __init__(self):
        super().__init__()

    async def fetch_model_list(self):
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
