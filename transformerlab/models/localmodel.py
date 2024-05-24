import os

from transformerlab.models import basemodel

import huggingface_hub

async def list_models(path: str, uninstalled_only: bool = True):
    """
    This function recursively calls itself to generate a list of models under folder.
    First try to determine if this directory is itself a model (and then check
    to see if we can support it).
    If this directory is not a model then search subdirectories for models.
    NOTE: If you pass this a directory with a large tree under it, this can take 
    a long time to run!
    """

    # TODO: Implement! :D
    return []

class LocalFilesystemModel(basemodel.BaseModel):
    def __init__(self, model_path):

        # The ID for this model will be the file or directory name without path
        model_id = os.path.basename(model_path)

        super().__init__(model_id)

        self.id = model_id
        self.name = model_id

        self.model_source = "local"
        self.source_id_or_path = model_path

        model_details = {}
        architecture = "unknown"
        formats = []

        self.architecture = architecture
        self.formats = formats

        # TODO: This is a HACK! Need to not have two sources for these fields
        self.json_data["architecture"] = self.architecture
        self.json_data["formats"] = self.formats
        self.json_data["source"] = self.model_source
        self.json_data["source_id_or_path"] = self.source_id_or_path
        self.json_data["model_filename"] = self.model_filename