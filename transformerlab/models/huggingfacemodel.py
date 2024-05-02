import os

from transformerlab.models import basemodel

import huggingface_hub


def list_models(uninstalled_only: bool = True):
    return []


class HuggingFaceModel(basemodel.BaseModel):
        
    def __init__(self, hugging_face_id):
        super().__init__(hugging_face_id)

        self.id = hugging_face_id
        self.name = hugging_face_id

        # TODO: How to figure out architecture?
        self.architecture = "GGUF"


        self.model_source = "huggingface"
        self.source_id_or_path = hugging_face_id

        # TODO: filename needed for GGUF
        self.model_filename = None

       # inherit json_data from the parent and only update specific fields
        self.json_data["uniqueID"] = self.id
        self.json_data["name"] = self.name
        self.json_data["architecture"] = self.architecture
