from transformerlab.models import basemodel

import os
import json
import errno
import shutil

_LM_MODEL_EXTS = (".gguf", ".safetensors", ".pt", ".bin")


async def list_models():
    try:
        models_dir = lmstudio_models_dir()
    except Exception as e:
        print("Failed to locate LM Studio models directory:")
        print(str(e))
        return []

    if not models_dir:
        return []

    models = []
    for root, _, files in os.walk(models_dir):
        for fname in files:
            if fname.lower().endswith(_LM_MODEL_EXTS):
                model_path = os.path.join(root, fname)
                models.append(LMStudioModel(model_path))

    return models


class LMStudioModel(basemodel.BaseModel):
    def __init__(self, model_path: str):
        filename = os.path.basename(model_path)
        super().__init__(model_id=filename)

        self.source = "lmstudio"
        self.name = f"{os.path.splitext(filename)[0]} (LM Studio)"
        self.source_id_or_path = os.path.abspath(model_path)
        self.model_filename = filename

    async def get_json_data(self):
        json_data = await super().get_json_data()

        ext = os.path.splitext(self.model_filename)[1].lower()
        if ext == ".gguf":
            json_data["architecture"] = "GGUF"
            json_data["formats"] = ["GGUF"]
        elif ext in (".safetensors", ".pt", ".bin"):
            json_data["architecture"] = "PyTorch"
            json_data["formats"] = ["safetensors" if ext == ".safetensors" else "pt"]
        else:
            json_data["architecture"] = ""
            json_data["formats"] = []

        json_data["source_id_or_path"] = self.source_id_or_path
        return json_data

    async def install(self):
        input_model_path = self.source_id_or_path
        if not input_model_path or not os.path.isfile(input_model_path):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), input_model_path)

        from lab.dirs import get_models_dir

        output_filename = self.id
        output_path = os.path.join(get_models_dir(), output_filename)

        if os.path.exists(output_path):
            raise FileExistsError(errno.EEXIST, "Model already exists", output_path)
        os.makedirs(output_path, exist_ok=True)

        link_name = os.path.join(output_path, output_filename)
        os.symlink(input_model_path, link_name)

        json_data = await self.get_json_data()

        model_description = {
            "model_id": self.id,
            "model_filename": output_filename,
            "name": self.name,
            "source": self.source,
            "json_data": {
                "uniqueID": self.id,
                "name": self.name,
                "model_filename": output_filename,
                "description": f"LM Studio model {self.source_id_or_path}",
                "source": self.source,
                "architecture": json_data["architecture"],
            },
        }

        model_info_file = os.path.join(output_path, "index.json")
        with open(model_info_file, "w") as f:
            json.dump(model_description, f)


def lmstudio_models_dir():
    try:
        lm_dir = os.environ["LMSTUDIO_MODELS"]
    except KeyError:
        lm_dir = os.path.join(os.path.expanduser("~"), ".lmstudio", "models")

    if os.path.isdir(lm_dir):
        return lm_dir

    if shutil.which("lmstudio"):
        return lm_dir

    return None
