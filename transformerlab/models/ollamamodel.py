"""
This package defines OllamaModel which represents a model that is stored
in the Ollama cache and can be imported into Transformer Lab.
"""

from transformerlab.models import basemodel

import os
import json
import errno

from transformerlab.shared import dirs


async def list_models():
    try:
        ollama_model_library = ollama_models_library_dir()
    except Exception:
        return []

    models = []
    with os.scandir(ollama_model_library) as dirlist:
        # Scan the ollama cache repos for cached models
        for entry in dirlist:
            if entry.is_dir():
                ollama_model = OllamaModel(entry.name)
                models.append(ollama_model)

    return models


class OllamaModel(basemodel.BaseModel):
    """
    Wrapper for models imported from Ollama.
    These models are kept in the ollama cache (usually ~/.ollama)
    """

    def __init__(self, ollama_id):

        # A convention in Transformer Lab is that GGUF models are named
        # modelname.gguf. Most models in ollama will not have the gguf part.
        file_name, file_extension = os.path.splitext(ollama_id)
        if file_extension == ".gguf":
            import_id = ollama_id
        else:
            import_id = f"{ollama_id}.gguf"

        super().__init__(import_id)

        # The actual modelfile is in the ollama cache
        self.source = "ollama"
        self.source_id_or_path = ollama_id
        self.model_filename = self._get_model_blob_filename()
        print(self.model_filename)

    def _get_model_blob_filename(self):
        """
        This returns just the filename of the blob containing the actual model
        If anything goes wrong along the way this returns None
        """

        # Get the path to the manifest file
        try:
            library_dir = ollama_models_library_dir()
        except Exception:
            self.status = "failed to find ollama library"
            return None

        # Read in the manifest file
        manifestfile = os.path.join(
            library_dir, self.source_id_or_path, "latest")
        try:
            with open(manifestfile, "r") as f:
                filedata = json.load(f)

        except FileNotFoundError:
            print("ollama manifest file not found:", manifestfile)
            return None

        # The format of v2 schema is that there is a list called "layers"
        # Objects in layers have data on the files in the blobs directory
        # those files can be model, license, template, params
        # we are after the model file
        schemaVersion = filedata.get("schemaVersion", None)
        if schemaVersion == 2:
            layers = filedata.get("layers", [])
            for layer in layers:
                # Each layer has a mediaType field describing what the file contains
                # and a digest field with the name of the file
                if layer.get("mediaType", None) == "application/vnd.ollama.image.model":
                    # Check if the specified file exists or not!
                    digestvalue = layer.get("digest", None)
                    models_dir = ollama_models_dir()
                    blobs_dir = os.path.join(models_dir, "blobs")

                    # ollama lists the file with a ":" that needs to be converted to a "-"
                    modelfile = digestvalue.replace(":", "-")
                    model_path = os.path.join(blobs_dir, modelfile)
                    try:
                        with open(model_path, "r") as f:
                            return model_path
                    except FileNotFoundError:
                        self.status = f"model file does not exist {modelfile}"
                        return None

            # If we get here it means schemaVersion is 2 but there was no listed model
            self.status = "no valid ollama.image.model attribute"

        # schemaVersion is not 2. We only support 2.
        self.status = f"unsupported ollama schemaVersion {schemaVersion}"
        return None

    async def get_json_data(self):
        # inherit json_data from the parent and only update specific fields
        json_data = await super().get_json_data()

        # Assume all models from ollama are GGUF
        json_data["architecture"] = "GGUF"
        json_data["formats"] = ["GGUF"]
        json_data["source_id_or_path"] = self.model_filename

        return json_data

    async def install(self):
        input_model_path = self.model_filename
        output_model_id = self.id
        output_filename = self.id

        # Make sure our source file exists
        if not input_model_path:
            raise ValueError(f"No modelfile set for ollama model {self.id}")
        elif not os.path.exists(output_path):
            raise FileNotFoundError(errno.ENOENT, os.strerror(
                errno.ENOENT), input_model_path)

        # Create a directory for the model. Make sure it doesn't exist already.
        output_path = os.path.join(dirs.MODELS_DIR, output_model_id)
        if os.path.exists(output_path):
            raise FileExistsError(
                errno.EEXIST, "Directory already exists", output_path)
        os.makedirs(output_path)

        # Create a link in the directory that points to the source blob
        link_name = os.path.join(output_path, output_filename)
        os.symlink(input_model_path, link_name)

        # Create an info.json file so this can be read by the system
        # TODO: Add parameters to json_data
        model_description = [
            {
                "model_id": output_model_id,
                "model_filename": output_filename,
                "name": f"{self.name} (Ollama)",
                "local_model": True,
                "json_data": {
                    "uniqueID": output_model_id,
                    "name": f"{self.name} (Ollama)",
                    "model_filename": output_filename,
                    "description": f"Link to Ollama model {self.source_id_or_path}",
                    "source": "transformerlab",
                    "architecture": "GGUF",
                    "huggingface_repo": ""
                },
            }
        ]
        model_info_file = os.path.join(output_path, "info.json")
        with open(model_info_file, "w") as f:
            json.dump(model_description, f)


#########################
#  DIRECTORY STRUCTURE  #
#########################


def ollama_models_dir():
    try:
        ollama_dir = os.environ["OLLAMA_MODELS"]
    except KeyError:
        ollama_dir = os.path.join(os.path.expanduser("~"), ".ollama", "models")

    # Check that the directory actually exists
    if not os.path.isdir(ollama_dir):
        return None

    return ollama_dir


def ollama_models_library_dir():
    models_dir = ollama_models_dir()

    if not models_dir:
        raise FileNotFoundError(errno.ENOENT, os.strerror(
            errno.ENOENT), "Ollama models directory")

    library_dir = os.path.join(
        models_dir, "manifests", "registry.ollama.ai", "library")

    if not os.path.isdir(library_dir):
        raise NotADirectoryError(errno.ENOENT, os.strerror(
            errno.ENOENT), library_dir)

    return library_dir
