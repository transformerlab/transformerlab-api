class BaseModel:
    """
    A basic representation of a Model in TransformerLab.

    Properties:
    model_id:       a string that is unique to Transformer Lab
    name:           Printable name for the model (how it appears in the app)
    model_store:    Where the model is stored (huggingface, local_path, etc.)

    installed:      True if this model can be used in TLab's Local Store
    model_path:     Path to the model in its model_store.
                    This is usually either a file system path, 
                    or a unique repo ID to get a model out of an app's cache.
    model_filename: With model_path, a specific filename for this model (vs. a folder)
                    For example, this is used for GGUF files.
    architecture:   A string describing the model architecture used to determine
                    support for the model and how to run

    json_data:      an unstructured data blob that can contain any data relevant 
                    to the model or its model_store.
    """

    
    def __init__(self, id_in_model_store):
        """
        The constructor takes an ID (id_in_model_store) that is unique to the model source.
        This may be different than the unique ID used in Transformer Lab (self.id).

        That is, model sources (hard drive folders, application caches, web sites, etc.)
        may have many models with the same id_in_model_store, but their self.id must be 
        unique to TransformerLab in order to import in to the Transfoerm Lab store.
        """

        self.id = id_in_model_store
        self.name = id_in_model_store
        self.model_store = None

        self.installed = False
        self.model_path = None
        self.model_filename = None
        self.architecture = "unknown"

        # While json_data is unstructured and flexible
        # These are the fields that the app generally expects to exist
        self.json_data = {
            "uniqueID": self.id,
            "name": self.id,
            "description": "",
            "huggingface_repo": "",
            "parameters": "",
            "context": "",
            "architecture": self.architecture,
            "license": "",
            "logo": "",

            # The following are from huggingface_hu.hf_api.ModelInfo
            "private": False, 
            "gated": False, # Literal["auto", "manual", False]
            "model_type": "",
            "library_name": "", 
            "transformers_version": ""
        }


    def get_path_to_model(self):
        if self.model_filename:
            return os.path.join(model_path, self.model_filename)
        else:
            return self.model_path