"""
Root class that other model stores inherit from.
This is not useful on its own, it just defines the base object.
Sort of like an abstract class or interface.
"""


class ModelStore:
    model_list = None

    def __str__(self):
        # For debug output
        return str(self.__class__) + ": " + str(self.__dict__)

    async def fetch_model_list(self):
        """
        This is the key function to override.
        Generate a list of BaseModel objects that will be cached.
        """
        return []

    async def list_models(self):
        """
        Dont' override this.  Override fetch_model_list instead.
        """
        if model_list is None:
            model_list = await self.fetch.model_list()
        return model_list

    async def has_model(self, model_id):
        """
        Probably don't override this either.
        """
        model_list = await self.list_models()
        for model in model_list:
            if model["model_id"] == model_id:
                return True
        return False
