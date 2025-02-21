"""
ModelStore - Root class that other model stores inherit from.
This is not useful on its own, it just defines the base object.
Sort of like an abstract class or interface.
"""


class ModelStore:

    def __str__(self):
        # For debug output
        return str(self.__class__) + ": " + str(self.__dict__)

    async def list_models(self):
        return []

    async def has_model(self, model_id):
        """
        Probably don't need to override this.
        """
        model_list = await self.list_models()
        for model in model_list:
            if model["model_id"] == model_id:
                return True
        return False
