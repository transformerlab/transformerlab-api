from transformerlab.db.db import experiment_get
from lab import Experiment


async def get_experiment_by_id(experimentId: int):
    """
    Helper function to get SDK Experiment from an ID.
    """
    # first get the experiment name:
    data = await experiment_get(experimentId)

    # if the experiment does not exist, return none
    if data is None:
        return None

    experiment_name = data["name"]
    experiment = Experiment.get(experiment_name)
    return experiment
