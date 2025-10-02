from lab import Dataset


def create_local_dataset(dataset_id, json_data=None):
    # Create a new dataset
    new_dataset = Dataset.create(dataset_id)

    # Take description from json_data if it exists
    description = json_data.get("description", "") if isinstance(json_data, dict) else ""
    new_dataset.set_metadata(
        location="local", description=description, size=-1, json_data=json_data if json_data is not None else {}
    )
    return new_dataset
