import os
import json

from lab import Experiment
from lab import dirs as lab_dirs
from lab import storage


def experiment_get_all():
    experiments = []
    experiments_dir = lab_dirs.get_experiments_dir()
    if storage.exists(experiments_dir):
        try:
            exp_dirs = storage.ls(experiments_dir, detail=False)
            # Sort the directories
            exp_dirs = sorted(exp_dirs)
            for exp_path in exp_dirs:
                if storage.isdir(exp_path):
                    # Extract the directory name from the path
                    exp_dir = exp_path.rstrip("/").split("/")[-1]
                    exp_dict = experiment_get(exp_dir)
                    if exp_dict:
                        experiments.append(exp_dict)
        except Exception:
            pass
    return experiments


def experiment_create(name: str, config: dict) -> str:
    Experiment.create_with_config(name, config)
    return name


def experiment_get(id):
    try:
        exp = Experiment.get(id)
        data = exp.get_json_data()
        # Parse config field from JSON string to dict if needed
        config = data.get("config", {})
        if isinstance(config, str):
            try:
                data["config"] = json.loads(config)
            except json.JSONDecodeError:
                data["config"] = {}
        return data
    except Exception as e:
        print(f"Error getting experiment {id}: {e}")


def experiment_delete(id):
    try:
        exp = Experiment.get(id)
        exp.delete()
    except Exception as e:
        print(f"Error deleting experiment {id}: {e}")


def experiment_update(id, config):
    try:
        exp = Experiment.get(id)
        exp.update_config(config)
    except Exception as e:
        print(f"Error updating experiment {id}: {e}")


def experiment_update_config(id, key, value):
    try:
        exp = Experiment.get(id)
        exp.update_config_field(key, value)
    except Exception as e:
        print(f"Error updating experiment config key {key}: {e}")


def experiment_save_prompt_template(id, template):
    try:
        exp_obj = Experiment.get(id)
        exp_obj.update_config_field("prompt_template", template)
    except Exception as e:
        print(f"Error saving prompt template: {e}")


def experiment_update_configs(id, updates: dict):
    try:
        exp_obj = Experiment.get(id)
        exp_obj.update_config(updates)
    except Exception as e:
        print(f"Error updating experiment config: {e}")
