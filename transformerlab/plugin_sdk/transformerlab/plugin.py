import os
import json
import sqlite3
import itertools
import sys
import logging
from pathlib import Path


# useful constants
WORKSPACE_DIR = os.getenv("_TFL_WORKSPACE_DIR")
if WORKSPACE_DIR is None:
    print("Plugin Harness Error: Environment variable _TFL_WORKSPACE_DIR is not set. Quitting.")
    exit(1)
TEMP_DIR = os.path.join(WORKSPACE_DIR, "temp")

# Maintain a singleton database connection
db = None


def get_db_connection():
    """
    Returns an SQLite DB connection to the Transformer Lab DB
    """
    global db
    if db is None:
        dbfile = os.path.join(WORKSPACE_DIR, "llmlab.sqlite3")
        db = sqlite3.connect(dbfile, isolation_level=None)

        # Need to set these every time we open a connection
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA synchronous=normal")
        db.execute("PRAGMA busy_timeout=30000")
    return db


def get_dataset_path(dataset_id: str):
    """
    Returns the ID or filesystem path to pass to load_dataset() for a given ID.
    """
    db = get_db_connection()
    cursor = db.execute("SELECT location FROM dataset WHERE dataset_id = ?", (dataset_id,))
    row = cursor.fetchone()
    cursor.close()

    # if no rows exist then the dataset hasn't been installed!
    if row is None:
        raise Exception(f"No dataset named {dataset_id} installed.")

    # dataset_location will be either "local" or "huggingface"
    # (and if it's something else we're going to treat "huggingface" as default)
    # if it's local then pass it the path to the dataset directory
    dataset_location = row[0]
    if dataset_location == "local":
        return os.path.join(WORKSPACE_DIR, "datasets", dataset_id)

    # Otherwise assume it is a Huggingface ID
    else:
        return dataset_id


def get_db_config_value(key: str):
    """
    Returns the value of a config key from the database.
    """
    db = get_db_connection()
    cursor = db.execute("SELECT value FROM config WHERE key = ?", (key,))
    row = cursor.fetchone()
    cursor.close()

    if row is None:
        return None
    return row[0]


def test_wandb_login(project_name: str = "TFL_Training"):
    import netrc
    from pathlib import Path

    netrc_path = Path.home() / (".netrc" if os.name != "nt" else "_netrc")
    if netrc_path.exists():
        auth = netrc.netrc(netrc_path).authenticators("api.wandb.ai")
        if auth:
            os.environ["WANDB_PROJECT"] = project_name
            os.environ["WANDB_DISABLED"] = "false"
            report_to = ["tensorboard", "wandb"]
            return True, report_to
        else:
            os.environ["WANDB_DISABLED"] = "true"
            return False, ["tensorboard"]
    else:
        os.environ["WANDB_DISABLED"] = "true"
        return False, ["tensorboard"]


def experiment_get_by_name(name):
    db = get_db_connection()
    cursor = db.execute("SELECT * FROM experiment WHERE name = ?", (name,))
    row = cursor.fetchone()

    if row is None:
        return None

    # Convert the SQLite row into a JSON object with keys
    desc = cursor.description
    column_names = [col[0] for col in desc]
    row_dict = dict(itertools.zip_longest(column_names, row))

    # Exclude created_at and updated_at fields to match our SQLAlchemy utility
    excluded_fields = {"created_at", "updated_at"}
    result = {k: v for k, v in row_dict.items() if k not in excluded_fields}

    cursor.close()
    return result


def experiment_get(id):
    db = get_db_connection()
    if id is None or id == "undefined":
        return None
    cursor = db.execute("SELECT * FROM experiment WHERE id = ?", (id,))
    row = cursor.fetchone()

    if row is None:
        return None

    # Convert the SQLite row into a JSON object with keys
    desc = cursor.description
    column_names = [col[0] for col in desc]
    row_dict = dict(itertools.zip_longest(column_names, row))

    # Exclude created_at and updated_at fields to match our SQLAlchemy utility
    excluded_fields = {"created_at", "updated_at"}
    result = {k: v for k, v in row_dict.items() if k not in excluded_fields}

    cursor.close()
    return result


def get_experiment_id_from_name(name: str):
    """
    Returns the experiment ID from the experiment name.
    """
    if isinstance(name, str):
        data = experiment_get_by_name(name)
        if data is None:
            return name
        return data["id"]
    return name


def get_experiment_config(name: str):
    """
    Returns the experiment config from the experiment name.
    """
    if isinstance(name, str):
        experiment_id = get_experiment_id_from_name(name)
        data = experiment_get(experiment_id)
        if data is None:
            return None, experiment_id
        if not isinstance(data["config"], dict):
            return json.loads(data["config"]), experiment_id
        else:
            return data["config"], experiment_id

    return None, experiment_id


def get_python_executable(plugin_dir):
    """Check if a virtual environment exists and return the appropriate Python executable"""
    # Check for virtual environment in the plugin directory
    venv_path = os.path.join(plugin_dir, "venv")

    if os.path.isdir(venv_path):
        print("Virtual environment found, using it for evaluation...")
        # Determine the correct path to the Python executable based on the platform
        python_executable = os.path.join(venv_path, "bin", "python")

        if os.path.exists(python_executable):
            return python_executable

    # Fall back to system Python if venv not found or executable doesn't exist
    print("No virtual environment found, using system Python...")
    return sys.executable


class Job:
    """
    Used to update status and info of long-running jobs.
    """

    def __init__(self, job_id):
        self.id = job_id
        self.db = get_db_connection()
        self.should_stop = False

    def update_progress(self, progress: int):
        """
        Update the percent complete for this job.

        progress: int representing percent complete
        """
        # Remove manual updated_at - SQLAlchemy handles this automatically
        self.db.execute(
            "UPDATE job SET progress = ? WHERE id = ?",
            (progress, self.id),
        )

        # While we are updating the progress, check if we should stop by
        # checking if should_stop == True in the job_data:
        cursor = self.db.execute("SELECT job_data FROM job WHERE id = ?", (self.id,))
        row = cursor.fetchone()
        cursor.close()

        if row is not None and row[0] is not None:
            try:
                data = row[0]

                # Handle different data types that might come from SQLite/SQLAlchemy
                if isinstance(data, str):
                    # Try to parse as JSON
                    job_data = json.loads(data)

                    # Check if the result is still a string (double-encoded JSON)
                    if isinstance(job_data, str):
                        # Try to parse again
                        job_data = json.loads(job_data)
                elif isinstance(data, dict):
                    # Already a dictionary
                    job_data = data
                else:
                    # Some other type, skip stop check
                    return

                if job_data.get("stop", False):
                    self.should_stop = True

            except (json.JSONDecodeError, TypeError, ValueError):
                # If job_data is malformed, just continue
                pass

    def update_status(self, status: str):
        """
        Update the status of this job.

        status: str representing the status of the job
        """
        # Remove manual updated_at - SQLAlchemy handles this automatically
        self.db.execute(
            "UPDATE job SET status = ? WHERE id = ?",
            (status, self.id),
        )

    def get_status(self):
        """
        Get the status of this job.
        """
        cursor = self.db.execute("SELECT status FROM job WHERE id = ?", (self.id,))
        row = cursor.fetchone()
        cursor.close()

        if row is not None:
            return row[0]
        return None

    def get_progress(self):
        """
        Get the progress of this job.
        """
        cursor = self.db.execute("SELECT progress FROM job WHERE id = ?", (self.id,))
        row = cursor.fetchone()
        cursor.close()

        if row is not None:
            return row[0]
        return None

    def get_experiment_id(self):
        """
        Get the experiment_id of this job.
        """
        cursor = self.db.execute("SELECT experiment_id FROM job WHERE id = ?", (self.id,))
        row = cursor.fetchone()
        cursor.close()

        if row is not None:
            return row[0]
        return None

    def get_job_data(self):
        """
        Get the job_data of this job.
        """
        cursor = self.db.execute("SELECT job_data FROM job WHERE id = ?", (self.id,))
        row = cursor.fetchone()
        cursor.close()

        if row is not None and row[0] is not None:
            try:
                data = row[0]

                # Handle different data types that might come from SQLite/SQLAlchemy
                if isinstance(data, str):
                    # Try to parse as JSON
                    parsed = json.loads(data)

                    # Check if the result is still a string (double-encoded JSON)
                    if isinstance(parsed, str):
                        # Try to parse again
                        return json.loads(parsed)
                    else:
                        return parsed
                elif isinstance(data, dict):
                    # Already a dictionary
                    return data
                else:
                    # Some other type, try to convert to dict
                    return dict(data) if data else {}

            except (json.JSONDecodeError, TypeError, ValueError):
                # If job_data is malformed, return empty dict
                print(f"Warning: Malformed job_data for job {self.id}: {row[0]}")
                return {}
        return {}

    def set_tensorboard_output_dir(self, tensorboard_dir: str):
        """
        Sets the directory that tensorboard output is stored.
        """
        self.add_to_job_data("tensorboard_output_dir", tensorboard_dir)

    def add_to_job_data(self, key: str, value):
        """
        Adds a key-value pair to the job_data JSON object.
        """
        try:
            # Fetch current job_data
            cursor = self.db.execute("SELECT job_data FROM job WHERE id = ?", (self.id,))
            row = cursor.fetchone()
            cursor.close()

            job_data = {}
            if row and row[0] is not None:
                data = row[0]
                # Handle different data types that might come from SQLite/SQLAlchemy
                if isinstance(data, str):
                    try:
                        # Try to parse as JSON
                        job_data = json.loads(data)

                        # Check if the result is still a string (double-encoded JSON)
                        if isinstance(job_data, str):
                            # Try to parse again
                            job_data = json.loads(job_data)
                    except Exception:
                        job_data = {}
                elif isinstance(data, dict):
                    # Already a dictionary
                    job_data = data
                else:
                    job_data = {}

            # Update the key - handle different value types
            # if isinstance(value, str):
            #     # Try to parse as JSON, if that fails store as string
            #     try:
            #         job_data[key] = json.loads(value)
            # except (json.JSONDecodeError, TypeError):
            #     job_data[key] = value
            # else:
            # Store value as-is (dict, list, number, bool, etc.)
            job_data[key] = value

            # Save back as JSON
            self.db.execute(
                "UPDATE job SET job_data = ? WHERE id = ?",
                (json.dumps(job_data), self.id),
            )
        except Exception as e:
            print(f"Error adding to job data: {e}")

    def update_job_data(self, key: str, value):
        """
        Updates a key-value pair in the job_data JSON object.
        """
        try:
            # Fetch current job_data
            cursor = self.db.execute("SELECT job_data FROM job WHERE id = ?", (self.id,))
            row = cursor.fetchone()
            cursor.close()

            job_data = {}
            if row and row[0] is not None:
                data = row[0]
                # Handle different data types that might come from SQLite/SQLAlchemy
                if isinstance(data, str):
                    try:
                        # Try to parse as JSON
                        job_data = json.loads(data)

                        # Check if the result is still a string (double-encoded JSON)
                        if isinstance(job_data, str):
                            # Try to parse again
                            job_data = json.loads(job_data)
                    except Exception:
                        job_data = {}
                elif isinstance(data, dict):
                    # Already a dictionary
                    job_data = data
                else:
                    job_data = {}

            # Update the key - handle different value types
            if isinstance(value, str):
                # Try to parse as JSON, if that fails store as string
                try:
                    job_data[key] = json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    job_data[key] = value
            else:
                # Store value as-is (dict, list, number, bool, etc.)
                job_data[key] = value

            # Save back as JSON
            self.db.execute(
                "UPDATE job SET job_data = ? WHERE id = ?",
                (json.dumps(job_data), self.id),
            )
        except Exception as e:
            print(f"Error updating job data: {e}")

    def set_job_completion_status(
        self,
        completion_status: str,
        completion_details: str,
        score: dict = None,
        additional_output_path: str = None,
        plot_data_path: str = None,
    ):
        """
        A job could be in the "complete" state but still have failed, so this
        function is used to set the job completion status. i.e. how the task
        that the job is executing has completed.
        and if the job failed, the details of the failure.
        Score should be a json of the format {"metric_name": value, ...}
        """
        try:
            if completion_status not in ("success", "failed"):
                raise ValueError("completion_status must be either 'success' or 'failed'")

            # Add to job data completion_status and completion_details
            self.add_to_job_data("completion_status", completion_status)
            self.add_to_job_data("completion_details", completion_details)
            
            # Update the job status field if there's a failure
            if completion_status == "failed":
                self.update_status("FAILED")

            # Update the job status field if there's a failure
            if completion_status == "failed":
                self.update_status("FAILED")

            # # Initialize job_data as empty JSON object if it's NULL
            # self.db.execute(
            #     "UPDATE job SET job_data = COALESCE(job_data, '{}') WHERE id = ? AND job_data IS NULL", (self.id,)
            # )

            # Determine if additional_output_path is valid
            valid_output_path = (
                additional_output_path if additional_output_path and additional_output_path.strip() != "" else None
            )

            valid_plot_data_path = plot_data_path if plot_data_path and plot_data_path.strip() != "" else None

            # # Build the SQL query and parameters dynamically using json_set
            # sql = "UPDATE job SET job_data = json_set(job_data, '$.completion_status', ?, '$.completion_details', ?"
            # params = [completion_status, completion_details]

            if score is not None:
                self.add_to_job_data("score", score)

            if valid_output_path is not None:
                self.add_to_job_data("additional_output_path", valid_output_path)

            if valid_plot_data_path is not None:
                self.add_to_job_data("plot_data_path", valid_plot_data_path)

            # sql += ") WHERE id = ?"
            # params.append(self.id)

            # self.db.execute(sql, tuple(params))
        except Exception as e:
            print(f"Error setting job completion status: {e}")


def generate_model_json(
    model_id: str,
    architecture: str,
    model_filename: str = "",
    output_directory: str | None = None,
    json_data: dict = {},
):
    """
    The generates the json file needed for a model to be read in the models directory.

    model_id: ID of the model without author prefix. This will also be the directory the file is output to.
    architecture: A string that is used to determine which plugins support this model.
    filename: (Optional) A string representing model_filename or "" if none.
    output_directory: (Optional) The directory to output this file. Otherwise TLab models directory.
    json_data: (Default empty) A dictionary of values to add to the json_data of this model.

    Returns the object used to generate the JSON.
    """
    model_description = {
        "model_id": f"TransformerLab/{model_id}",
        "model_filename": model_filename,
        "name": model_id,
        "local_model": True,
        "json_data": {
            "uniqueID": f"TransformerLab/{model_id}",
            "name": model_id,
            "model_filename": model_filename,
            "description": "Generated by Transformer Lab.",
            "source": "transformerlab",
            "architecture": architecture,
            "huggingface_repo": "",
        },
    }

    # Add and update any fields passed in json_data object
    # This overwrites anything defined above with values passed in
    model_description["json_data"].update(json_data)

    # Output the json to the file
    if not output_directory:
        output_directory = os.path.join(WORKSPACE_DIR, "models", model_id)
    with open(os.path.join(output_directory, "info.json"), "w") as outfile:
        json.dump(model_description, outfile)

    return model_description


def setup_model_worker_logger(name: str = "transformerlab") -> logging.Logger:
    """
    Configure and assign a dedicated logger for the FastChat model worker.

    This function initializes a logger that writes to the global TransformerLab log file
    and assigns it to FastChat's base_model_worker, enabling centralized logging.

    Args:
        name (str): The name of the logger to initialize.

    Returns:
        logging.Logger: A configured logger instance bound to FastChat's model worker.
    """

    from fastchat.utils import build_logger
    import fastchat.serve.base_model_worker

    if "TFL_HOME_DIR" in os.environ:
        HOME_DIR = os.environ["TFL_HOME_DIR"]
        if not os.path.exists(HOME_DIR):
            print(f"Error: Home directory {HOME_DIR} does not exist")
            os.makedirs(HOME_DIR, exist_ok=True)
        print(f"Home directory is set to: {HOME_DIR}")
    else:
        HOME_DIR = Path.home() / ".transformerlab"
        os.makedirs(name=HOME_DIR, exist_ok=True)
        print(f"Using default home directory: {HOME_DIR}")

    GLOBAL_LOG_PATH = os.path.join(HOME_DIR, "transformerlab.log")

    logger: logging.Logger = build_logger(name, GLOBAL_LOG_PATH)

    # Patch FastChat's base model worker logger
    fastchat.serve.base_model_worker.logger = logger

    return logger
