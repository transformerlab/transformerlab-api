import os
import json
import sqlite3

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


def set_db_config_value(key: str, value: str):
    """
    Sets the value of a config key in the database.
    """
    db = get_db_connection()
    cursor = db.execute("INSERT OR REPLACE INTO config (key, value) VALUES (?, ?)", (key, value))
    cursor.close()





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
        self.db.execute(
            "UPDATE job SET progress = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (progress, self.id),
        )

        # While we are updating the progress, check if we should stop by
        # checking if should_stop == True in the job_data:
        cursor = self.db.execute("SELECT job_data FROM job WHERE id = ?", (self.id,))
        row = cursor.fetchone()
        cursor.close()

        if row is not None:
            job_data = json.loads(row[0])
            if job_data.get("stop", False):
                self.should_stop = True

    def update_status(self, status: str):
        """
        Update the status of this job.

        status: str representing the status of the job
        """
        self.db.execute(
            "UPDATE job SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (status, self.id),
        )

    def set_tensorboard_output_dir(self, tensorboard_dir: str):
        """
        Sets the directory that tensorboard output is stored.
        """
        self.db.execute(
            "UPDATE job SET job_data = json_insert(job_data, '$.tensorboard_output_dir', ?) WHERE id = ?",
            (tensorboard_dir, self.id),
        )

    def add_to_job_data(self, key: str, value: str):
        """
        Adds a key-value pair to the job_data JSON object.
        """
        try:
            self.db.execute(
                "UPDATE job SET job_data = json_insert(job_data, '$." + key + "', ?) WHERE id = ?",
                (value, self.id),
            )
        except Exception as e:
            print(f"Error adding to job data: {e}")

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

            # Determine if additional_output_path is valid
            valid_output_path = (
                additional_output_path if additional_output_path and additional_output_path.strip() != "" else None
            )

            valid_plot_data_path = plot_data_path if plot_data_path and plot_data_path.strip() != "" else None

            # Build the SQL query and parameters dynamically.
            sql = "UPDATE job SET job_data = json_insert(job_data, '$.completion_status', ?, '$.completion_details', ?"
            params = [completion_status, completion_details]

            if score is not None:
                score = json.dumps(score)
                sql += ", '$.score', ?"
                params.append(score)

            if valid_output_path is not None:
                sql += ", '$.additional_output_path', ?"
                params.append(valid_output_path)

            if valid_plot_data_path is not None:
                sql += ", '$.plot_data_path', ?"
                params.append(valid_plot_data_path)

            sql += ") WHERE id = ?"
            params.append(self.id)

            self.db.execute(sql, tuple(params))
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
        "model_id": f"TransformerLab-mlx/{model_id}",
        "model_filename": model_filename,
        "name": model_id,
        "local_model": True,
        "json_data": {
            "uniqueID": f"TransformerLab-mlx/{model_id}",
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
