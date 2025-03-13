import os
import time
import argparse
import functools
import traceback
from typing import List, Any
import json
from tensorboardX import SummaryWriter
from datasets import load_dataset, get_dataset_split_names

from transformerlab.plugin import Job, get_dataset_path, test_wandb_login


class TFLPlugin:
    """Decorator class for TransformerLab plugins with automatic argument handling"""

    def __init__(self):
        self._job = None
        self._parser = argparse.ArgumentParser(description="TransformerLab Plugin")
        self._parser.add_argument("--job_id", type=str, help="Job identifier")
        self._parser.add_argument("--dataset_name", type=str, help="Dataset to use")
        self._parser.add_argument("--model_name", type=str, help="Model to use")

        # Flag to track if args have been parsed
        self._args_parsed = False

    @property
    def job(self):
        """Get the job object, initializing if necessary"""
        if not self._job:
            self._ensure_args_parsed()
            self._job = Job(self.job_id)
        return self._job

    def _ensure_args_parsed(self):
        """Parse arguments if not already done"""
        if not self._args_parsed:
            args, _ = self._parser.parse_known_args()
            # Transfer all arguments to attributes of self
            for key, value in vars(args).items():
                setattr(self, key, value)
            self._args_parsed = True

    def add_argument(self, *args, **kwargs):
        """Add an argument to the parser"""
        self._parser.add_argument(*args, **kwargs)
        return self

    def job_wrapper(self, progress_start: int = 0, progress_end: int = 100):
        """Decorator for wrapping a function with job status updates"""

        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # Ensure args are parsed and job is initialized
                self._ensure_args_parsed()

                self.add_job_data("start_time", time.strftime("%Y-%m-%d %H:%M:%S"))

                # Update starting progress
                self.job.update_progress(progress_start)

                try:
                    # Call the wrapped function
                    result = func(*args, **kwargs)

                    # Update final progress and success status
                    self.job.update_progress(progress_end)
                    self.job.set_job_completion_status("success", "Job completed successfully")

                    return result

                except Exception as e:
                    # Capture the full error
                    error_msg = f"Error in Job: {str(e)}\n{traceback.format_exc()}"
                    print(error_msg)

                    # Update job with failure status
                    self.job.set_job_completion_status("failed", "Error occurred while executing job")
                    self.add_job_data("end_time", time.strftime("%Y-%m-%d %H:%M:%S"))

                    # Re-raise the exception
                    raise

            return wrapper

        return decorator

    def progress_update(self, progress: int):
        """Update job progress"""
        self.job.update_progress(progress)
        if self.job.should_stop:
            self.job.update_status("STOPPED")
            raise KeyboardInterrupt("Job stopped by user")

    def add_job_data(self, key: str, value: Any):
        """Add data to job"""
        self.job.add_to_job_data(key, value)

    def load_dataset(self, dataset_types: List[str] = ["train"]):
        """Decorator for loading datasets with error handling"""

        self._ensure_args_parsed()

        if not hasattr(self, "dataset_name") or not self.dataset_name:
            self.job.set_job_completion_status("failed", "Dataset name not provided")
            self.add_job_data("end_time", time.strftime("%Y-%m-%d %H:%M:%S"))
            raise ValueError("Dataset name not provided")

        try:
            # Get the dataset path/ID
            dataset_target = get_dataset_path(self.dataset_name)
            # Get the available splits
            available_splits = get_dataset_split_names(dataset_target)

            # Handle different validation split names
            dataset_splits = {}
            for dataset_type in dataset_types:
                dataset_splits[dataset_type] = dataset_type

            if "validation" in available_splits and "valid" in dataset_splits:
                dataset_splits["valid"] = "validation"
            elif "valid" in dataset_types and "valid" not in available_splits:
                print("No validation slice found in dataset, using train split as 80-20 for training and validation")
                dataset_splits["valid"] = "train[-10%:]"
                dataset_splits["train"] = "train[:80%]"

            # Load each dataset split
            datasets = {}
            for dataset_type in dataset_splits:
                datasets[dataset_type] = load_dataset(
                    dataset_target, split=dataset_splits[dataset_type], trust_remote_code=True
                )

            print(f"Loaded train dataset with {len(datasets['train'])} examples.")

            if "valid" in dataset_types:
                print(f"Loaded valid dataset with {len(datasets['valid'])} examples.")

            return datasets

        except Exception as e:
            error_msg = f"Error loading dataset: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            self.job.set_job_completion_status("failed", "Failed to load dataset")
            self.add_job_data("end_time", time.strftime("%Y-%m-%d %H:%M:%S"))
            raise


class TrainerTFLPlugin(TFLPlugin):
    """Enhanced decorator class for TransformerLab training plugins"""

    def __init__(self):
        super().__init__()
        # Add training-specific default arguments
        self._parser.add_argument("--input_file", type=str, help="Path to configuration file")

        # Training state tracking
        self._config_parsed = False

    def _ensure_args_parsed(self):
        """Parse arguments if not already done"""
        if not self._args_parsed:
            args, _ = self._parser.parse_known_args()
            # Transfer all arguments to attributes of self
            for key, value in vars(args).items():
                setattr(self, key, value)
            self._args_parsed = True

        if self._args_parsed and not self._config_parsed:
            if hasattr(self, "input_file") and self.input_file:
                self.load_config()
                self._config_parsed = True

    def create_progress_callback(self, framework="huggingface", **kwargs):
        """
        Create a progress callback for various ML frameworks.

        Args:
            framework: The framework to create a callback for (e.g., "huggingface")
            **kwargs: Additional arguments specific to the callback

        Returns:
            A callback object compatible with the specified framework
        """
        self._ensure_args_parsed()

        if framework.lower() == "huggingface" or framework.lower() == "hf":
            # Import here to avoid dependency issues if HF isn't installed
            try:
                from transformers import TrainerCallback

                class TFLProgressCallback(TrainerCallback):
                    """Callback that updates progress in TransformerLab DB during HuggingFace training"""

                    def __init__(self, tfl_instance):
                        self.tfl = tfl_instance

                    def on_step_end(self, args, state, control, **callback_kwargs):
                        if state.is_local_process_zero:
                            if state.max_steps > 0:
                                progress = state.global_step / state.max_steps
                                progress = int(progress * 100)
                                self.tfl.progress_update(progress)

                                # Check if job should be stopped
                                if self.tfl.job.should_stop:
                                    control.should_training_stop = True

                        return control

                return TFLProgressCallback(self)

            except ImportError:
                raise ImportError("Could not create HuggingFace callback. Please install transformers package.")

        else:
            raise ValueError(f"Unsupported framework: {framework}. Supported frameworks: huggingface")

    def load_config(self):
        """Decorator for loading configuration from input file"""

        try:
            import json

            # Load configuration from file
            with open(self.input_file) as json_file:
                input_config = json.load(json_file)

            if "config" in input_config:
                self._config = input_config["config"]
            else:
                self._config = input_config

            # Transfer config values to instance attributes for easy access
            for key, value in self._config.items():
                if not hasattr(self, key) or getattr(self, key) is None:
                    setattr(self, key, value)

        except Exception as e:
            error_msg = f"Error loading configuration: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            self.job.set_job_completion_status("failed", "Error loading configuration")
            self.add_job_data("end_time", time.strftime("%Y-%m-%d %H:%M:%S"))
            raise

    def setup_train_logging(self, wandb_project_name: str = "TFL_Training"):
        """Setup Weights and Biases and TensorBoard logging

        Args:
            wandb_project_name: Name of the W&B project

        Returns:
            List of reporting targets (e.g. ["tensorboard", "wandb"])
        """
        self._ensure_args_parsed()
        if not hasattr(self, "template_name") or not self.template_name:
            self.template_name = "default"
        # Add tensorboard_output_dir
        self.tensorboard_output_dir = os.path.join(self.output_dir, f"job_{self.job_id}_{self.template_name}")
        print("Writing tensorboard logs to:", self.output_dir)

        # Store the tensorboard output dir in the job
        self.add_job_data("tensorboard_output_dir", self.output_dir)

        # Check config or direct attribute for wandb logging preference
        log_to_wandb = False
        if hasattr(self, "_config") and self._config:
            log_to_wandb = self._config.get("log_to_wandb", False)
        elif hasattr(self, "log_to_wandb"):
            log_to_wandb = self.log_to_wandb

        report_to = ["tensorboard"]

        if log_to_wandb:
            try:
                wandb_success, report_to = test_wandb_login(wandb_project_name)

                if wandb_success:
                    print(f"W&B logging enabled (project: {wandb_project_name})")
                    report_to.append("wandb")
                else:
                    print("W&B API key not found. W&B logging disabled.")
                    self.add_job_data("wandb_logging", False)

            except Exception as e:
                print(f"Error setting up W&B: {str(e)}. Continuing without W&B.")
                self.add_job_data("wandb_logging", False)
                report_to = ["tensorboard"]

        return report_to


class EvalsTFLPlugin(TFLPlugin):
    """Enhanced decorator class for TransformerLab evaluation plugins"""

    def __init__(self):
        super().__init__()
        # Add common evaluation-specific arguments
        self._parser.add_argument("--run_name", default="evaluation", type=str, help="Name for the evaluation run")
        self._parser.add_argument("--experiment_name", default="", type=str, help="Name of the experiment")
        self._parser.add_argument("--eval_name", default="", type=str, help="Name of the evaluation")

        # Dictionary to store parsed tasks
        self._parsed_tasks = None
        self._predefined_tasks = {}
        self._unknown_args_dict = {}

    def _ensure_args_parsed(self):
        """Parse arguments if not already done"""
        if not self._args_parsed:
            args, unknown_args = self._parser.parse_known_args()

            # Transfer all known arguments to attributes of self
            for key, value in vars(args).items():
                setattr(self, key, value)

            self._parse_unknown_args(unknown_args)
            self._args_parsed = True

    def _parse_unknown_args(self, unknown_args):
        """Parse unknown arguments which change with each eval"""
        key = None
        for arg in unknown_args:
            if arg.startswith("--"):  # Argument key
                key = arg.lstrip("-")
                setattr(self, key, True)
            elif key:  # Argument value
                setattr(self, key, arg)
                key = None

    def setup_eval_logging(self):
        """Setup TensorBoard logging for evaluations

        Returns:
            str: Path to the TensorBoard output directory
        """
        self._ensure_args_parsed()

        today = time.strftime("%Y%m%d-%H%M%S")
        workspace_dir = os.environ.get("_TFL_WORKSPACE_DIR", "./")

        # Create tensorboard directory structure
        tensorboard_dir = os.path.join(workspace_dir, "experiments", self.experiment_name, "tensorboards")
        os.makedirs(tensorboard_dir, exist_ok=True)

        # Find directory based on eval name
        combined_dir = None
        for dir_name in os.listdir(tensorboard_dir):
            if self.run_name == dir_name or self.run_name == dir_name.lower():
                combined_dir = os.path.join(tensorboard_dir, dir_name)
                break

        if combined_dir is None:
            combined_dir = os.path.join(tensorboard_dir, self.run_name)

        output_dir = os.path.join(combined_dir, f"evaljob_{self.job_id}_{today}")
        os.makedirs(output_dir, exist_ok=True)

        # Store the writer and output directory as instance variables
        self.tensorboard_output_dir = output_dir

        # Create writer and store it
        self.writer = SummaryWriter(output_dir)

        # Store the output directory in the job
        self.job.set_tensorboard_output_dir(output_dir)

        print(f"Writing tensorboard logs to {output_dir}")
        return output_dir

    def log_metric(self, metric_name, value, step=1):
        """Log a metric to TensorBoard

        Args:
            metric_name: Name of the metric
            value: Value of the metric
            step: Step number for TensorBoard
        """
        if hasattr(self, "writer"):
            self.writer.add_scalar(f"eval/{metric_name}", value, step)
        else:
            print("Warning: TensorBoard writer not initialized. Call setup_eval_logging first.")

    def get_output_file_path(self, suffix="", is_plotting=False, dir_only=False):
        """Get path for saving evaluation outputs

        Args:
            suffix: Optional suffix for the filename
            is_plotting: Whether this is for plotting data (uses .json extension)

        Returns:
            str: Full path for output file
        """
        import os

        self._ensure_args_parsed()

        workspace_dir = os.environ.get("_TFL_WORKSPACE_DIR", "./")
        experiment_dir = os.path.join(workspace_dir, "experiments", self.experiment_name)
        eval_dir = os.path.join(experiment_dir, "evals", self.eval_name, self.job_id)

        os.makedirs(eval_dir, exist_ok=True)

        if dir_only:
            print("EVAL DIR", eval_dir)
            return eval_dir

        if is_plotting:
            # For plotting data, we use a JSON file with a specific naming convention
            plotting_suffix = suffix if suffix else "plotting"
            if not plotting_suffix.endswith(".json"):
                plotting_suffix += ".json"
            return os.path.join(eval_dir, f"plot_data_{self.job_id}_{plotting_suffix}")
        else:
            # For regular outputs
            if suffix:
                if not any(suffix.endswith(ext) for ext in (".csv", ".json", ".txt")):
                    suffix += ".csv"
                return os.path.join(eval_dir, f"output_{self.job_id}_{suffix}")
            else:
                return os.path.join(eval_dir, f"output_{self.job_id}.csv")

    def save_evaluation_results(self, metrics_df):
        """Save evaluation results and generate plotting data

        Args:
            metrics_df: DataFrame containing evaluation metrics with
                       required columns "test_case_id", "metric_name", "score"

        Returns:
            tuple: Paths to the saved files (output_path, plot_data_path)

        Raises:
            ValueError: If required columns are missing from the DataFrame
        """
        # Validate that required columns exist
        required_columns = ["test_case_id", "metric_name", "score"]
        missing_columns = [col for col in required_columns if col not in metrics_df.columns]

        if missing_columns:
            raise ValueError(f"Missing required columns in metrics DataFrame: {missing_columns}")

        # Save full DataFrame to CSV
        output_path = self.get_output_file_path()
        metrics_df.to_csv(output_path, index=False)
        print(f"Saved detailed evaluation results to {output_path}")

        # Create and save plotting data
        plot_data_path = self.get_output_file_path(is_plotting=True)

        # Extract and format plotting data
        plotting_data = metrics_df[["test_case_id", "metric_name", "score"]].copy()

        # Format metric names for better display (replace underscores with spaces and capitalize)
        plotting_data["metric_name"] = plotting_data["metric_name"].apply(lambda x: x.replace("_", " ").title())

        # Save as JSON
        plotting_data.to_json(plot_data_path, orient="records", lines=False)
        print(f"Saved plotting data to {plot_data_path}")

        self.job.add_to_job_data("additional_output_path", output_path)
        self.job.add_to_job_data("plot_data_path", plot_data_path)

        # Print average scores by metric
        print("\n===== Evaluation Results =====")
        metrics = metrics_df["metric_name"].unique()
        score_list = []
        for metric in metrics:
            avg_score = metrics_df[metrics_df["metric_name"] == metric]["score"].mean()
            print(f"Average {metric}: {avg_score:.4f}")
            score_list.append({"type": metric, "score": avg_score})

        self.job.add_to_job_data("score", json.dumps(score_list))

        return output_path, plot_data_path


# Create singleton instances
tfl = TFLPlugin()
tfl_trainer = TrainerTFLPlugin()
tfl_evals = EvalsTFLPlugin()
