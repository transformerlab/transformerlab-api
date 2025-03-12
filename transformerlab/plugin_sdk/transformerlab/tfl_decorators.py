import argparse
import functools
import traceback
from typing import List, Any
from datasets import load_dataset

from transformerlab.plugin import Job, get_dataset_path, test_wandb_login


class TFLPlugin:
    """Decorator class for TransformerLab plugins with automatic argument handling"""

    def __init__(self):
        self._job = None
        self._parser = argparse.ArgumentParser(description="TransformerLab Plugin")
        self._parser.add_argument("--job_id", type=str, required=True, help="Job identifier")
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

                # Update starting progress
                self.job.update_progress(progress_start)

                try:
                    # Call the wrapped function
                    result = func(*args, **kwargs)

                    # Update final progress and success status
                    self.job.update_progress(progress_end)
                    self.job.set_job_completion_status("success", f"{func.__name__} completed successfully")

                    return result

                except Exception as e:
                    # Capture the full error
                    error_msg = f"Error in {func.__name__}: {str(e)}\n{traceback.format_exc()}"
                    print(error_msg)

                    # Update job with failure status
                    self.job.set_job_completion_status("failed", error_msg)

                    # Re-raise the exception
                    raise

            return wrapper

        return decorator

    def load_dataset(self, dataset_types: List[str] = ["train"]):
        """Decorator for loading datasets with error handling"""

        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # Ensure args are parsed
                self._ensure_args_parsed()

                if not hasattr(self, "dataset_name") or not self.dataset_name:
                    self.job.set_job_completion_status("failed", "Dataset name not provided")
                    raise ValueError("Dataset name not provided")

                try:
                    # Get the dataset path/ID
                    dataset_target = get_dataset_path(self.dataset_name)

                    # Load each dataset split
                    datasets = {}
                    for dataset_type in dataset_types:
                        datasets[dataset_type] = load_dataset(
                            dataset_target, split=dataset_type, trust_remote_code=True
                        )

                    # Call the wrapped function with the datasets
                    return func(datasets, *args, **kwargs)

                except Exception as e:
                    error_msg = f"Error loading dataset: {str(e)}\n{traceback.format_exc()}"
                    print(error_msg)
                    self.job.set_job_completion_status("failed", error_msg)
                    raise

            return wrapper

        return decorator

    def progress_update(self, progress: int):
        """Update job progress"""
        self.job.update_progress(progress)

    def add_job_data(self, key: str, value: Any):
        """Add data to job"""
        self.job.add_to_job_data(key, value)


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
                if not hasattr(self, key):
                    setattr(self, key, value)

        except Exception as e:
            error_msg = f"Error loading configuration: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            self.job.set_job_completion_status("failed", error_msg)
            raise

    def setup_wandb(self, project_name: str = "TFL_Training"):
        """Decorator for setting up Weights & Biases logging"""

        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                self._ensure_args_parsed()

                # Check config or direct attribute for wandb logging preference
                log_to_wandb = False
                if hasattr(self, "_config") and self._config:
                    log_to_wandb = self._config.get("log_to_wandb", False)
                elif hasattr(self, "log_to_wandb"):
                    log_to_wandb = self.log_to_wandb

                report_to = ["tensorboard"]

                if log_to_wandb:
                    try:
                        wandb_success, report_to = test_wandb_login(project_name)

                        if wandb_success:
                            print(f"W&B logging enabled (project: {project_name})")
                            self.add_job_data("wandb_logging", True)
                        else:
                            print("W&B API key not found. W&B logging disabled.")
                            self.add_job_data("wandb_logging", False)

                        # Pass report_to as kwarg to the function
                        kwargs["report_to"] = report_to

                    except Exception as e:
                        print(f"Error setting up W&B: {str(e)}. Continuing without W&B.")
                        self.add_job_data("wandb_logging", False)
                        kwargs["report_to"] = ["tensorboard"]
                else:
                    kwargs["report_to"] = ["tensorboard"]

                return func(*args, **kwargs)

            return wrapper

        return decorator


# Create a singleton instance
tfl = TFLPlugin()
tfl_trainer = TrainerTFLPlugin()
