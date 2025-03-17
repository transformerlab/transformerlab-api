import time
import argparse
import functools
import asyncio
import traceback
from typing import List, Any, Dict
from datasets import load_dataset, get_dataset_split_names

from transformerlab.plugin import Job, get_dataset_path


class DotDict(dict):
    """Dictionary subclass that allows attribute access to dictionary keys"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class TLabPlugin:
    """Decorator class for TransformerLab plugins with automatic argument handling"""

    def __init__(self):
        self._job = None
        self._parser = argparse.ArgumentParser(description="TransformerLab Plugin")
        self._parser.add_argument("--job_id", type=str, help="Job identifier")
        self._parser.add_argument("--dataset_name", type=str, help="Dataset to use")
        self._parser.add_argument("--model_name", type=str, help="Model to use")

        # Store all parsed arguments in this dictionary
        self.params = DotDict()

        # Flag to track if args have been parsed
        self._args_parsed = False

    @property
    def job(self):
        """Get the job object, initializing if necessary"""
        if not self._job:
            self._ensure_args_parsed()
            self._job = Job(self.params.job_id)
        return self._job

    def _ensure_args_parsed(self):
        """Parse arguments if not already done"""
        if not self._args_parsed:
            args, _ = self._parser.parse_known_args()
            # Store all arguments in the parameters dictionary
            self.params = vars(args)
            self._args_parsed = True

    def add_argument(self, *args, **kwargs):
        """Add an argument to the parser"""
        self._parser.add_argument(*args, **kwargs)
        return self

    def job_wrapper(
        self,
        progress_start: int = 0,
        progress_end: int = 100,
        wandb_project_name: str = "TLab_Training",
        manual_logging: bool = False,
    ):
        """Decorator for wrapping a function with job status updates"""

        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # Ensure args are parsed and job is initialized
                self._ensure_args_parsed()

                self.add_job_data("start_time", time.strftime("%Y-%m-%d %H:%M:%S"))
                self.add_job_data("model_name", self.params.model_name)
                self.add_job_data("template_name", self.params.template_name)

                # Update starting progress
                self.job.update_progress(progress_start)

                try:
                    # Setup logging
                    if self.tlab_plugin_type == "trainer":
                        self.setup_train_logging(wandb_project_name=wandb_project_name, manual_logging=manual_logging)
                    # Call the wrapped function
                    result = func(*args, **kwargs)

                    # Update final progress and success status
                    self.job.update_progress(progress_end)
                    self.job.set_job_completion_status("success", "Job completed successfully")
                    self.add_job_data("end_time", time.strftime("%Y-%m-%d %H:%M:%S"))
                    if manual_logging and getattr(self.params, "wandb_run") is not None:
                        self.wandb_run.finish()

                    return result

                except Exception as e:
                    # Capture the full error
                    error_msg = f"Error in Job: {str(e)}\n{traceback.format_exc()}"
                    print(error_msg)

                    # Update job with failure status
                    self.job.set_job_completion_status("failed", "Error occurred while executing job")
                    self.add_job_data("end_time", time.strftime("%Y-%m-%d %H:%M:%S"))
                    if manual_logging and getattr(self.params, "wandb_run") is not None:
                        self.wandb_run.finish()

                    # Re-raise the exception
                    raise

            return wrapper

        return decorator

    def async_job_wrapper(
        self,
        progress_start: int = 0,
        progress_end: int = 100,
        wandb_project_name: str = "TLab_Training",
        manual_logging=False,
    ):
        """Decorator for wrapping an async function with job status updates"""

        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # Ensure args are parsed and job is initialized
                try:
                    self._ensure_args_parsed()
                except Exception as e:
                    print(f"Error parsing arguments: {str(e)}")
                    raise

                self.add_job_data("start_time", time.strftime("%Y-%m-%d %H:%M:%S"))
                self.add_job_data("model_name", self.params.model_name)
                self.add_job_data("template_name", self.params.template_name)

                # Update starting progress
                self.job.update_progress(progress_start)

                async def run_async():
                    try:
                        # Setup logging
                        if self.tlab_plugin_type == "trainer":
                            self.setup_train_logging(
                                wandb_project_name=wandb_project_name, manual_logging=manual_logging
                            )

                        # Call the wrapped async function
                        result = await func(*args, **kwargs)

                        # Update final progress and success status
                        self.job.update_progress(progress_end)
                        self.job.set_job_completion_status("success", "Job completed successfully")
                        self.add_job_data("end_time", time.strftime("%Y-%m-%d %H:%M:%S"))
                        if manual_logging and getattr(self, "wandb_run") is not None:
                            self.wandb_run.finish()

                        return result

                    except Exception as e:
                        # Capture the full error
                        error_msg = f"Error in Async Job: {str(e)}\n{traceback.format_exc()}"
                        print(error_msg)

                        # Update job with failure status
                        self.job.set_job_completion_status("failed", "Error occurred while executing job")
                        self.add_job_data("end_time", time.strftime("%Y-%m-%d %H:%M:%S"))
                        if manual_logging and getattr(self, "wandb_run") is not None:
                            self.wandb_run.finish()

                        # Re-raise the exception
                        raise

                # Use asyncio.run() inside the wrapper
                return asyncio.run(run_async())

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

        if not self.params.dataset_name:
            self.job.set_job_completion_status("failed", "Dataset name not provided")
            self.add_job_data("end_time", time.strftime("%Y-%m-%d %H:%M:%S"))
            raise ValueError("Dataset name not provided")

        try:
            # Get the dataset path/ID
            dataset_target = get_dataset_path(self.params.dataset_name)
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
