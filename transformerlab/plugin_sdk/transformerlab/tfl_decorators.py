import argparse
import functools
import traceback
import inspect
import sys
from typing import List, Optional, Union, Callable, Dict, Any, Type
from datasets import load_dataset

from transformerlab.plugin import Job, get_dataset_path


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

    def create_progress_callback(self, framework="huggingface", **kwargs):
        """
        Create a progress callback for various ML frameworks.

        Args:
            framework: The framework to create a callback for (e.g., "huggingface", "pytorch_lightning")
            **kwargs: Additional arguments specific to the callback

        Returns:
            A callback object compatible with the specified framework
        """
        self._ensure_args_parsed()

        if framework.lower() == "huggingface":
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

        # elif framework.lower() == "pytorch_lightning":
        #     try:
        #         import pytorch_lightning as pl

        #         class TFLLightningCallback(pl.Callback):
        #             """Callback that updates progress in TransformerLab DB during PyTorch Lightning training"""

        #             def __init__(self, tfl_instance):
        #                 self.tfl = tfl_instance

        #             def on_batch_end(self, trainer, pl_module):
        #                 current = trainer.global_step
        #                 total = trainer.estimated_stepping_batches
        #                 if total > 0:
        #                     progress = int((current / total) * 100)
        #                     self.tfl.progress_update(progress)

        #                     # Check if job should be stopped
        #                     if self.tfl.job.should_stop:
        #                         trainer.should_stop = True

        #         return TFLLightningCallback(self)

        #     except ImportError:
        #         raise ImportError("Could not create PyTorch Lightning callback. Please install pytorch_lightning package.")

        else:
            raise ValueError(
                f"Unsupported framework: {framework}. Supported frameworks: huggingface, pytorch_lightning, keras"
            )


# Create a singleton instance
tfl = TFLPlugin()
