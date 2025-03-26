import argparse
import asyncio
import functools
import os
import time
import traceback
import requests
import json
from pydantic import BaseModel
from typing import Any, List

from datasets import get_dataset_split_names, load_dataset

from transformerlab.plugin import Job, get_dataset_path
import transformerlab.plugin


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
                    elif self.tlab_plugin_type == "evals":
                        self.setup_eval_logging(wandb_project_name=wandb_project_name, manual_logging=manual_logging)

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
                        elif self.tlab_plugin_type == "evals":
                            self.setup_eval_logging(
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

    def get_experiment_config(self, experiment_name: str):
        """Get experiment configuration"""
        return transformerlab.plugin.get_experiment_config(experiment_name)

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
            if "train" in dataset_types:
                print(f"Loaded train dataset with {len(datasets['train'])} examples.")

            if "valid" in dataset_types:
                print(f"Loaded valid dataset with {len(datasets['valid'])} examples.")

            if "test" in dataset_types:
                print(f"Loaded test dataset with {len(datasets['test'])} examples.")

            return datasets

        except Exception as e:
            error_msg = f"Error loading dataset: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            self.job.set_job_completion_status("failed", "Failed to load dataset")
            self.add_job_data("end_time", time.strftime("%Y-%m-%d %H:%M:%S"))
            raise

    def load_evaluation_model(self, field_name="generation_model", model_type=None, model_name=None):
        """
        Load an appropriate model for evaluation based on configuration

        Args:
            field_name: Field name for the generation model
            model_type: Model type ('local', 'openai', 'claude', 'custom') or None to auto-detect
            model_name: Model name to use (defaults to self.model_name)

        Returns:
            A model object wrapped for evaluation use
        """
        from langchain_openai import ChatOpenAI  # noqa

        # Use provided values or class attributes
        model_name = model_name or self.params.model_name
        generation_model = self.params.get(field_name, "local")

        # Auto-detect model type if not provided
        if not model_type:
            gen_model = generation_model.lower()
            if "local" in gen_model:
                model_type = "local"
            elif "azure" in gen_model:
                model_type = "azure"
            elif "openai" in gen_model or "gpt" in gen_model:
                model_type = "openai"
            elif "claude" in gen_model or "anthropic" in gen_model:
                model_type = "claude"
            elif "custom" in gen_model:
                model_type = "custom"
            else:
                model_type = "local"  # Default

        # Load the appropriate model
        if model_type == "local":
            self.check_local_server()
            custom_model = ChatOpenAI(
                api_key="dummy",
                base_url="http://localhost:8338/v1",
                model=model_name,
            )
            return self._create_local_model_wrapper(custom_model)

        elif model_type == "claude":
            anthropic_api_key = transformerlab.plugin.get_db_config_value("ANTHROPIC_API_KEY")
            if not anthropic_api_key or anthropic_api_key.strip() == "":
                raise ValueError("Please set the Anthropic API Key from Settings.")

            os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key
            return self._create_commercial_model_wrapper("claude", generation_model)

        elif model_type == "azure":
            azure_api_details = transformerlab.plugin.get_db_config_value("AZURE_OPENAI_DETAILS")
            if not azure_api_details or azure_api_details.strip() == "":
                raise ValueError("Please set the Azure OpenAI Details from Settings.")

            return self._create_commercial_model_wrapper("azure", "")

        elif model_type == "openai":
            openai_api_key = transformerlab.plugin.get_db_config_value("OPENAI_API_KEY")
            if not openai_api_key or openai_api_key.strip() == "":
                raise ValueError("Please set the OpenAI API Key from Settings.")

            os.environ["OPENAI_API_KEY"] = openai_api_key
            return self._create_commercial_model_wrapper("openai", generation_model)

        elif model_type == "custom":
            custom_api_details = transformerlab.plugin.get_db_config_value("CUSTOM_MODEL_API_KEY")
            if not custom_api_details or custom_api_details.strip() == "":
                raise ValueError("Please set the Custom API Details from Settings.")

            return self._create_commercial_model_wrapper("custom", "")

        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def check_local_server(self):
        """Check if the local model server is running"""
        response = requests.get("http://localhost:8338/server/worker_healthz")
        if response.status_code != 200 or not isinstance(response.json(), list) or len(response.json()) == 0:
            raise RuntimeError("Local Model Server is not running. Please start it before running the evaluation.")

    def _create_local_model_wrapper(self, model):
        """Create a wrapper for local models"""
        # Import here to avoid circular imports
        from deepeval.models.base_model import DeepEvalBaseLLM  # noqa
        from langchain.schema import HumanMessage, SystemMessage  # noqa

        plugin_model_name = self.params.model_name

        class TRLAB_MODEL(DeepEvalBaseLLM):
            def __init__(self, model):
                self.model = model
                self.chat_completions_url = "http://localhost:8338/v1/chat/completions"
                self.generation_model_name = plugin_model_name
                self.api_key = "dummy"

            def load_model(self):
                return self.model

            def generate(self, prompt: str) -> str:
                chat_model = self.load_model()
                return chat_model.invoke(prompt).content

            async def a_generate(self, prompt: str) -> str:
                chat_model = self.load_model()
                res = await chat_model.ainvoke(prompt)
                return res.content

            def generate_without_instructor(self, messages: List[dict]) -> BaseModel:
                chat_model = self.load_model()
                modified_messages = []
                for message in messages:
                    if message["role"] == "system":
                        modified_messages.append(SystemMessage(**message))
                    else:
                        modified_messages.append(HumanMessage(**message))
                return chat_model.invoke(modified_messages).content

            def get_model_name(self):
                return self.model

        return TRLAB_MODEL(model)

    def _create_commercial_model_wrapper(self, model_type, model_name):
        """Create a wrapper for commercial models"""
        from anthropic import Anthropic
        from deepeval.models.base_model import DeepEvalBaseLLM
        from openai import OpenAI, AzureOpenAI

        class CustomCommercialModel(DeepEvalBaseLLM):
            def __init__(self, model_type="claude", model_name="claude-3-7-sonnet-latest"):
                self.model_type = model_type
                self.generation_model_name = model_name

                if model_type == "claude":
                    self.chat_completions_url = "https://api.anthropic.com/v1/chat/completions"
                    anthropic_api_key = transformerlab.plugin.get_db_config_value("ANTHROPIC_API_KEY")
                    self.api_key = anthropic_api_key
                    if not anthropic_api_key or anthropic_api_key.strip() == "":
                        raise ValueError("Please set the Anthropic API Key from Settings.")
                    else:
                        os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key
                    self.model = Anthropic()
                elif model_type == "azure":
                    azure_api_details = transformerlab.plugin.get_db_config_value("AZURE_OPENAI_DETAILS")
                    if not azure_api_details or azure_api_details.strip() == "":
                        raise ValueError("Please set the Azure OpenAI Details from Settings.")
                    azure_api_details = json.loads(azure_api_details)

                    self.model = AzureOpenAI(
                        api_key=azure_api_details["azure_openai_api_key"],
                        api_version=azure_api_details["openai_api_version"],
                        azure_endpoint=azure_api_details["azure_endpoint"],
                    )
                    self.generation_model_name = azure_api_details["azure_deployment"]
                    self.model_name = azure_api_details["azure_deployment"]

                    self.chat_completions_url = f"{azure_api_details['azure_endpoint']}/openai/deployments/{azure_api_details['azure_deployment']}/chat/completions?api-version={azure_api_details['openai_api_version']}"
                    self.api_key = azure_api_details["azure_openai_api_key"]

                elif model_type == "openai":
                    self.chat_completions_url = "https://api.openai.com/v1/chat/completions"
                    openai_api_key = transformerlab.plugin.get_db_config_value("OPENAI_API_KEY")
                    self.api_key = openai_api_key
                    if not openai_api_key or openai_api_key.strip() == "":
                        raise ValueError("Please set the OpenAI API Key from Settings.")
                    else:
                        os.environ["OPENAI_API_KEY"] = openai_api_key
                    self.model = OpenAI()

                elif model_type == "custom":
                    custom_api_details = transformerlab.plugin.get_db_config_value("CUSTOM_MODEL_API_KEY")

                    if not custom_api_details or custom_api_details.strip() == "":
                        raise ValueError("Please set the Custom API Details from Settings.")
                    else:
                        custom_api_details = json.loads(custom_api_details)
                        self.model = OpenAI(
                            api_key=custom_api_details["customApiKey"],
                            base_url=custom_api_details["customBaseURL"],
                        )
                        self.chat_completions_url = f"{custom_api_details['customBaseURL']}/chat/completions"
                        self.api_key = custom_api_details["customApiKey"]
                        self.generation_model_name = custom_api_details["customModelName"]
                        self.model_name = custom_api_details["customModelName"]

            def load_model(self):
                return self.model

            def generate(self, prompt: str, schema=None):
                client = self.load_model()
                if schema:
                    import instructor

                    if self.model_type == "claude":
                        instructor_client = instructor.from_anthropic(client)
                    else:
                        instructor_client = instructor.from_openai(client)

                    resp = instructor_client.messages.create(
                        model=self.generation_model_name,
                        max_tokens=1024,
                        messages=[{"role": "user", "content": prompt}],
                        response_model=schema,
                    )
                    return resp
                else:
                    response = client.chat.completions.create(
                        model=self.generation_model_name,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    return response.choices[0].message.content

            async def a_generate(self, prompt: str, schema=None):
                return self.generate(prompt, schema)

            def generate_without_instructor(self, messages: List[dict]):
                client = self.load_model()
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                )
                return response.choices[0].message.content

            def get_model_name(self):
                return self.generation_model_name

        return CustomCommercialModel(model_type, model_name)
