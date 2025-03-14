import os
import time
import argparse
import functools
import asyncio
import traceback
from typing import List, Any
import json
from tensorboardX import SummaryWriter
from datasets import load_dataset, get_dataset_split_names
import requests

import transformerlab.plugin
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
                self.add_job_data("model_name", self.model_name)
                if hasattr(self, "template_name"):
                    self.add_job_data("template_name", self.template_name)
                elif hasattr(self, "run_name"):
                    self.add_job_data("template_name", self.run_name)

                # Update starting progress
                self.job.update_progress(progress_start)

                try:
                    # Call the wrapped function
                    result = func(*args, **kwargs)

                    # Update final progress and success status
                    self.job.update_progress(progress_end)
                    self.job.set_job_completion_status("success", "Job completed successfully")
                    self.add_job_data("end_time", time.strftime("%Y-%m-%d %H:%M:%S"))

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

    def async_job_wrapper(self, progress_start: int = 0, progress_end: int = 100):
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
                self.add_job_data("model_name", self.model_name)
                self.add_job_data("template_name", self.template_name)

                # Update starting progress
                self.job.update_progress(progress_start)

                async def run_async():
                    try:
                        # Call the wrapped async function
                        result = await func(*args, **kwargs)

                        # Update final progress and success status
                        self.job.update_progress(progress_end)
                        self.job.set_job_completion_status("success", "Job completed successfully")
                        self.add_job_data("end_time", time.strftime("%Y-%m-%d %H:%M:%S"))

                        return result

                    except Exception as e:
                        # Capture the full error
                        error_msg = f"Error in Async Job: {str(e)}\n{traceback.format_exc()}"
                        print(error_msg)

                        # Update job with failure status
                        self.job.set_job_completion_status("failed", "Error occurred while executing job")
                        self.add_job_data("end_time", time.strftime("%Y-%m-%d %H:%M:%S"))

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
        import os  # noqa
        from langchain_openai import ChatOpenAI  # noqa

        # Use provided values or class attributes
        model_name = model_name or self.model_name
        generation_model = getattr(self, field_name, "local")

        # Auto-detect model type if not provided
        if not model_type:
            gen_model = generation_model.lower()
            if "local" in gen_model:
                model_type = "local"
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
        from pydantic import BaseModel  # noqa

        plugin_model_name = self.model_name

        class TRLAB_MODEL(DeepEvalBaseLLM):
            def __init__(self, model):
                self.model = model
                self.chat_completions_url = "http://localhost:8338/v1/chat/completions"
                self.generation_model_name = plugin_model_name

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
        from deepeval.models.base_model import DeepEvalBaseLLM
        from openai import OpenAI
        from anthropic import Anthropic

        class CustomCommercialModel(DeepEvalBaseLLM):
            def __init__(self, model_type="claude", model_name="claude-3-7-sonnet-latest"):
                self.model_type = model_type
                self.generation_model_name = model_name

                if model_type == "claude":
                    self.chat_completions_url = "https://api.anthropic.com/v1/chat/completions"
                    anthropic_api_key = transformerlab.plugin.get_db_config_value("ANTHROPIC_API_KEY")
                    if not anthropic_api_key or anthropic_api_key.strip() == "":
                        raise ValueError("Please set the Anthropic API Key from Settings.")
                    else:
                        os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key
                    self.model = Anthropic()

                elif model_type == "openai":
                    self.chat_completions_url = "https://api.openai.com/v1/chat/completions"
                    openai_api_key = transformerlab.plugin.get_db_config_value("OPENAI_API_KEY")
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
                        self.generation_model_name = custom_api_details["customModelName"]

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
        self._parser.add_argument("--template_name", default="evaluation", type=str, help="Name for the evaluation run")
        self._parser.add_argument("--experiment_name", default="", type=str, help="Name of the experiment")
        self._parser.add_argument("--eval_name", default="", type=str, help="Name of the evaluation")

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

        self.add_job_data("score", json.dumps(score_list))

        return output_path, plot_data_path


class GenTFLPlugin(TFLPlugin):
    """Enhanced decorator class for TransformerLab generation plugins"""

    def __init__(self):
        super().__init__()
        # Add common generation-specific arguments
        self._parser.add_argument("--run_name", default="generated", type=str, help="Name for the generated dataset")
        self._parser.add_argument("--experiment_name", default="default", type=str, help="Name of the experiment")
        self._parser.add_argument("--model_adapter", default=None, type=str, help="Model adapter to use")
        self._parser.add_argument("--generation_model", default="local", type=str, help="Model to use for generation")
        self._parser.add_argument("--generation_type", default="local", type=str, help="Model to use for generation")

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

    def save_generated_dataset(self, df, additional_metadata=None, dataset_id=None):
        """Save generated data to file and create dataset in TransformerLab

        Args:
            df: DataFrame containing generated data
            additional_metadata: Optional dict with additional metadata to save

        Returns:
            tuple: Paths to the saved files (output_file, dataset_name)
        """
        self._ensure_args_parsed()

        # Create output directory
        output_dir = self.get_output_file_path(dir_only=True)
        os.makedirs(output_dir, exist_ok=True)

        # Save to file
        output_file = os.path.join(output_dir, f"{self.run_name}_{self.job_id}.json")
        df.to_json(output_file, orient="records", lines=False)
        print(f"Generated data saved to {output_file}")

        # Store metadata
        metadata = {
            "generation_model": self.generation_model,
            "generation_type": getattr(self, "generation_type", "scratch"),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "sample_count": len(df),
        }

        if additional_metadata:
            metadata.update(additional_metadata)

        # Save metadata
        metadata_file = os.path.join(output_dir, f"{self.run_name}_{self.job_id}_metadata.json")
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        # Upload to TransformerLab as a dataset
        try:
            self.upload_to_transformerlab(output_file, dataset_id)
            if dataset_id:
                self.add_job_data("dataset_id", dataset_id)
            else:
                self.add_job_data("dataset_name", self.run_name)
        except Exception as e:
            print(f"Error uploading to TransformerLab: {e}")
            self.add_job_data("upload_error", str(e))

        return output_file, self.run_name

    def upload_to_transformerlab(self, output_file_path, dataset_id=None):
        """Create a dataset in TransformerLab from the generated file

        Args:
            output_file_path: Path to the generated file

        Returns:
            bool: Whether upload was successful
        """
        try:
            api_url = "http://localhost:8338/"

            # Create a new dataset
            if not dataset_id:
                params = {"dataset_id": self.run_name, "generated": True}
            else:
                params = {"dataset_id": dataset_id, "generated": True}

            response = requests.get(api_url + "data/new", params=params)
            if response.status_code != 200:
                raise RuntimeError(f"Error creating a new dataset: {response.json()}")

            # Upload the file
            with open(output_file_path, "rb") as json_file:
                files = {"files": json_file}
                response = requests.post(api_url + "data/fileupload", params=params, files=files)

            if response.status_code != 200:
                raise RuntimeError(f"Error uploading the dataset: {response.json()}")

            print(f"Dataset '{self.run_name}' uploaded successfully to TransformerLab")
            return True

        except Exception as e:
            print(f"Error uploading to TransformerLab: {e}")
            raise

    def get_output_file_path(self, suffix="", dir_only=False):
        """Get path for saving generated outputs

        Args:
            suffix: Optional suffix for the filename
            dir_only: Whether to return just the directory

        Returns:
            str: Full path for output file or directory
        """
        self._ensure_args_parsed()

        workspace_dir = os.environ.get("_TFL_WORKSPACE_DIR", "./")
        experiment_dir = os.path.join(workspace_dir, "experiments", self.experiment_name)
        dataset_dir = os.path.join(experiment_dir, "datasets")

        # Create a specific directory for this generation job
        gen_dir = os.path.join(dataset_dir, f"{self.run_name}_{self.job_id}")
        os.makedirs(gen_dir, exist_ok=True)

        if dir_only:
            return gen_dir

        if suffix:
            return os.path.join(gen_dir, f"{self.run_name}_{suffix}")
        else:
            return os.path.join(gen_dir, f"{self.run_name}.json")

    def generate_expected_outputs(self, input_values, task=None, scenario=None, input_format=None, output_format=None):
        """Generate expected outputs for given inputs using loaded model

        Args:
            input_values: List of input values
            task: Optional task description
            scenario: Optional scenario description
            input_format: Optional input format description
            output_format: Optional output format description

        Returns:
            list: Generated expected outputs
        """
        # Use provided values or class attributes if available
        task = task or getattr(self, "task", "")
        scenario = scenario or getattr(self, "scenario", "")
        input_format = input_format or getattr(self, "input_format", "")
        output_format = output_format or getattr(self, "expected_output_format", "")

        # Load model if not already available as instance attribute
        if not hasattr(self, "generation_model_instance"):
            self.generation_model_instance = self.load_evaluation_model(field_name="generation_model")

        model = self.generation_model_instance

        # Generate outputs
        expected_outputs = []
        for i, input_val in enumerate(input_values):
            prompt = f"""Given a task, scenario and expected input as well as output formats, generate the output for a given input.
                    \n\nTask: {task}
                    \n\nScenario: {scenario}
                    \n\nExpected Output Format: {output_format}
                    \n\nExpected Input Format: {input_format}
                    \n\n Generate the output for the following input: {input_val}.
                    \n Output: """

            messages = [{"role": "system", "content": prompt}]

            # Try to use generate_without_instructor if available
            try:
                expected_output = model.generate_without_instructor(messages)
            except AttributeError:
                # Fall back to normal generate
                expected_output = model.generate(prompt)

            expected_outputs.append(expected_output)

            # Update progress for long generations
            if i % 5 == 0 and len(input_values) > 10:
                progress = int((i / len(input_values)) * 100)
                self.progress_update(progress)

        return expected_outputs


# Create singleton instances
tfl = TFLPlugin()
tfl_trainer = TrainerTFLPlugin()
tfl_evals = EvalsTFLPlugin()
tfl_gen = GenTFLPlugin()
