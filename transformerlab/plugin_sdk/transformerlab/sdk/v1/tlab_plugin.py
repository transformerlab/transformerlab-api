import argparse
import asyncio
import functools
import os
import time
import traceback
import requests
import json
from pydantic import BaseModel
from typing import Any, List, Type

from datasets import get_dataset_split_names, get_dataset_config_names, load_dataset

try:
    from transformerlab.plugin import Job, get_dataset_path
    import transformerlab.plugin as tlab_core
except ModuleNotFoundError:
    from transformerlab.plugin_sdk.transformerlab.plugin import Job, get_dataset_path
    import transformerlab.plugin_sdk.transformerlab.plugin as tlab_core


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
                start_time = time.strftime("%Y-%m-%d %H:%M:%S")
                self.add_job_data("start_time", start_time)
                self.params.start_time = start_time
                self.add_job_data("model_name", self.params.model_name)
                self.add_job_data("template_name", self.params.template_name)
                self.add_job_data("model_adapter", self.params.get("model_adapter", ""))

                # Update starting progress
                self.progress_update(progress_start)

                try:
                    # Setup logging
                    if self.tlab_plugin_type == "trainer":
                        self.setup_train_logging(wandb_project_name=wandb_project_name, manual_logging=manual_logging)
                    elif self.tlab_plugin_type == "evals":
                        self.setup_eval_logging(wandb_project_name=wandb_project_name, manual_logging=manual_logging)

                    # Call the wrapped function
                    result = func(*args, **kwargs)

                    # Update final progress and success status
                    self.progress_update(progress_end)
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
                self.progress_update(progress_start)

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
                        self.progress_update(progress_end)
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
        job_data = self.job.get_job_data()
        if job_data.get("sweep_progress") is not None:
            if int(job_data.get("sweep_progress")) != 100:
                self.job.update_job_data("sweep_subprogress", progress)
                return

        self.job.update_progress(progress)
        if self.job.should_stop:
            self.job.update_status("STOPPED")
            raise KeyboardInterrupt("Job stopped by user")

    def get_experiment_config(self, experiment_name: str):
        """Get experiment configuration"""
        return tlab_core.get_experiment_config(experiment_name)

    def add_job_data(self, key: str, value: Any):
        """Add data to job"""
        self.job.add_to_job_data(key, value)

    def load_dataset(self, dataset_types: List[str] = ["train"], config_name: str = None):
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
            # Get the available config names
            available_configs = get_dataset_config_names(dataset_target)

            if available_configs and available_configs[0] == "default":
                available_configs.pop(0)
                config_name = None
                print("Default config found, ignoring config_name")

            if config_name and config_name not in available_configs:
                raise ValueError(f"Config name {config_name} not found in dataset")

            if not config_name and len(available_configs) > 0:
                config_name = available_configs[0]
                print(f"Using default config name: {config_name}")

            # Handle different validation split names
            dataset_splits = {}
            for dataset_type in dataset_types:
                dataset_splits[dataset_type] = dataset_type

            # Check if train split is available and handle it if not available
            if "train" in dataset_types and "train" not in available_splits:
                print(
                    "WARNING: No train split found in dataset, we will use the first available split as train.\n Training a model on non-train splits is not recommended."
                )
                dataset_splits["train"] = available_splits[0]
                print(f"Using `{dataset_splits['train']}` for the training split.")

            if "validation" in available_splits and "valid" in dataset_splits:
                dataset_splits["valid"] = "validation"
            elif "valid" in dataset_types and "valid" not in available_splits:
                print("No validation slice found in dataset, using train split as 80-20 for training and validation")
                dataset_splits["valid"] = dataset_splits["train"] + "[-20%:]"
                dataset_splits["train"] = dataset_splits["train"] + "[:80%]"

            # If dataset_splits for train is same as any other split, make it a 80:20 thing to not have same data in train and test/valid
            for expected_split, actual_split in dataset_splits.items():
                if expected_split != "train" and actual_split == dataset_splits["train"]:
                    dataset_splits[expected_split] = dataset_splits["train"] + "[-20%:]"
                    dataset_splits["train"] = dataset_splits["train"] + "[:80%]"
                    print(
                        f"Using `{dataset_splits[expected_split]}` for the {expected_split} split as its same as train split."
                    )

            # Load each dataset split
            datasets = {}
            for dataset_type in dataset_splits:
                datasets[dataset_type] = load_dataset(
                    dataset_target, data_dir=config_name, split=dataset_splits[dataset_type], trust_remote_code=True
                )
            if "train" in dataset_types:
                print(f"Loaded train dataset with {len(datasets['train'])} examples.")
            else:
                print("WARNING: No train dataset loaded, ensure you have a train split in your dataset.")

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
            anthropic_api_key = tlab_core.get_db_config_value("ANTHROPIC_API_KEY")
            if not anthropic_api_key or anthropic_api_key.strip() == "":
                raise ValueError("Please set the Anthropic API Key from Settings.")

            os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key
            return self._create_commercial_model_wrapper("claude", generation_model)

        elif model_type == "azure":
            azure_api_details = tlab_core.get_db_config_value("AZURE_OPENAI_DETAILS")
            if not azure_api_details or azure_api_details.strip() == "":
                raise ValueError("Please set the Azure OpenAI Details from Settings.")

            return self._create_commercial_model_wrapper("azure", "")

        elif model_type == "openai":
            openai_api_key = tlab_core.get_db_config_value("OPENAI_API_KEY")
            if not openai_api_key or openai_api_key.strip() == "":
                raise ValueError("Please set the OpenAI API Key from Settings.")

            os.environ["OPENAI_API_KEY"] = openai_api_key
            return self._create_commercial_model_wrapper("openai", generation_model)

        elif model_type == "custom":
            custom_api_details = tlab_core.get_db_config_value("CUSTOM_MODEL_API_KEY")
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
        import json
        import re

        plugin_model_name = self.params.model_name

        class TRLAB_MODEL(DeepEvalBaseLLM):
            def __init__(self, model):
                self.model = model
                self.chat_completions_url = "http://localhost:8338/v1/chat/completions"
                self.generation_model_name = plugin_model_name
                self.api_key = "dummy"

            def load_model(self):
                return self.model

            def _repair_json_string(self, json_str):
                """Apply basic JSON repairs"""
                original = json_str
                repairs_made = []

                # First, clean up obviously wrong escaped quotes in comma-separated lists
                before = json_str
                # Fix patterns like: "word","\"other\",\"word\"," -> "word","other","word",
                json_str = re.sub(r'",\\"([^"]+)\\",', r'","\1",', json_str)
                json_str = re.sub(r'\\",\\"', r'","', json_str)
                if json_str != before:
                    repairs_made.append("cleaned up malformed escaped quotes")

                # Fix unescaped quotes inside string values - precise targeting
                before = json_str
                # Only target quotes that are clearly inside string values
                # Look for: space/letter + " + content + " + period/letter (but not JSON structure)
                # This specifically targets: called "Generate From Raw Text Plugin".
                json_str = re.sub(r'(\w)\s+"([^"]+)"\s*([.\w])', r'\1 \\"\2\\" \3', json_str)
                if json_str != before:
                    repairs_made.append("escaped unescaped quotes in strings")

                # Remove trailing commas
                before = json_str
                json_str = re.sub(r",\s*([}\]])", r"\1", json_str)
                if json_str != before:
                    repairs_made.append("removed trailing commas")

                # Fix missing commas between object properties
                before = json_str
                json_str = re.sub(r'(["\d\]\}])\s*("[\w\s]+"\s*:)', r"\1, \2", json_str)
                if json_str != before:
                    repairs_made.append("added missing commas between properties")

                # Fix missing commas between array elements
                before = json_str
                json_str = re.sub(r'(["\d\]\}])\s*(["\[\{])', r"\1, \2", json_str)
                if json_str != before:
                    repairs_made.append("added missing commas between array elements")

                # Fix objects with multiple strings (convert to array of strings)
                before = json_str
                # Handle complex cases with many comma-separated strings in braces
                # Pattern: {"string1","string2","string3",...} -> ["string1","string2","string3",...]
                json_str = re.sub(r'\{\s*("(?:[^"\\]|\\.)*"(?:\s*,\s*"(?:[^"\\]|\\.)*")*)\s*\}', r"[\1]", json_str)

                # More aggressive pattern for badly formatted objects with just strings
                # This catches cases where strings might have various quote issues
                if json_str == before:
                    # Look for any objects and check if they're malformed (no colons = just comma-separated strings)
                    pattern = r"\{[^{}]*\}"
                    matches = list(re.finditer(pattern, json_str))

                    malformed_matches = []
                    for match in matches:
                        obj_str = match.group(0)
                        # Check if this object has no colons (indicates comma-separated strings, not key-value pairs)
                        if ":" not in obj_str and "," in obj_str and '"' in obj_str:
                            malformed_matches.append(match)

                    if malformed_matches:
                        # Process matches in reverse order to avoid offset issues
                        for match in reversed(malformed_matches):
                            obj_str = match.group(0)
                            # Extract quoted strings from THIS specific object only
                            strings = re.findall(r'"([^"]*)"', obj_str)
                            if len(strings) > 1:  # Only if multiple strings found
                                # Check if these are complete sentences or just words
                                are_sentences = any(
                                    (s.endswith(".") or s.endswith("!") or s.endswith("?")) and len(s.split()) > 3
                                    for s in strings
                                )

                                # Check for duplicates and clean them up
                                unique_strings = []
                                for s in strings:
                                    # Normalize string for comparison (remove extra whitespace, lowercase)
                                    normalized = " ".join(s.lower().split())
                                    if normalized not in [" ".join(us.lower().split()) for us in unique_strings]:
                                        unique_strings.append(s)

                                if are_sentences:
                                    # These are complete sentences - convert to array
                                    replacement = "[" + ",".join([f'"{s}"' for s in unique_strings]) + "]"
                                    json_str = json_str[: match.start()] + replacement + json_str[match.end() :]
                                else:
                                    # These are words - join into one sentence
                                    joined_text = " ".join(unique_strings).strip()
                                    # Clean up any double spaces or odd punctuation
                                    joined_text = re.sub(r"\s+", " ", joined_text)
                                    joined_text = re.sub(r"\s+([.,!?;:])", r"\1", joined_text)
                                    # Create single string replacement
                                    replacement = f'"{joined_text}"'
                                    json_str = json_str[: match.start()] + replacement + json_str[match.end() :]

                if json_str != before:
                    repairs_made.append("converted multi-string objects to arrays")

                # Fix objects with just string values (no property names)
                before = json_str
                json_str = re.sub(r'\{\s*"([^"]+)"\s*\}', r'{"value": "\1"}', json_str)
                if json_str != before:
                    repairs_made.append("added missing property names")

                # Fix missing colons after property names
                before = json_str
                json_str = re.sub(r'("[\w\s]+")(\s+)(["\d\[\{])', r"\1:\2\3", json_str)
                json_str = re.sub(r'("[\w\s]+")(["\d\[\{])', r"\1: \2", json_str)
                if json_str != before:
                    repairs_made.append("added missing colons")

                # Fix unmatched braces
                open_braces = json_str.count("{")
                close_braces = json_str.count("}")
                if open_braces > close_braces:
                    missing = open_braces - close_braces
                    json_str += "}" * missing
                    repairs_made.append(f"added {missing} closing braces")

                # Fix unmatched brackets
                open_brackets = json_str.count("[")
                close_brackets = json_str.count("]")
                if open_brackets > close_brackets:
                    missing = open_brackets - close_brackets
                    json_str += "]" * missing
                    repairs_made.append(f"added {missing} closing brackets")

                # Remove any trailing model tokens
                before = json_str
                json_str = re.sub(r"<\|im_end\|>.*?$", "", json_str, flags=re.DOTALL)
                if json_str != before:
                    repairs_made.append("removed trailing model tokens")

                return json_str

            def _enhance_json_prompt(self, prompt):
                """Enhance prompts to improve JSON generation"""
                if "json" in prompt.lower() or "{" in prompt or "score" in prompt.lower():
                    print("📝 PROMPT ENHANCEMENT: Adding JSON instructions to prompt")
                    # Add JSON-specific instructions
                    json_instructions = """

CRITICAL: You must respond with valid JSON only. Follow these rules:
1. Output ONLY valid JSON - no explanations, no markdown, no extra text
2. Use double quotes for all strings
3. No trailing commas
4. Ensure all braces and brackets are properly closed
5. Numbers should not be quoted unless they're strings
6. Use null for empty values, not undefined

Example format: {"score": 0.8, "reason": "explanation here"}

Your response:"""
                    return prompt + json_instructions
                return prompt

            def _safe_schema_convert(self, data, schema, context):
                """Simple, robust schema conversion with Pydantic v2 compatibility"""
                print(f"🛡️ {context}: Converting to schema object")

                try:
                    # Try direct conversion first
                    result = schema.parse_obj(data)
                    print(f"✅ {context}: Direct schema conversion successful")
                    return result
                except Exception as e:
                    print(f"❌ {context}: Direct conversion failed: {e}")

                try:
                    # If data is a list and schema expects object with list field, try to fix it
                    if isinstance(data, list):
                        # Get schema fields using Pydantic v2 compatible method
                        if hasattr(schema, "__fields__"):
                            fields = schema.__fields__
                        elif hasattr(schema, "model_fields"):
                            fields = schema.model_fields
                        else:
                            fields = {}

                        for field_name, field_info in fields.items():
                            # Try to detect list fields (compatible with both Pydantic v1 and v2)
                            field_type = getattr(field_info, "annotation", getattr(field_info, "type_", None))
                            if field_type and hasattr(field_type, "__origin__") and field_type.__origin__ is list:
                                fixed_data = {field_name: data}
                                result = schema.parse_obj(fixed_data)
                                print(f"✅ {context}: List-to-object conversion successful")
                                return result
                except Exception as e:
                    print(f"❌ {context}: List conversion failed: {e}")

                try:
                    # Create minimal valid object - try different field access methods
                    if hasattr(schema, "__fields__"):
                        fields = schema.__fields__
                    elif hasattr(schema, "model_fields"):
                        fields = schema.model_fields
                    else:
                        fields = {}

                    minimal_data = {}
                    for field_name, field_info in fields.items():
                        # Get field type (compatible with both Pydantic v1 and v2)
                        field_type = getattr(field_info, "annotation", getattr(field_info, "type_", None))

                        if field_type and hasattr(field_type, "__origin__") and field_type.__origin__ is list:
                            minimal_data[field_name] = []
                        else:
                            minimal_data[field_name] = ""

                    result = schema.parse_obj(minimal_data)
                    print(f"⚠️ {context}: Created minimal valid object")
                    return result
                except Exception as e:
                    print(f"❌ {context}: Minimal object creation failed: {e}")

                try:
                    # Last resort: try default constructor
                    result = schema()
                    print(f"⚠️ {context}: Used default schema constructor")
                    return result
                except Exception as e:
                    print(f"💥 {context}: All schema creation failed: {e}")

                    # Emergency fallback - create a simple object with required attributes
                    class EmergencyFallback:
                        pass

                    fallback = EmergencyFallback()
                    # Try to add common DeepEval attributes
                    for attr in ["statements", "verdicts", "reason", "score"]:
                        if attr == "statements" or attr == "verdicts":
                            setattr(fallback, attr, [])
                        else:
                            setattr(fallback, attr, "")

                    print(f"🆘 {context}: Created emergency fallback with common attributes")
                    return fallback

            def _extract_and_repair_json(self, response_text):
                """Extract and attempt to repair JSON from model response"""
                try:
                    # First, try to parse as-is
                    result = json.loads(response_text)
                    return result
                except json.JSONDecodeError:
                    pass

                # Clean up model tokens and extra content first
                clean_text = response_text.strip()

                # Remove model tokens - be more aggressive about all variants
                clean_text = re.sub(r"<\|im_end\|>.*?$", "", clean_text, flags=re.DOTALL)
                clean_text = re.sub(r"<\|im_start\|>.*?<\|im_end\|>", "", clean_text, flags=re.DOTALL)
                clean_text = re.sub(r"```json\s*", "", clean_text)
                clean_text = re.sub(r"```\s*", "", clean_text)
                clean_text = clean_text.strip()

                # Extract JSON
                json_match = re.search(r"\{.*\}", clean_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0).strip()
                    try:
                        result = json.loads(json_str)
                        return result
                    except json.JSONDecodeError:
                        # Try basic repairs
                        json_str = self._repair_json_string(json_str)
                        try:
                            result = json.loads(json_str)
                            return result
                        except json.JSONDecodeError:
                            pass

                # All JSON repair steps failed - return error to let system crash
                return {"error": "Failed to generate valid JSON", "original_response": response_text}

            def generate(self, prompt: str, schema: Type[BaseModel] | None = None, **kwargs):
                enhanced_prompt = self._enhance_json_prompt(prompt)
                chat_model = self.load_model()
                response = chat_model.invoke(enhanced_prompt).content

                # If the response looks like it should be JSON, try to fix it
                if any(indicator in prompt.lower() for indicator in ["json", "score", "evaluation", "rating"]):
                    try:
                        parsed_json = self._extract_and_repair_json(response)
                        if isinstance(parsed_json, dict):
                            # Check if this is a valid empty response
                            if "statements" in parsed_json and isinstance(parsed_json["statements"], list):
                                if schema is not None:
                                    return self._safe_schema_convert(parsed_json, schema, "GENERATE")
                                return json.dumps(parsed_json)
                            # Check if this is an error response
                            elif "error" in parsed_json and "original_response" in parsed_json:
                                raise ValueError(
                                    f"JSON repair failed: {parsed_json.get('original_response', response)}"
                                )
                            else:
                                if schema is not None:
                                    return self._safe_schema_convert(parsed_json, schema, "GENERATE")
                                return json.dumps(parsed_json)
                        else:
                            raise ValueError(f"JSON repair failed: Invalid response type: {type(parsed_json)}")

                    except Exception as e:
                        raise e

                return response

            async def a_generate(self, prompt: str, schema: Type[BaseModel] | None = None, **kwargs):
                enhanced_prompt = self._enhance_json_prompt(prompt)
                chat_model = self.load_model()
                res = await chat_model.ainvoke(enhanced_prompt)
                response = res.content

                # If the response looks like it should be JSON, try to fix it
                if any(indicator in prompt.lower() for indicator in ["json", "score", "evaluation", "rating"]):
                    try:
                        parsed_json = self._extract_and_repair_json(response)
                        if isinstance(parsed_json, dict) and "error" not in parsed_json:
                            if schema is not None:
                                return self._safe_schema_convert(parsed_json, schema, "A_GENERATE")
                            return json.dumps(parsed_json)
                        else:
                            raise ValueError(f"JSON repair failed: {parsed_json.get('original_response', response)}")

                    except Exception as e:
                        raise e

                return response

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
                return self.generation_model_name

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
                    self.base_url = "https://api.anthropic.com/v1"
                    anthropic_api_key = tlab_core.get_db_config_value("ANTHROPIC_API_KEY")
                    self.api_key = anthropic_api_key
                    if not anthropic_api_key or anthropic_api_key.strip() == "":
                        raise ValueError("Please set the Anthropic API Key from Settings.")
                    else:
                        os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key
                    self.model = Anthropic()
                elif model_type == "azure":
                    azure_api_details = tlab_core.get_db_config_value("AZURE_OPENAI_DETAILS")
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
                    self.base_url = "https://api.openai.com/v1"
                    openai_api_key = tlab_core.get_db_config_value("OPENAI_API_KEY")
                    self.api_key = openai_api_key
                    if not openai_api_key or openai_api_key.strip() == "":
                        raise ValueError("Please set the OpenAI API Key from Settings.")
                    else:
                        os.environ["OPENAI_API_KEY"] = openai_api_key
                    self.model = OpenAI()

                elif model_type == "custom":
                    custom_api_details = tlab_core.get_db_config_value("CUSTOM_MODEL_API_KEY")

                    if not custom_api_details or custom_api_details.strip() == "":
                        raise ValueError("Please set the Custom API Details from Settings.")
                    else:
                        custom_api_details = json.loads(custom_api_details)
                        self.model = OpenAI(
                            api_key=custom_api_details["customApiKey"],
                            base_url=custom_api_details["customBaseURL"],
                        )
                        self.chat_completions_url = f"{custom_api_details['customBaseURL']}/chat/completions"
                        self.base_url = f"{custom_api_details['customBaseURL']}"
                        self.api_key = custom_api_details["customApiKey"]
                        self.generation_model_name = custom_api_details["customModelName"]
                        self.model_name = custom_api_details["customModelName"]

            def load_model(self):
                return self.model

            def generate(self, prompt: str):
                client = self.load_model()
                    response = client.chat.completions.create(
                        model=self.generation_model_name,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    return response.choices[0].message.content

            async def a_generate(self, prompt: str):
                return self.generate(prompt)

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
