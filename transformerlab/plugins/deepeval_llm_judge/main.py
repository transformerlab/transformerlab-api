import importlib
import json
import os

import instructor
import pandas as pd
import requests
from anthropic import Anthropic
from deepeval import evaluate
from deepeval.dataset import EvaluationDataset
from deepeval.metrics import GEval
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from langchain_openai import ChatOpenAI
from openai import OpenAI
from pydantic import BaseModel

from transformerlab.tfl_decorators import tfl_evals

# Define metric groups for validation
two_input_metrics = ["AnswerRelevancyMetric", "BiasMetric", "ToxicityMetric"]
three_input_metrics = [
    "FaithfulnessMetric",
    "ContextualPrecisionMetric",
    "ContextualRecallMetric",
    "ContextualRelevancyMetric",
    "HallucinationMetric",
]
custom_metric = ["GEval"]


def get_metric_class(metric_name: str):
    """
    Import the metric class based on the metric name
    :param metric_name: Name of the metric
    :return: Metric class
    """
    module = importlib.import_module("deepeval.metrics")
    try:
        metric_class = getattr(module, metric_name)
        return metric_class
    except AttributeError:
        print(f"Metric {metric_name} not found in deepeval.metrics")
        raise ValueError(f"Metric {metric_name} not found in deepeval.metrics")


def check_local_server():
    response = requests.get("http://localhost:8338/server/worker_healthz")
    if response.status_code != 200 or not isinstance(response.json(), list) or len(response.json()) == 0:
        print("Local Model Server is not running. Please start it before running the evaluation.")
        raise RuntimeError("Local Model Server is not running. Please start it before running the evaluation.")


class TRLAB_MODEL(DeepEvalBaseLLM):
    def __init__(self, model):
        self.model = model

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        return chat_model.invoke(prompt).content

    async def a_generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        res = await chat_model.ainvoke(prompt)
        return res.content

    def get_model_name(self):
        return tfl_evals.model_name


class CustomCommercialModel(DeepEvalBaseLLM):
    def __init__(self, model_type="claude", model_name="claude-3-7-sonnet-latest"):
        self.model_type = model_type
        self.model_name = model_name
        import transformerlab.plugin

        if model_type == "claude":
            anthropic_api_key = transformerlab.plugin.get_db_config_value("ANTHROPIC_API_KEY")
            if not anthropic_api_key or anthropic_api_key.strip() == "":
                print("Please set the Anthropic API Key from Settings.")
                raise ValueError("Please set the Anthropic API Key from Settings.")
            else:
                os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key
            self.model = Anthropic()

        elif model_type == "openai":
            openai_api_key = transformerlab.plugin.get_db_config_value("OPENAI_API_KEY")
            if not openai_api_key or openai_api_key.strip() == "":
                print("Please set the OpenAI API Key from Settings.")
                raise ValueError("Please set the OpenAI API Key from Settings.")
            else:
                os.environ["OPENAI_API_KEY"] = openai_api_key
            self.model = OpenAI()

        elif model_type == "custom":
            custom_api_details = transformerlab.plugin.get_db_config_value("CUSTOM_MODEL_API_KEY")
            if not custom_api_details or custom_api_details.strip() == "":
                print("Please set the Custom API Details from Settings.")
                raise ValueError("Please set the Custom API Details from Settings.")
            else:
                custom_api_details = json.loads(custom_api_details)
                self.model = OpenAI(
                    api_key=custom_api_details["customApiKey"],
                    base_url=custom_api_details["customBaseURL"],
                )
                self.model_name = custom_api_details["customModelName"]

    def load_model(self):
        return self.model

    def generate(self, prompt: str, schema: BaseModel) -> BaseModel:
        client = self.load_model()
        if self.model_type == "claude":
            instructor_client = instructor.from_anthropic(client)
        else:
            instructor_client = instructor.from_openai(client)

        resp = instructor_client.messages.create(
            model=self.model_name,
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            response_model=schema,
        )
        return resp

    async def a_generate(self, prompt: str, schema: BaseModel) -> BaseModel:
        return self.generate(prompt, schema)

    def get_model_name(self):
        return self.model_name


# Use the job_wrapper decorator to handle job status updates
@tfl_evals.job_wrapper(progress_start=0, progress_end=100)
def run_evaluation():
    """Run DeepEval LLM Judge metrics"""
    # Setup evaluation logging
    tfl_evals.setup_eval_logging()

    # Type Casting for threshold and limit
    tfl_evals.threshold = float(tfl_evals.threshold) if hasattr(tfl_evals, "threshold") and tfl_evals.threshold else 0.5
    tfl_evals.limit = float(tfl_evals.limit) if tfl_evals.limit else 1.0

    # Parse tasks and prepare metrics
    tasks = tfl_evals.tasks.split(",")
    original_metrics = []
    for task in tasks:
        if task != "Custom (GEval)":
            original_metrics.append(task)
        else:
            original_metrics.append("GEval")

    tasks = [task.strip().replace(" ", "") + "Metric" if task != "Custom (GEval)" else "GEval" for task in tasks]

    # Setup GEval metrics configuration
    if "GEval" in tasks and not tfl_evals.context_geval:
        two_input_metrics.append("GEval")
    else:
        three_input_metrics.append("GEval")

    # Load model for evaluation
    try:
        if "local" not in tfl_evals.generation_model.lower():
            if "openai" in tfl_evals.generation_model.lower() or "gpt" in tfl_evals.generation_model.lower():
                trlab_model = CustomCommercialModel("openai", tfl_evals.generation_model)
            elif "claude" in tfl_evals.generation_model.lower() or "anthropic" in tfl_evals.generation_model.lower():
                trlab_model = CustomCommercialModel("claude", tfl_evals.generation_model)
            elif "custom" in tfl_evals.generation_model.lower():
                trlab_model = CustomCommercialModel("custom", "")
        else:
            check_local_server()
            custom_model = ChatOpenAI(
                api_key="dummy",
                base_url="http://localhost:8338/v1",
                model=tfl_evals.model_name,
            )
            trlab_model = TRLAB_MODEL(model=custom_model)

        print("Model loaded successfully")
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        raise

    tfl_evals.progress_update(10)

    try:
        dataset = tfl_evals.load_dataset()
        df = dataset["train"].to_pandas()
        print("Dataset loaded successfully")
    except Exception as e:
        print(f"An error occurred while reading the dataset: {e}")
        raise

    tfl_evals.progress_update(20)

    # Validate dataset structure
    required_columns = ["input", "output", "expected_output"]
    if not all(col in df.columns for col in required_columns):
        raise ValueError(
            "The dataset should have the columns `input`, `output` and `expected_output` mandatory. "
            "Please re-upload the dataset with the correct columns."
        )

    # Validate dataset content based on metrics
    if all(elem in two_input_metrics for elem in tasks):
        # Validate two-input metrics requirements
        if (
            not df["input"].notnull().all()
            or not df["output"].notnull().all()
            or not df["expected_output"].notnull().all()
        ):
            raise ValueError(
                f"The dataset should have all non-null values in the columns `input`, `output` and `expected_output` "
                f"columns for the metrics: {tasks}"
            )
    else:
        # Check if context column exists or create one
        if "context" not in df.columns:
            print("Using the expected_output column as the context")
            df["context"] = df["expected_output"]

        # Validate three-input metrics requirements
        if (
            not df["input"].notnull().all()
            or not df["output"].notnull().all()
            or not df["expected_output"].notnull().all()
            or not df["context"].notnull().all()
        ):
            raise ValueError(
                f"The dataset should have all non-null values in the columns `input`, `output`, `expected_output` "
                f"and `context` columns for the metrics: {tasks}"
            )

    tfl_evals.progress_update(30)

    # Initialize metrics
    metrics_arr = []
    for met in tasks:
        if met not in custom_metric:
            metric_class = get_metric_class(met)
            metric = metric_class(model=trlab_model, threshold=tfl_evals.threshold, include_reason=True)
            metrics_arr.append(metric)
        else:
            evaluation_params = [
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.EXPECTED_OUTPUT,
            ]
            if tfl_evals.context_geval:
                evaluation_params.append(LLMTestCaseParams.RETRIEVAL_CONTEXT)
            metric = GEval(
                name=tfl_evals.geval_name,
                criteria=tfl_evals.geval_criteria,
                evaluation_params=evaluation_params,
                model=trlab_model,
            )
            metrics_arr.append(metric)

    print("Metrics loaded successfully")
    tfl_evals.progress_update(40)

    # Create test cases
    test_cases = []
    if all(elem in two_input_metrics for elem in tasks):
        for _, row in df.iterrows():
            test_cases.append(
                LLMTestCase(input=row["input"], actual_output=row["output"], expected_output=row["expected_output"])
            )
    elif any(elem in three_input_metrics for elem in tasks):
        if "HallucinationMetric" not in tasks:
            for _, row in df.iterrows():
                test_cases.append(
                    LLMTestCase(
                        input=row["input"],
                        actual_output=row["output"],
                        expected_output=row["expected_output"],
                        retrieval_context=[row["context"]],
                    )
                )
        else:
            for _, row in df.iterrows():
                test_cases.append(
                    LLMTestCase(
                        input=row["input"],
                        actual_output=row["output"],
                        expected_output=row["expected_output"],
                        context=[row["context"]],
                    )
                )

    # Apply limit if specified
    if tfl_evals.limit and float(tfl_evals.limit) != 1.0:
        num_samples = max(int(len(test_cases) * float(tfl_evals.limit)), 1)
        test_cases = test_cases[:num_samples]

    print(f"Test cases loaded successfully: {len(test_cases)}")
    tfl_evals.progress_update(50)

    # Create and evaluate dataset
    dataset = EvaluationDataset(test_cases)
    output = evaluate(dataset, metrics_arr)

    tfl_evals.progress_update(80)

    # Process results
    additional_report = []
    for test_case in output.test_results:
        for metric in test_case.metrics_data:
            temp_report = {
                "test_case_id": test_case.name,
                "metric_name": metric.name,
                "score": metric.score,
                "input": test_case.input,
                "actual_output": test_case.actual_output,
                "expected_output": test_case.expected_output,
                "reason": metric.reason,
            }
            additional_report.append(temp_report)

    # Create metrics DataFrame and save results
    metrics_df = pd.DataFrame(additional_report)
    output_path, plot_data_path = tfl_evals.save_evaluation_results(metrics_df)

    # Log metrics to TensorBoard
    for metric in metrics_df["metric_name"].unique():
        metric_value = metrics_df[metrics_df["metric_name"] == metric]["score"].mean()
        tfl_evals.log_metric(metric, metric_value)

    tfl_evals.progress_update(100)
    print(f"Detailed Report saved to {output_path}")
    print(f"Metrics plot data saved to {plot_data_path}")
    print("Evaluation completed successfully")

    return output_path, plot_data_path


# Run the evaluation when script is executed
run_evaluation()
