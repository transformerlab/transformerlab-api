from langchain_openai import ChatOpenAI
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase
from deepeval import evaluate
from deepeval.dataset import EvaluationDataset
import importlib
import argparse
import sys
import traceback
import os
import pandas as pd
import requests


parser = argparse.ArgumentParser(
    description='Run Eleuther AI LM Evaluation Harness.')
parser.add_argument('--model_name', default='gpt-j-6b', type=str,
                    help='Model to use for evaluation.')
parser.add_argument('--experiment_name', default='', type=str)
parser.add_argument('--eval_name', default='', type=str)
parser.add_argument('--task', default='', type=str)
parser.add_argument("--model_adapter", default=None, type=str,)
parser.add_argument("--dataset_path", default=None, type=str,)
parser.add_argument("--threshold", default=0.5, type=float)

args, other = parser.parse_known_args()

args.task = args.task.strip().replace(" ", "") + "Metric"


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
        sys.exit(1)


# Generating custom TRLAB model
class TRLAB_MODEL(DeepEvalBaseLLM):
    def __init__(
        self,
        model
    ):
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
        return args.model_name


# Replace these with real values
custom_model = ChatOpenAI(
    api_key="dummy",
    base_url="http://localhost:8338/v1",
    model=args.model_name,
)
trlab_model = TRLAB_MODEL(model=custom_model)

print("Model loaded successfully")


def check_local_server():
    response = requests.get('http://localhost:8338/server/worker_healthz')
    print(response.json())
    if response.status_code != 200 or not isinstance(response.json(), list) or len(response.json()) == 0:
        print("Local Model Server is not running. Please start it before running the evaluation.")
        sys.exit(1)


two_input_metrics = ["AnswerRelevancyMetric", "BiasMetric", "ToxicityMetric"]
three_input_metrics = ["FaithfulnessMetric", "ContextualPrecisionMetric",
                       "ContextualRecallMetric", "ContextualRelevancyMetric", "HallucinationMetric"]


def run_evaluation():
    # Load the csv_file
    if args.dataset_path is None:
        print("No dataset found. Please upload the dataset correctly")
        sys.exit(1)
    if not os.path.exists(args.dataset_path):
        print(f"No dataset found at {args.dataset_path}")
        sys.exit(1)

    check_local_server()

    df = pd.read_csv(args.dataset_path)

    if args.task in two_input_metrics:
        # Make sure the df has all non-null values in the columns `input` and `output`
        if not df['input'].notnull().all() or not df['output'].notnull().all():
            print(
                f"The dataset should have all non-null values in the columns `input` and `output` columns for the metric: {args.task}")
            sys.exit(1)
    elif args.task in three_input_metrics:
        # Check if there is a column called `context` in the dataset
        if 'context' not in df.columns:
            print(
                f"The dataset should have a column `context` for the metric: {args.task}")
            sys.exit(1)
        # Make sure the df has all non-null values in the columns `input`, `output`, and `context`
        if not df['input'].notnull().all() or not df['output'].notnull().all() or not df['context'].notnull().all():
            print(
                f"The dataset should have all non-null values in the columns `input`, `output`, and `context` columns for the metric: {args.task}")
            sys.exit(1)

    if args.task in two_input_metrics or args.task in three_input_metrics:
        metric_class = get_metric_class(args.task)
        metric = metric_class(
            model=trlab_model, threshold=args.threshold, include_reason=True)
        print("Metric loaded successfully")
        test_cases = []

        if args.task in two_input_metrics:
            for _, row in df.iterrows():
                test_cases.append(LLMTestCase(
                    input=row['input'], actual_output=row['output']))
        elif args.task in three_input_metrics:
            if args.task != "HallucinationMetric":
                for _, row in df.iterrows():
                    test_cases.append(LLMTestCase(
                        input=row['input'], actual_output=row['output'], expected_output=row['context'], retrieval_context=[row['context']]))
            else:
                for _, row in df.iterrows():
                    test_cases.append(LLMTestCase(
                        input=row['input'], actual_output=row['output'], context=[row['context']]))

        print(f"Test cases loaded successfully: {len(test_cases)}")

        dataset = EvaluationDataset(test_cases)

    try:
        output = evaluate(dataset, [metric])
    except ValueError as e:
        # Print the whole traceback
        traceback.print_exc()
        print(f"An error occurred while running the evaluation: {e}")
        sys.exit(1)

    print(output)
    print("Evaluation completed successfully")


run_evaluation()
