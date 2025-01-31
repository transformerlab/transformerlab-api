import argparse
import importlib
import os
import sys
import traceback


import pandas as pd
import requests
from deepeval import evaluate
from deepeval.dataset import EvaluationDataset
from deepeval.metrics import GEval
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from anthropic import Anthropic
from openai import OpenAI
import instructor

parser = argparse.ArgumentParser(
    description='Run DeepEval metrics for LLM-as-judge evaluation.')
parser.add_argument('--model_name', default='gpt-j-6b', type=str,
                    help='Model to use for evaluation.')
parser.add_argument('--experiment_name', default='', type=str)
parser.add_argument('--eval_name', default='', type=str)
parser.add_argument('--task', default='', type=str)
parser.add_argument("--model_adapter", default=None, type=str,)
parser.add_argument("--dataset_path", default=None, type=str,)
parser.add_argument("--threshold", default=0.5, type=float)
parser.add_argument("--geval_name", default='', type=str)
parser.add_argument("--geval_context", default='', type=str)
parser.add_argument("--context_geval", default=None, type=str)
parser.add_argument("--judge_model", default=None, type=str)
parser.add_argument("--commercial_api_key", default=None, type=str)

args, other = parser.parse_known_args()

if args.task != "Custom (GEval)":
    args.task = args.task.strip().replace(" ", "") + "Metric"
else:
    args.task = "GEval"


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


class CustomCommercialModel(DeepEvalBaseLLM):
    def __init__(self, model_type="claude", model_name="Claude 3.5 Sonnet"):
        self.model_type = model_type
        self.model_name = self.set_model_name(model_name)
        if model_type == "claude":
            if os.environ.get("ANTHROPIC_API_KEY") is None and args.commercial_api_key is not None:
                os.environ["ANTHROPIC_API_KEY"] = args.commercial_api_key
            elif os.environ.get("ANTHROPIC_API_KEY") is None and args.commercial_api_key is None:
                print("Please set the Commerical API Key field")
                sys.exit(1)
            self.model = Anthropic()

        elif model_type == "openai":
            if os.environ.get("OPENAI_API_KEY") is None and args.commercial_api_key is not None:
                os.environ["OPENAI_API_KEY"] = args.commercial_api_key
            if os.environ.get("OPENAI_API_KEY") is None and args.commercial_api_key is None:
                print("Please set the Commerical API Key field")
                sys.exit(1)

            self.model = OpenAI()

    def load_model(self):
        return self.model

    def set_model_name(self, model_name):
        dic = {
            "Claude 3.5 Sonnet": "claude-3.5-sonnet-latest",
            "Claude 3.5 Haiku": "claude-3.5-haiku-latest",
            "OpenAI GPT 4o": "gpt-4o",
            "OpenAI GPT 4o Mini": "gpt-4o-mini",
        }
        return dic[model_name]

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


def check_local_server():
    response = requests.get('http://localhost:8338/server/worker_healthz')
    if response.status_code != 200 or not isinstance(response.json(), list) or len(response.json()) == 0:
        print("Local Model Server is not running. Please start it before running the evaluation.")
        sys.exit(1)


try:
    if 'local' not in args.judge_model.lower():
        if 'openai' in args.judge_model.lower():
            trlab_model = CustomCommercialModel("openai", args.judge_model)
        else:
            trlab_model = CustomCommercialModel("claude", args.judge_model)
    else:
        check_local_server()
        custom_model = ChatOpenAI(
            api_key="dummy",
            base_url="http://localhost:8338/v1",
            model=args.model_name,
        )
        trlab_model = TRLAB_MODEL(model=custom_model)


except Exception as e:
    print(f"An error occurred while loading the model: {e}")
    sys.exit(1)

print("Model loaded successfully")


two_input_metrics = ["AnswerRelevancyMetric", "BiasMetric", "ToxicityMetric"]
three_input_metrics = ["FaithfulnessMetric", "ContextualPrecisionMetric",
                       "ContextualRecallMetric", "ContextualRelevancyMetric", "HallucinationMetric"]

custom_metric = ["GEval"]

try:
    if args.task == "GEval":
        if not args.context_geval:
            two_input_metrics.append("GEval")
        else:
            three_input_metrics.append("GEval")
except Exception as e:
    print(f"An error occurred while checking the metric: {e}")
    sys.exit(1)


def run_evaluation():
    # Load the csv_file
    if args.dataset_path is None:
        print("No dataset found. Please upload the dataset correctly")
        sys.exit(1)
    if not os.path.exists(args.dataset_path):
        print(f"No dataset found at {args.dataset_path}")
        sys.exit(1)

    # check_local_server()
    try:
        df = pd.read_csv(args.dataset_path)
        print("Dataset loaded successfully")
    except Exception as e:
        print(f"An error occurred while reading the dataset: {e}")
        sys.exit(1)

    required_columns = ['input', 'output', 'expected_output']
    # Check if df has the specified columns
    if not all(col in df.columns for col in required_columns):
        print(
            f"The dataset should have the columns `input`, `output` and `expected_output` mandatory. Please re-upload the dataset with the correct columns.")
        sys.exit(1)

    if args.task in two_input_metrics:
        # Make sure the df has all non-null values in the columns `input` and `output`
        if not df['input'].notnull().all() or not df['output'].notnull().all() or not df['expected_output'].notnull().all():
            print(
                f"The dataset should have all non-null values in the columns `input`, `output` and `expected_output` columns for the metric: {args.task}")
            sys.exit(1)
    elif args.task in three_input_metrics:
        # Check if there is a column called `context` in the dataset
        if 'context' not in df.columns:
            print(
                f"The dataset should have a column `context` for the metric: {args.task}")
            print("Using the expected_output column as the context")
            df["context"] = df["expected_output"]

        # Make sure the df has all non-null values in the columns `input`, `output`, and `context`
        if not df['input'].notnull().all() or not df['output'].notnull().all() or not df['expected_output'].notnull().all() or not df['context'].notnull().all():
            print(
                f"The dataset should have all non-null values in the columns `input`, `output`, `expected_output` and `context` columns for the metric: {args.task}")
            sys.exit(1)

    if args.task not in custom_metric:
        metric_class = get_metric_class(args.task)
        metric = metric_class(
            model=trlab_model, threshold=args.threshold, include_reason=True)
        print("Metric loaded successfully")
        test_cases = []
        if args.task in two_input_metrics:
            for _, row in df.iterrows():
                try:
                    # print("ROW", row)
                    test_cases.append(LLMTestCase(
                        input=row['input'], actual_output=row['output'], expected_output=row['context']))
                    # print("TEST CASES", test_cases)
                except Exception as e:
                    print(
                        f"An error occurred while creating the test case: {e}")
                    # sys.exit(1)
        elif args.task in three_input_metrics:
            if args.task != "HallucinationMetric":
                for _, row in df.iterrows():
                    test_cases.append(LLMTestCase(
                        input=row['input'], actual_output=row['output'], expected_output=row['context'], retrieval_context=[row['context']]))
            else:
                for _, row in df.iterrows():
                    test_cases.append(LLMTestCase(
                        input=row['input'], actual_output=row['output'], expected_output=row['context'], context=[row['context']]))

    else:
        try:
            evaluation_params = [LLMTestCaseParams.INPUT,
                                 LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT]
            if args.context_geval:
                evaluation_params.append(LLMTestCaseParams.RETRIEVAL_CONTEXT)
            metric = GEval(name=args.geval_name,
                           criteria=args.geval_context,
                           evaluation_params=[
                               LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
                           model=trlab_model,
                           )
            test_cases = []
            for _, row in df.iterrows():
                if args.context_geval:
                    test_cases.append(LLMTestCase(
                        input=row['input'], actual_output=row['output'], expected_output=row['context'], retrieval_context=[row['context']]))
                else:
                    test_cases.append(LLMTestCase(
                        input=row['input'], actual_output=row['output'], expected_output=row['context']))
        except Exception as e:
            print(f"An error occurred while creating the test case: {e}")
            sys.exit(1)

    print(f"Test cases loaded successfully: {len(test_cases)}")
    dataset = EvaluationDataset(test_cases[1:])

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
