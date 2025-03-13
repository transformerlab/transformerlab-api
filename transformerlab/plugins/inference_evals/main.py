import json
import os
from typing import List

import instructor
import pandas as pd
import requests
from anthropic import Anthropic
from deepeval.models.base_model import DeepEvalBaseLLM
from langchain.schema import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from openai import OpenAI
from pydantic import BaseModel
from requests_batching import process_dataset

import transformerlab.plugin

from transformerlab.tfl_decorators import tfl_evals

# Metrics mapping
metrics_map = {
    "Time to First Token (TTFT)": "time_to_first_token",
    "Total Time": "time_total",
    "Prompt Tokens": "prompt_tokens",
    "Completion Tokens": "completion_tokens",
    "Total Tokens": "total_tokens",
    "Tokens per Second": "tokens_per_second",
}


def check_local_server():
    """Check if the local model server is running"""
    response = requests.get("http://localhost:8338/server/worker_healthz")
    if response.status_code != 200 or not isinstance(response.json(), list) or len(response.json()) == 0:
        print("Local Model Server is not running. Please start it before running the evaluation.")
        raise RuntimeError("Local Model Server is not running. Please start it before running the evaluation.")


# Generating custom TRLAB model
class TRLAB_MODEL(DeepEvalBaseLLM):
    def __init__(self, model):
        self.model = model

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        return chat_model.invoke(prompt).content

    def generate_without_instructor(self, messages: List[dict]) -> BaseModel:
        chat_model = self.load_model()
        modified_messages = []
        for message in messages:
            if message["role"] == "system":
                modified_messages.append(SystemMessage(**message))
            else:
                modified_messages.append(HumanMessage(**message))
        return chat_model.invoke(modified_messages).content

    async def a_generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        res = await chat_model.ainvoke(prompt)
        return res.content

    async def generate_batched(self, df: pd.DataFrame, sys_prompt_col=None) -> pd.DataFrame:
        updated_df = await process_dataset(
            df,
            batch_size=tfl_evals.batch_size,
            model=tfl_evals.model_name,
            inference_url="http://localhost:8338/v1/chat/completions",
            api_key="dummy",
            sys_prompt_col=sys_prompt_col,
            input_col=tfl_evals.input_column,
            output_col=tfl_evals.output_column,
            temperature=float(tfl_evals.temperature),
            max_tokens=int(tfl_evals.max_tokens),
            top_p=float(tfl_evals.top_p),
        )
        return updated_df

    def get_model_name(self):
        return tfl_evals.model_name


class CustomCommercialModel(DeepEvalBaseLLM):
    def __init__(self, model_type="claude", model_name="Claude 3.5 Sonnet"):
        self.model_type = model_type
        self.model_name = model_name

        if model_type == "claude":
            anthropic_api_key = transformerlab.plugin.get_db_config_value("ANTHROPIC_API_KEY")
            if not anthropic_api_key or anthropic_api_key.strip() == "":
                raise ValueError("Please set the Anthropic API Key from Settings.")
            else:
                os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key
            self.model = Anthropic()

        elif model_type == "openai":
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

    def generate_without_instructor(self, messages: List[dict]) -> BaseModel:
        client = self.load_model()
        response = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
        )
        return response.choices[0].message.content

    async def a_generate(self, prompt: str, schema: BaseModel) -> BaseModel:
        return self.generate(prompt, schema)

    async def generate_batched(self, df: pd.DataFrame, sys_prompt_col=None) -> pd.DataFrame:
        INFERENCE_URL = (
            "https://api.openai.com/v1/chat/completions"
            if self.model_type == "openai"
            else "https://api.anthropic.com/v1/chat/completions"
        )
        api_key = (
            os.environ.get("OPENAI_API_KEY") if self.model_type == "openai" else os.environ.get("ANTHROPIC_API_KEY")
        )
        updated_df = await process_dataset(
            df,
            batch_size=tfl_evals.batch_size,
            model=self.model_name,
            inference_url=INFERENCE_URL,
            api_key=api_key,
            sys_prompt_col=sys_prompt_col,
            input_col=tfl_evals.input_column,
            output_col=tfl_evals.output_column,
            temperature=float(tfl_evals.temperature),
            max_tokens=int(tfl_evals.max_tokens),
            top_p=float(tfl_evals.top_p),
        )
        return updated_df

    def get_model_name(self):
        return self.model_name


@tfl_evals.async_job_wrapper(progress_start=0, progress_end=100)
async def run_evaluation():
    """Run the inference evaluation"""

    # Setup evaluation logging
    tfl_evals.setup_eval_logging()

    # Type casting for avoiding errors
    tfl_evals.batch_size = int(tfl_evals.batch_size)
    tfl_evals.temperature = float(tfl_evals.temperature)
    tfl_evals.max_tokens = int(tfl_evals.max_tokens)
    tfl_evals.top_p = float(tfl_evals.top_p)

    # Parse the tasks
    tasks = tfl_evals.tasks.split(",")

    tfl_evals.progress_update(10)

    # Load the appropriate model
    try:
        if hasattr(tfl_evals, "generation_model") and "local" not in tfl_evals.generation_model.lower():
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

    tfl_evals.progress_update(20)

    # Load the dataset
    dataset = tfl_evals.load_dataset()
    df = dataset["train"].to_pandas()

    print("Dataset fetched successfully")

    sys_prompt_col = None
    if hasattr(tfl_evals, "system_prompt") and tfl_evals.system_prompt:
        df["system_prompt"] = tfl_evals.system_prompt
        sys_prompt_col = "system_prompt"

    tfl_evals.progress_update(30)

    # Run batch generation
    df = await trlab_model.generate_batched(df, sys_prompt_col=sys_prompt_col)
    print("Batched generation completed successfully")

    tfl_evals.progress_update(70)

    # Process metrics
    metrics = []
    for metric in tasks:
        metric_value = df[metrics_map[metric]].mean()
        tfl_evals.log_metric(metric, metric_value)

    # Create metrics DataFrame
    for idx, row in df.iterrows():
        for metric in tasks:
            if row[metrics_map[metric]] is not None:
                score = round(float(row[metrics_map[metric]]), 4)
            else:
                score = 0.0
            metrics.append(
                {
                    "test_case_id": f"test_case_{idx}",
                    "metric_name": metric,
                    "score": score,
                    "input": row[tfl_evals.input_column],
                    "output": row[tfl_evals.output_column],
                }
            )

    tfl_evals.progress_update(80)

    # Save results using EvalsTFLPlugin
    metrics_df = pd.DataFrame(metrics)
    output_path, plot_data_path = tfl_evals.save_evaluation_results(metrics_df)

    print(f"Metrics saved to {output_path}")
    print(f"Plotting data saved to {plot_data_path}")
    print("Evaluation completed.")

    return True


run_evaluation()
