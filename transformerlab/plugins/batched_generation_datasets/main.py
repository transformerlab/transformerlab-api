import argparse
import os
import sys
from typing import List
import asyncio

import instructor
import pandas as pd
import requests
import transformerlab.plugin
from anthropic import Anthropic
from deepeval.models.base_model import DeepEvalBaseLLM
from langchain_openai import ChatOpenAI
from openai import OpenAI
from pydantic import BaseModel
from langchain.schema import SystemMessage, HumanMessage

from requests_batching import process_dataset


parser = argparse.ArgumentParser(description="Run Synthesizer for generating data.")
parser.add_argument(
    "--run_name",
    default="test",
    type=str,
)
parser.add_argument("--model_name", default="gpt-j-6b", type=str, help="Model to use for evaluation.")
parser.add_argument("--dataset_name", default="test", type=str)
parser.add_argument(
    "--model_adapter",
    default=None,
    type=str,
)
parser.add_argument("--dataset_split", default="train", type=str)
parser.add_argument("--generation_model", default="Local", type=str)
parser.add_argument("--input_column", default=None, type=str)
parser.add_argument("--output_column", default=None, type=str)
parser.add_argument("--experiment_name", default="test", type=str)
parser.add_argument("--job_id", default=None, type=str)
parser.add_argument("--system_prompt", default=None, type=str)
parser.add_argument("--batch_size", default=128, type=int)
parser.add_argument("--temperature", default=0.01, type=float)
parser.add_argument("--max_tokens", default=1024, type=int)
parser.add_argument("--top_p", default=1.0, type=float)


args, other = parser.parse_known_args()

if args.job_id:
    job = transformerlab.plugin.Job(args.job_id)
    job.update_progress(0)
else:
    print("Job ID not provided.")
    sys.exit(1)

job.update_progress(0)


def check_local_server():
    response = requests.get("http://localhost:8338/server/worker_healthz")
    if response.status_code != 200 or not isinstance(response.json(), list) or len(response.json()) == 0:
        print("Local Model Server is not running. Please start it before running the evaluation.")
        sys.exit(1)


def fetch_dataset():
    if args.dataset_name is None or len(args.dataset_name.strip()) <= 1:
        print("Dataset not provided.")
        sys.exit(1)
    try:
        params = {"dataset_id": args.dataset_name, "split": args.dataset_split}
        response = requests.get("http://localhost:8338/data/get", params=params)
        if response.status_code != 200:
            print(f"Error fetching the dataset: {response.json()}")
            job.set_job_completion_status("failed", f"Error fetching the dataset: {response.json()}")
            sys.exit(1)
        else:
            df = pd.DataFrame(response.json()["data"]["columns"])
            file_location = response.json()["data"]["file_location"]
            if file_location is None or not os.path.exists(file_location):
                file_location = os.path.join(
                    os.environ.get("_TFL_WORKSPACE_DIR"), "datasets", args.dataset_name + "_with_outputs"
                )
            return df, file_location
    except Exception as e:
        print(f"An error occurred while fetching the dataset: {e}")
        job.set_job_completion_status("failed", f"An error occurred while fetching the dataset: {e}")
        sys.exit(1)


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
            batch_size=args.batch_size,
            model=args.model_name,
            inference_url="http://localhost:8338/v1/chat/completions",
            api_key="dummy",
            sys_prompt_col=sys_prompt_col,
            input_col=args.input_column,
            output_col=args.output_column,
            temperature=float(args.temperature),
            max_tokens=int(args.max_tokens),
            top_p=float(args.top_p),
        )
        return updated_df

    def get_model_name(self):
        return args.model_name


class CustomCommercialModel(DeepEvalBaseLLM):
    def __init__(self, model_type="claude", model_name="Claude 3.5 Sonnet"):
        self.model_type = model_type
        self.model_name = self.set_model_name(model_name)
        if model_type == "claude":
            if os.environ.get("ANTHROPIC_API_KEY") is None:
                print("Please set the Anthropic API Key from Settings.")
                job.set_job_completion_status("failed", "Please set the Anthropic API Key from Settings.")
                sys.exit(1)
            self.model = Anthropic()

        elif model_type == "openai":
            if os.environ.get("OPENAI_API_KEY") is None:
                print("Please set the OpenAI API Key from Settings.")
                job.set_job_completion_status("failed", "Please set the OpenAI API Key from Settings.")
                sys.exit(1)
            self.model = OpenAI()

    def load_model(self):
        return self.model

    def set_model_name(self, model_name):
        dic = {
            "Claude 3.5 Sonnet": "claude-3-5-sonnet-latest",
            "Claude 3.5 Haiku": "claude-3-5-haiku-latest",
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
            batch_size=args.batch_size,
            model=self.model,
            inference_url=INFERENCE_URL,
            api_key=api_key,
            sys_prompt_col=sys_prompt_col,
            input_col=args.input_column,
            output_col=args.output_column,
            temperature=float(args.temperature),
            max_tokens=int(args.max_tokens),
            top_p=float(args.top_p),
        )
        return updated_df

    def get_model_name(self):
        return self.model_name


try:
    if "local" not in args.generation_model.lower():
        if "openai" in args.generation_model.lower():
            trlab_model = CustomCommercialModel("openai", args.generation_model)
        else:
            trlab_model = CustomCommercialModel("claude", args.generation_model)
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
    job.set_job_completion_status("failed", "An error occurred while loading the model")
    sys.exit(1)

print("Model loaded successfully")
job.update_progress(0.5)


async def run_batched_generation():
    try:
        check_local_server()
        df, file_location = fetch_dataset()
        print("Dataset fetched successfully")
        sys_prompt_col = None
        if args.system_prompt:
            df["system_prompt"] = args.system_prompt
            sys_prompt_col = "system_prompt"
        df = await trlab_model.generate_batched(df, sys_prompt_col=sys_prompt_col)
        print("Batched generation completed successfully")
        job.update_progress(99)
        # List all files in the directory file_location ending with .json
        json_files = [f for f in os.listdir(file_location) if f.endswith(".json")]
        # Iterate over the list of json files
        for json_file in json_files:
            # Get the full path of the file
            file_path = os.path.join(file_location, json_file)
            # Delete the file
            os.remove(file_path)
        print("Replacing the old dataset with the new one having outputs...")
        file_name = os.path.join(file_location, args.dataset_name + "_" + args.job_id + "_with_outputs.json")

        df.to_json(file_name, orient="records", lines=False)
        job.update_progress(100)
        job.set_job_completion_status(
            "success", f"Dataset {args.dataset_name} updated successfully", additional_output_path=file_name
        )

        print("Updated the dataset successfully")
    except Exception as e:
        print(f"An error occurred while running batched generation: {e}")
        job.set_job_completion_status("failed", f"An error occurred while running batched generation: {e}")
        sys.exit(1)


print("Running batched generation...")
asyncio.run(run_batched_generation())
