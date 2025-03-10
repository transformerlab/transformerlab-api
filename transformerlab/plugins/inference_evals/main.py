try:
    import argparse
    import asyncio
    import json
    import os
    import sys
    import time
    from typing import List

    import instructor
    import pandas as pd
    import requests
    from datetime import datetime
    from anthropic import Anthropic
    from datasets import load_dataset
    from deepeval.models.base_model import DeepEvalBaseLLM
    from langchain.schema import HumanMessage, SystemMessage
    from langchain_openai import ChatOpenAI
    from openai import OpenAI
    from pydantic import BaseModel
    from requests_batching import process_dataset
    from tensorboardX import SummaryWriter


    import transformerlab.plugin

    parser = argparse.ArgumentParser(description="Run Synthesizer for generating data.")
    parser.add_argument(
        "--run_name",
        default="test",
        type=str,
    )
    parser.add_argument("--tasks", default="task", type=str)
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
    parser.add_argument("--eval_name", default="", type=str)
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

    args.tasks = args.tasks.split(",")
    metrics_map = {
        "Time to First Token (TTFT)": "time_to_first_token",
        "Total Time": "time_total",
        "Prompt Tokens": "prompt_tokens",
        "Completion Tokens": "completion_tokens",
        "Total Tokens": "total_tokens",
        "Tokens per Second": "tokens_per_second",
    }

    mapped_metrics = [metrics_map[x] for x in args.tasks if x in metrics_map.keys()]

    today = time.strftime("%Y%m%d-%H%M%S")
    tensorboard_dir = os.path.join(os.environ["_TFL_WORKSPACE_DIR"], "experiments", args.experiment_name, "tensorboards")
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)
    # Find directory to put in based on eval name
    combined_tensorboard_dir = None
    for dir in os.listdir(tensorboard_dir):
        if args.run_name == dir or args.run_name == dir.lower():
            combined_tensorboard_dir = os.path.join(tensorboard_dir, dir)
    if combined_tensorboard_dir is None:
        combined_tensorboard_dir = os.path.join(tensorboard_dir, args.run_name)
    output_dir = os.path.join(combined_tensorboard_dir, f"evaljob_{args.job_id}_{today}")
    os.makedirs(output_dir, exist_ok=True)
    writer = SummaryWriter(output_dir)
    job.set_tensorboard_output_dir(output_dir)
    print("Writing tensorboard logs to", output_dir)

except Exception as e:
    print(f"An error occurred while importing libraries: {e}")
    job.set_job_completion_status("failed", "An error occurred while importing libraries.")
    sys.exit(1)


print("ARGS", args)
try:
    job.add_to_job_data("config", str(args))
    job.add_to_job_data("template_name", args.run_name)
    job.add_to_job_data("model_name", args.model_name)
    job.add_to_job_data("start_time", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
except Exception as e:
    print(f"An error occurred while adding job data: {e}")
    job.add_to_job_data("end_time", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    job.set_job_completion_status("failed", "An error occurred while adding job data.")
    sys.exit(1)


def check_local_server():
    response = requests.get("http://localhost:8338/server/worker_healthz")
    if response.status_code != 200 or not isinstance(response.json(), list) or len(response.json()) == 0:
        print("Local Model Server is not running. Please start it before running the evaluation.")
        sys.exit(1)


def get_tflab_dataset():
    try:
        dataset_target = transformerlab.plugin.get_dataset_path(args.dataset_name)
    except Exception as e:
        job.set_job_completion_status("failed", "Failure to get dataset")
        raise e
    dataset = {}
    dataset_types = ["train"]
    for dataset_type in dataset_types:
        try:
            dataset[dataset_type] = load_dataset(dataset_target, split=dataset_type, trust_remote_code=True)

        except Exception as e:
            job.set_job_completion_status("failed", "Failure to load dataset")
            raise e
    # Convert the dataset to a pandas dataframe
    df = dataset["train"].to_pandas()
    return df


def get_output_file_path():
    experiment_dir = os.path.join(os.environ["_TFL_WORKSPACE_DIR"], "experiments", args.experiment_name)
    p = os.path.join(experiment_dir, "evals", args.eval_name)
    os.makedirs(p, exist_ok=True)
    return os.path.join(p, f"detailed_output_{args.job_id}.csv")


def get_plotting_data(metrics_df):
    file_path = get_output_file_path()
    file_path = file_path.replace(".csv", "_plotting.json")
    metrics_data = metrics_df[["test_case_id", "metric_name", "score"]].copy()
    metrics_data["metric_name"] = metrics_data["metric_name"].apply(lambda x: x.replace("_", " ").title())
    metrics_data.to_json(file_path, orient="records", lines=False)

    return file_path


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
        self.model_name = model_name
        if model_type == "claude":
            anthropic_api_key = transformerlab.plugin.get_db_config_value("ANTHROPIC_API_KEY")
            if not anthropic_api_key or anthropic_api_key.strip() == "":
                print("Please set the Anthropic API Key from Settings.")
                job.set_job_completion_status("failed", "Please set the Anthropic API Key from Settings.")
                sys.exit(1)
            else:
                os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key
            self.model = Anthropic()

        elif model_type == "openai":
            openai_api_key = transformerlab.plugin.get_db_config_value("OPENAI_API_KEY")
            if not openai_api_key or openai_api_key.strip() == "":
                print("Please set the OpenAI API Key from Settings.")
                job.set_job_completion_status("failed", "Please set the OpenAI API Key from Settings.")
                sys.exit(1)
            else:
                os.environ["OPENAI_API_KEY"] = openai_api_key
            self.model = OpenAI()

        elif model_type == "custom":
            custom_api_details = transformerlab.plugin.get_db_config_value("CUSTOM_MODEL_API_KEY")
            if not custom_api_details or custom_api_details.strip() == "":
                print("Please set the Custom API Details from Settings.")
                job.set_job_completion_status("failed", "Please set the Custom API Details from Settings.")
                sys.exit(1)
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
            batch_size=args.batch_size,
            model=self.model_name,
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
        if "openai" in args.generation_model.lower() or "gpt" in args.generation_model.lower():
            trlab_model = CustomCommercialModel("openai", args.generation_model)
        elif "claude" in args.generation_model.lower() or "anthropic" in args.generation_model.lower():
            trlab_model = CustomCommercialModel("claude", args.generation_model)
        elif "custom" in args.generation_model.lower():
            trlab_model = CustomCommercialModel("custom", "")
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
        df = get_tflab_dataset()
        print("Dataset fetched successfully")
        sys_prompt_col = None
        if args.system_prompt:
            df["system_prompt"] = args.system_prompt
            sys_prompt_col = "system_prompt"
        job.update_progress(20)
        df = await trlab_model.generate_batched(df, sys_prompt_col=sys_prompt_col)
        print("Batched generation completed successfully")
        job.update_progress(90)
        
        score_list = []
        metrics = []
        for metric in args.tasks:
            score_list.append({"type": metric, "score": df[metrics_map[metric]].mean()})
            writer.add_scalar(f"eval/{metric}", df[metrics_map[metric]].mean(), 1)
        

        for idx, row in df.iterrows():
            for metric in args.tasks:
                metrics.append( 
                    {
                    "test_case_id": f"test_case_{idx}",
                    "metric_name": metric,
                    "score": round(float(row[metrics_map[metric]]), 4),
                    "input": row[args.input_column],
                    "output": row[args.output_column]
                    }
                )
        # for metrics in :
            # scores_list.append({"type":"time_to_first_token", "value":row["time_to_first_token"]})
        metrics_df = pd.DataFrame(metrics)
        output_path = get_output_file_path()
        metrics_df.to_csv(output_path, index=False)
        job.update_progress(99)
        plot_data_path = get_plotting_data(metrics_df)
        print(f"Metrics saved to {output_path}")
        print(f"Plotting data saved to {plot_data_path}")
        print("Evaluation completed.")
        job.update_progress(100)
        job.add_to_job_data("end_time", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        job.set_job_completion_status(
            "success",
            "Evaluation completed successfully.",
            score=score_list,
            additional_output_path=output_path,
            plot_data_path=plot_data_path,
        )
    except Exception as e:
        print(f"An error occurred while running batched generation: {e}")
        job.set_job_completion_status("failed", f"An error occurred while running batched generation: {e}")
        sys.exit(1)

print("Running batched generation...")
asyncio.run(run_batched_generation())
