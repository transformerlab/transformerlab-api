import argparse
import os
import sys
import traceback
from typing import List

import instructor
import requests
import transformerlab.plugin
from anthropic import Anthropic
from deepeval.models import DeepEvalBaseEmbeddingModel
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.synthesizer import Synthesizer
from deepeval.synthesizer.config import ContextConstructionConfig
from langchain_openai import ChatOpenAI
from openai import OpenAI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

parser = argparse.ArgumentParser(description="Run Synthesizer for generating data.")
parser.add_argument(
    "--run_name",
    default="test",
    type=str,
)
parser.add_argument("--model_name", default="gpt-j-6b", type=str, help="Model to use for evaluation.")
parser.add_argument("--embedding_model", default="Snowflake/arctic-embed-m", type=str)
parser.add_argument("--dataset_name", default="test", type=str)
parser.add_argument(
    "--model_adapter",
    default=None,
    type=str,
)
parser.add_argument("--generation_model", default="claude-3-5-sonnet", type=str)
parser.add_argument("--generation_type", default="docs", type=str, help="Type of generation: scratch, context, docs")
parser.add_argument("--evolution_config", default=None, type=str)
parser.add_argument("--docs", default=None, type=str)
parser.add_argument("--context", default=None, type=str)
parser.add_argument("--input_format", default=None, type=str)
parser.add_argument("--expected_output_format", default=None, type=str)
parser.add_argument("--task", default=None, type=str)
parser.add_argument("--scenario", default=None, type=str)
parser.add_argument("--experiment_name", default="test", type=str)
parser.add_argument("--job_id", default=None, type=str)
parser.add_argument("--chunk_size", default=256, type=int)
parser.add_argument("--max_contexts_per_document", default=10, type=int)
parser.add_argument("--max_context_length", default=3, type=int)
parser.add_argument("--max_goldens_per_context", default=2, type=int)


args, other = parser.parse_known_args()

# Set the generation type to `docs`
args.generation_type = "docs"


if args.generation_type not in ["scratch", "context", "docs"]:
    print("Generation type must be one of `scratch`, `context`, `docs`")
    sys.exit(1)

if args.job_id:
    job = transformerlab.plugin.Job(args.job_id)
    job.update_progress(0)
else:
    print("Job ID not provided.")
    sys.exit(1)

job.update_progress(0)

print(f"Generation type: {args.generation_type}")


def check_local_server():
    response = requests.get("http://localhost:8338/server/worker_healthz")
    print(response.json())
    if response.status_code != 200 or not isinstance(response.json(), list) or len(response.json()) == 0:
        print("Local Model Server is not running. Please start it before running the evaluation.")
        sys.exit(1)


class CustomEmbeddingModel(DeepEvalBaseEmbeddingModel):
    def __init__(self, model_name: str = "Snowflake/arctic-embed-m"):
        self.model_name = model_name
        self.model = self.load_model()

    def load_model(self):
        print(f"Loading Embedding Model... : {self.model_name.strip()}")
        try:
            model = SentenceTransformer(self.model_name.strip(), trust_remote_code=True)
            return model
        except Exception as e:
            print(f"An error occurred while loading the model: {e}")
            job.set_job_completion_status("failed", "An error occurred while loading the model")
            sys.exit(1)

    def embed_text(self, text: str) -> List[float]:
        return self.model.encode(text, convert_to_numpy=True).tolist()

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, convert_to_numpy=True).tolist()

    async def a_embed_text(self, text: str) -> List[float]:
        return self.embed_text(text)

    async def a_embed_texts(self, texts: List[str]) -> List[List[float]]:
        return self.embed_texts(texts)

    def get_model_name(self):
        return f"Custom HuggingFace Embedding Model ({self.model_name})"


# Generating custom TRLAB model
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

    async def a_generate(self, prompt: str, schema: BaseModel) -> BaseModel:
        return self.generate(prompt, schema)

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

if args.generation_type == "docs":
    if not args.embedding_model or len(args.embedding_model.strip()) <= 1:
        args.embedding_model = "Snowflake/arctic-embed-m"
        print(f"Embedding model not provided. Using default model: {args.embedding_model}")
    try:
        embedder = CustomEmbeddingModel(model_name=args.embedding_model.strip())
        print(f"Embedder loaded successfully: {args.embedding_model.strip()}")
        job.update_progress(0.75)
    except Exception as e:
        print(f"An error occurred while loading the embedding model: {e}")
        job.set_job_completion_status("failed", "An error occurred while loading the embedding model")
        sys.exit(1)


def generation_from_docs(docs: list):
    try:
        context_config = ContextConstructionConfig(
            embedder=CustomEmbeddingModel(args.embedding_model),
            critic_model=trlab_model,
            chunk_size=args.chunk_size,
            max_contexts_per_document=args.max_contexts_per_document,
            max_context_length=args.max_context_length,
        )
        job.update_progress(1)

        synthesizer = Synthesizer(model=trlab_model)
        print("Synthesizer initialized successfully")
        job.update_progress(1.5)
        synthesizer.generate_goldens_from_docs(
            document_paths=docs,
            context_construction_config=context_config,
            include_expected_output=True,
            max_goldens_per_context=args.max_goldens_per_context,
        )
        job.update_progress(80)
        df = synthesizer.to_pandas()
        return df

    except Exception as e:
        print(f"An error occurred while generating data from docs: {e}")
        traceback.print_exc()
        job.set_job_completion_status("failed", f"An error occurred while generating data from docs: {e}")
        sys.exit(1)


def generate_tflab_dataset(output_file_path: str):
    try:
        api_url = "http://localhost:8338/"
        # Create a new dataset
        params = {"dataset_id": args.run_name, "generated": True}
        response = requests.get(api_url + "data/new", params=params)
        if response.status_code != 200:
            print(f"Error creating a new dataset: {response.json()}")
            job.set_job_completion_status("failed", f"Error creating a new dataset: {response.json()}")
            sys.exit(1)
        job.update_progress(95)
        with open(output_file_path, "rb") as json_file:
            files = {"files": json_file}
            response = requests.post(api_url + "data/fileupload", params=params, files=files)
        if response.status_code != 200:
            print(f"Error uploading the dataset: {response.json()}")
            job.set_job_completion_status("failed", f"Error uploading the dataset: {response.json()}")
            sys.exit(1)
        else:
            print("Dataset uploaded successfully")
            job.update_progress(100)
            job.set_job_completion_status(
                "success",
                f"Data generated successfully as dataset {args.run_name}",
                additional_output_path=output_file_path,
            )
    except Exception as e:
        print(f"An error occurred while generating data: {e}")
        job.set_job_completion_status("failed", f"An error occurred while generating data: {e}")
        sys.exit(1)


def get_docs_list(docs: str) -> List[str]:
    docs_list = docs.split(",")
    documents_dir = os.path.join(os.environ.get("_TFL_WORKSPACE_DIR"), "experiments", args.experiment_name, "documents")
    for i, doc in enumerate(docs_list):
        file_name = doc
        doc_path = os.path.join(documents_dir, doc)
        if os.path.isdir(doc_path):
            docs_list.remove(file_name)
            print(f"Directory found: {doc_path}. Fetching all files in the directory...")
            # Get only first-level files from the given folder. TODO: Change this to handle multilevel folders (using os.walk) when we implement that in the future
            for file in os.listdir(doc_path):
                if not os.path.isdir(os.path.join(doc_path, file)):
                    docs_list.append(os.path.join(doc_path, file))
        else:
            docs_list[i] = os.path.join(documents_dir, doc)
    return docs_list


def run_generation():
    try:
        if not args.docs:
            print("Docs must be provided if generating using docs type.")
            job.set_job_completion_status("failed", "Docs must be provided if generating using docs type.")
            sys.exit(1)
        docs = get_docs_list(args.docs)

        if len(docs) == 0:
            print("No valid documents found. Exiting...")
            job.set_job_completion_status("failed", "No valid documents found.")
            sys.exit(1)
        print(f"Generating data from {len(docs)} documents: {docs}")

        df = generation_from_docs(docs)

        # Rename the column `actual_output` to `output` for consistency
        df.rename(columns={"actual_output": "output"}, inplace=True)
        print("Data generated successfully")
        print("Preview of the generated data:")
        print(df.head())
        output_dir = os.path.join(
            os.environ.get("_TFL_WORKSPACE_DIR"),
            "experiments",
            args.experiment_name,
            "datasets",
        )
        output_file = os.path.join(
            output_dir,
            f"{args.run_name}_{args.job_id}.json",
        )
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        print(f"Saving data json to {output_file}")
        df.to_json(output_file, orient="records", lines=False)
        job.update_progress(90)
        print("Mounting the dataset to the Transformer Lab workspace...")
        generate_tflab_dataset(output_file)

    except Exception as e:
        job.set_job_completion_status("failed", f"An error occurred while generating data: {e}")
        print(f"An error occurred while generating data: {e}")
        traceback.print_exc()
        sys.exit(1)


print("Running generation...")
run_generation()
