import argparse
import json
import os
import sys
import traceback
from typing import List
import transformerlab.plugin
import requests
import instructor
from anthropic import Anthropic
from deepeval.models import DeepEvalBaseEmbeddingModel
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.synthesizer import Evolution, Synthesizer
from deepeval.synthesizer.config import ContextConstructionConfig, EvolutionConfig, StylingConfig
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel

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
parser.add_argument("--generation_model", default="claude-3.5-sonnet", type=str)
parser.add_argument("--generation_type", default="scratch", type=str, help="Type of generation: scratch, context, docs")
parser.add_argument("--num_goldens", default=5, type=int)
parser.add_argument("--evolution_config", default=None, type=str)
parser.add_argument("--docs", default=None, type=str)
parser.add_argument("--context", default=None, type=str)
parser.add_argument("--input_format", default=None, type=str)
parser.add_argument("--expected_output_format", default=None, type=str)
parser.add_argument("--task", default=None, type=str)
parser.add_argument("--scenario", default=None, type=str)
parser.add_argument("--experiment_name", default="test", type=str)
parser.add_argument("--job_id", default=None, type=str)


args, other = parser.parse_known_args()

# Remove redundant instructions from generation type
args.generation_type = args.generation_type.strip().lower().split(" ")[0]


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


check_local_server()


class CustomEmbeddingModel(DeepEvalBaseEmbeddingModel):
    def __init__(self, model_name: str = "Snowflake/arctic-embed-m"):
        self.model_name = model_name
        self.model = self.load_model()

    def load_model(self):
        print(f"Loading Embedding Model... : {self.model_name.strip()}")
        return SentenceTransformer(self.model_name.strip())

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


def scratch_generation(styling_config: dict, evolution_config: dict = None):
    # Check if the styling config dictionary has the keys `input_format`, `expected_output_format`, `task`, `scenario`
    if not all(key in styling_config for key in ["input_format", "expected_output_format", "task", "scenario"]):
        job.set_job_completion_status(
            "failed",
            "Styling config dictionary must have the keys `input_format`, `expected_output_format`, `task`, `scenario`",
        )
        raise ValueError(
            "Styling config dictionary must have the keys `input_format`, `expected_output_format`, `task`, `scenario`"
        )

    # Check if the evolution config dictionary has the keys `REASONING`, `CONCRETIZING`, `CONSTRAINED`
    if evolution_config is not None and not all(
        key in evolution_config for key in ["REASONING", "CONCRETIZING", "CONSTRAINED"]
    ):
        job.set_job_completion_status(
            "failed", "Evolution config dictionary must have the keys `REASONING`, `CONCRETIZING`, `CONSTRAINED`"
        )
        raise ValueError("Evolution config dictionary must have the keys `REASONING`, `CONCRETIZING`, `CONSTRAINED`")
    print("Generating data from scratch...")
    job.update_progress(1)
    try:
        styling_config = StylingConfig(**styling_config)

        if not evolution_config:
            evolution_config = EvolutionConfig(
                evolutions={Evolution.REASONING: 1 / 3, Evolution.CONCRETIZING: 1 / 3, Evolution.CONSTRAINED: 1 / 3},
                num_evolutions=3,
            )
        else:
            evolution_config = EvolutionConfig(
                evolutions={
                    Evolution.REASONING: evolution_config["REASONING"],
                    Evolution.CONCRETIZING: evolution_config["CONCRETIZING"],
                    Evolution.CONSTRAINED: evolution_config["CONSTRAINED"],
                },
                num_evolutions=3,
            )

        synthesizer = Synthesizer(styling_config=styling_config, model=trlab_model, evolution_config=evolution_config)
        job.update_progress(1.5)
        print("Synthesizer initialized successfully")

        synthesizer.generate_goldens_from_scratch(num_goldens=args.num_goldens)
        job.update_progress(80)
        df = synthesizer.to_pandas()
        return df

    except Exception as e:
        print(f"An error occurred while generating data from scratch: {e}")
        job.set_job_completion_status("failed", f"An error occurred while generating data from scratch: {e}")
        traceback.print_exc()
        sys.exit(1)


def context_generation(context: str):
    print("Splitting context into sentences...")
    # Break the context into sentences
    splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", ". ", " ", ""], chunk_size=256, chunk_overlap=0)
    sentences = splitter.split_text(context)
    sentences = [[sentence] for sentence in sentences]
    print(f"Number of sentences in the context: {type(sentences)}: {sentences}")
    job.update_progress(1)
    # Generate goldens from contexts
    print("Generating data from contexts...")
    try:
        synthesizer = Synthesizer(model=trlab_model)
        print(f"Synthesizer initialized successfully")
        job.update_progress(1.5)
        synthesizer.generate_goldens_from_contexts(contexts=sentences, include_expected_output=True)
        job.update_progress(80)
        # Convert the generated data to a pandas dataframe
        df = synthesizer.to_pandas()
        return df

    except Exception as e:
        print(f"An error occurred while generating data from context: {e}")
        job.set_job_completion_status("failed", f"An error occurred while generating data from context: {e}")
        traceback.print_exc()
        sys.exit(1)


def generation_from_docs(docs: list):
    try:
        context_config = ContextConstructionConfig(
            embedder=CustomEmbeddingModel(args.embedding_model),
            critic_model=trlab_model,
            chunk_size=256,
        )
        job.update_progress(1)

        synthesizer = Synthesizer(model=trlab_model)
        print(f"Synthesizer initialized successfully")
        job.update_progress(1.5)
        synthesizer.generate_goldens_from_docs(
            document_paths=docs,
            context_construction_config=context_config,
            include_expected_output=True,
        )
        job.update_progress(80)
        df = synthesizer.to_pandas()
        return df

    except Exception as e:
        print(f"An error occurred while generating data from docs: {e}")
        traceback.print_exc()
        sys.exit(1)


def generate_tflab_dataset(output_file_path: str):
    try:
        api_url = "http://localhost:8338/"
        # Create a new dataset
        params = {"dataset_id": args.run_name}
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
            job.set_job_completion_status("success", f"Data generated successfully as dataset {args.run_name}")
    except Exception as e:
        print(f"An error occurred while generating data: {e}")
        job.set_job_completion_status("failed", f"An error occurred while generating data: {e}")
        sys.exit(1)


def run_generation():
    try:
        if args.generation_type == "scratch":
            # Check if the styling config and evolution config are provided
            if not args.input_format or not args.expected_output_format or not args.task or not args.scenario:
                print(
                    "Input format, expected output format, task and scenario must be provided for scratch generation."
                )
                sys.exit(1)
            else:
                styling_config = {
                    "input_format": args.input_format,
                    "expected_output_format": args.expected_output_format,
                    "task": args.task,
                    "scenario": args.scenario,
                }

            # Check if the evolution config is provided
            if args.evolution_config:
                evolution_config = json.loads(args.evolution_config)
            else:
                evolution_config = None

            df = scratch_generation(styling_config, evolution_config)

        elif args.generation_type == "context":
            if not args.context or len(args.context.strip()) <= 1:
                print("Context must be provided if generating using context type.")
                job.set_job_completion_status("failed", "Context must be provided if generating using context type.")
                sys.exit(1)
            df = context_generation(args.context)

        elif args.generation_type == "docs":
            if not args.docs:
                print("Docs must be provided if generating using docs type.")
                job.set_job_completion_status("failed", "Docs must be provided if generating using docs type.")
                sys.exit(1)
            docs = args.docs.split(",")
            # Check if the path provided in docs has the files present
            for doc in docs:
                if not os.path.exists(doc):
                    print(f"File {doc} not found. Skipping...")
                    docs.remove(doc)
            if len(docs) == 0:
                print("No valid documents found. Exiting...")
                job.set_job_completion_status("failed", "No valid documents found.")
                sys.exit(1)
            print(f"Generating data from {len(docs)} documents: {docs}")

            df = generation_from_docs(docs)

        else:
            print("Invalid generation type. Exiting...")
            job.set_job_completion_status("failed", "Invalid generation type.")
            sys.exit(1)

        # Rename the column `actual_output` to `output` for consistency
        df.rename(columns={"actual_output": "output"}, inplace=True)
        print("Data generated successfully")
        print("Preview of the generated data:")
        print(df.head())
        output_file = os.path.join(
            os.environ.get("_TFL_WORKSPACE_DIR"),
            "experiments",
            args.experiment_name,
            "datasets",
            args.generation_type,
            f"{args.run_name}.json",
        )
        print(f"Saving data json to {output_file}")
        df.to_json(output_file, orient="records", lines=False)
        job.update_progress(90)
        print("Mounting the dataset to the TransformerLab workspace...")
        generate_tflab_dataset(output_file)

    except Exception as e:
        job.set_job_completion_status("failed", f"An error occurred while generating data: {e}")
        print(f"An error occurred while generating data: {e}")
        traceback.print_exc()
        sys.exit(1)


print("Running generation...")
run_generation()
