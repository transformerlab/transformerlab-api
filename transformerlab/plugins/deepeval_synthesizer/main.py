import argparse
import json
import os
import sys
import traceback
from typing import List

import pandas as pd
import requests
from deepeval.models import DeepEvalBaseEmbeddingModel
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.synthesizer import Evolution, Synthesizer
from deepeval.synthesizer.config import (ContextConstructionConfig,
                                         EvolutionConfig, StylingConfig)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from sentence_transformers import SentenceTransformer

parser = argparse.ArgumentParser(
    description='Run Synthesizer for generating data.')
parser.add_argument('--model_name', default='gpt-j-6b', type=str,
                    help='Model to use for evaluation.')
parser.add_argument('--embedding_model',
                    default='Snowflake/arctic-embed-m', type=str)
parser.add_argument('--dataset_name', default='test', type=str)
parser.add_argument("--model_adapter", default=None, type=str,)
parser.add_argument("--generation_type", default="scratch",
                    type=str, help="Type of generation: scratch, context, docs")
parser.add_argument("--num_goldens", default=5, type=int)
parser.add_argument("--evolution_config", default=None, type=str)
parser.add_argument("--docs", default=None, type=str)
parser.add_argument("--context", default=None, type=str)
parser.add_argument("--input_format", default=None, type=str)
parser.add_argument("--expected_output_format", default=None, type=str)
parser.add_argument("--task", default=None, type=str)
parser.add_argument("--scenario", default=None, type=str)


args, other = parser.parse_known_args()

# Remove redundant instructions from generation type
args.generation_type = args.generation_type.strip().lower().split(" ")[0]

if args.generation_type not in ["scratch", "context", "docs"]:
    print("Generation type must be one of `scratch`, `context`, `docs`")
    sys.exit(1)

print(f"Generation type: {args.generation_type}")


def check_local_server():
    response = requests.get('http://localhost:8338/server/worker_healthz')
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


try:
    # Replace these with real values
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

if args.generation_type == "docs":
    embedder = CustomEmbeddingModel(model_name=args.embedding_model.strip())
    print(f"Embedder loaded successfully: {args.embedding_model.strip()}")


def scratch_generation(styling_config: dict, evolution_config: dict = None):
    # Check if the styling config dictionary has the keys `input_format`, `expected_output_format`, `task`, `scenario`
    if not all(key in styling_config for key in ['input_format', 'expected_output_format', 'task', 'scenario']):
        raise ValueError(
            "Styling config dictionary must have the keys `input_format`, `expected_output_format`, `task`, `scenario`")

    # Check if the evolution config dictionary has the keys `REASONING`, `CONCRETIZING`, `CONSTRAINED`
    if evolution_config is not None and not all(key in evolution_config for key in ['REASONING', 'CONCRETIZING', 'CONSTRAINED']):
        raise ValueError(
            "Evolution config dictionary must have the keys `REASONING`, `CONCRETIZING`, `CONSTRAINED`")
    print("Generating data from scratch...")
    try:
        styling_config = StylingConfig(**styling_config)

        if not evolution_config:
            evolution_config = EvolutionConfig(
                evolutions={
                    Evolution.REASONING: 1/3,
                    Evolution.CONCRETIZING: 1/3,
                    Evolution.CONSTRAINED: 1/3
                },
                num_evolutions=3
            )
        else:
            evolution_config = EvolutionConfig(evolutions={Evolution.REASONING: evolution_config['REASONING'],
                                                           Evolution.CONCRETIZING: evolution_config['CONCRETIZING'],
                                                           Evolution.CONSTRAINED: evolution_config['CONSTRAINED']},
                                               num_evolutions=3)

        synthesizer = Synthesizer(styling_config=styling_config,
                                  model=trlab_model, evolution_config=evolution_config)

        synthesizer.generate_goldens_from_scratch(num_goldens=args.num_goldens)

        df = synthesizer.to_pandas()
        return df

    except Exception as e:
        print(f"An error occurred while generating data from scratch: {e}")
        traceback.print_exc()
        sys.exit(1)


def context_generation(context: str):
    print("Splitting context into sentences...")
    # Break the context into sentences
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""], chunk_size=256, chunk_overlap=0)
    sentences = splitter.split_text(context)
    sentences = [[sentence] for sentence in sentences]
    print(
        f"Number of sentences in the context: {type(sentences)}: {sentences}")
    # Generate goldens from contexts
    print("Generating data from contexts...")
    try:
        synthesizer = Synthesizer(model=trlab_model)
        synthesizer.generate_goldens_from_contexts(
            contexts=sentences, include_expected_output=True)
    except Exception as e:
        print(f"An error occurred while generating data from context: {e}")
        traceback.print_exc()
        sys.exit(1)
    # Convert the generated data to a pandas dataframe
    df = synthesizer.to_pandas()
    return df


def generation_from_docs(docs: list):
    try:
        context_config = ContextConstructionConfig(
            embedder=CustomEmbeddingModel(args.embedding_model),
            critic_model=trlab_model,
            chunk_size=256,
        )

        print(f"Context Construction Config: {context_config}")
        print(f"DOCS: {docs}")

        synthesizer = Synthesizer(model=trlab_model)
        synthesizer.generate_goldens_from_docs(
            document_paths=docs,
            context_construction_config=context_config,
            include_expected_output=True,
        )
        df = synthesizer.to_pandas()
    except Exception as e:
        print(f"An error occurred while generating data from docs: {e}")
        traceback.print_exc()
        sys.exit(1)

    return df


def run_generation():
    try:
        if args.generation_type == 'scratch':
            # Check if the styling config and evolution config are provided
            if not args.input_format or not args.expected_output_format or not args.task or not args.scenario:
                print(
                    "Input format, expected output format, task and scenario must be provided for scratch generation.")
                sys.exit(1)
            else:
                styling_config = {
                    "input_format": args.input_format,
                    "expected_output_format": args.expected_output_format,
                    "task": args.task,
                    "scenario": args.scenario
                }

            # Check if the evolution config is provided
            if args.evolution_config:
                evolution_config = json.loads(args.evolution_config)
            else:
                evolution_config = None

            df = scratch_generation(styling_config, evolution_config)

        elif args.generation_type == 'context':
            if not args.context or len(args.context.strip()) <= 1:
                print("Context must be provided if generating using context type.")
                sys.exit(1)
            df = context_generation(args.context)

        elif args.generation_type == 'docs':
            if not args.docs:
                print("Docs must be provided if generating using docs type.")
                sys.exit(1)
            docs = args.docs.split(",")
            # Check if the path provided in docs has the files present
            for doc in docs:
                if not os.path.exists(doc):
                    print(f"File {doc} not found. Skipping...")
                    docs.remove(doc)
            if len(docs) == 0:
                print("No valid documents found. Exiting...")
                sys.exit(1)
            print(f"Generating data from {len(docs)} documents: {docs}")

            df = generation_from_docs(docs)

        else:
            print("Invalid generation type. Exiting...")
            sys.exit(1)

        # Rename the column `actual_output` to `output` for consistency
        df.rename(columns={"actual_output": "output"}, inplace=True)
        print("Data generated successfully")
        print("Preview of the generated data:")
        print(df.head())
        output_dir = os.path.join(
            os.environ.get('_TFL_WORKSPACE_DIR'), "datasets", f"{args.generation_type}_{args.dataset_name}.csv")
        df.to_csv(output_dir, index=False)
        print(f"Data saved to {output_dir}")
    except Exception as e:
        print(f"An error occurred while generating data: {e}")
        print(traceback.print_exc())
        sys.exit(1)


print("Running generation...")
run_generation()
