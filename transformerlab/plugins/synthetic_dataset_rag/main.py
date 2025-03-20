import argparse
import json
import os
import random
import sys
import traceback
from typing import List

import fitz
import requests
from anthropic import Anthropic
from langchain.docstore.document import Document as LangchainDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from openai import OpenAI
from tqdm.auto import tqdm

import transformerlab.plugin

# Set up command line arguments
parser = argparse.ArgumentParser(description="Generate synthetic QA pairs for RAG evaluation.")
parser.add_argument("--run_name", default="rag_eval", type=str)
parser.add_argument("--model_name", default="gpt-j-6b", type=str, help="Model to use for generation.")
parser.add_argument("--dataset_name", default="rag_eval", type=str)
parser.add_argument("--model_adapter", default=None, type=str)
parser.add_argument("--generation_model", default="claude-3-7-sonnet-latest", type=str)
parser.add_argument("--docs", default=None, type=str)
parser.add_argument("--experiment_name", default="test", type=str)
parser.add_argument("--job_id", default=None, type=str)
parser.add_argument("--chunk_size", default=2000, type=int)
parser.add_argument("--chunk_overlap", default=200, type=int)
parser.add_argument("--n_generations", default=10, type=int, help="Number of QA pairs to generate")

args, other = parser.parse_known_args()

# Initialize job tracking
if args.job_id:
    job = transformerlab.plugin.Job(args.job_id)
    job.update_progress(0)
else:
    print("Job ID not provided.")
    sys.exit(1)

print(f"Model Name: {args.generation_model}")


def check_local_server():
    response = requests.get("http://localhost:8338/server/worker_healthz")
    print(response.json())
    if response.status_code != 200 or not isinstance(response.json(), list) or len(response.json()) == 0:
        print("Local Model Server is not running. Please start it before running the evaluation.")
        sys.exit(1)


class CustomCommercialModel:
    def __init__(self, model_type="claude", model_name="claude-3-7-sonnet-latest"):
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

    def generate(self, prompt: str) -> str:
        client = self.load_model()

        if self.model_type == "claude":
            response = client.messages.create(
                model=self.model_name, max_tokens=1000, messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        else:  # openai or custom
            response = client.chat.completions.create(
                model=self.model_name, messages=[{"role": "user", "content": prompt}], max_tokens=1000
            )
            return response.choices[0].message.content


# Generating custom TRLAB model
class TRLAB_MODEL:
    def __init__(self, model):
        self.model = model

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        return chat_model.invoke(prompt).content


def get_docs_list(docs: str) -> List[dict]:
    """
    Convert document paths to a list of document data suitable for LangChain
    Supports text files, PDFs, and other document formats
    """
    docs_list = docs.split(",")
    documents_dir = os.path.join(os.environ.get("_TFL_WORKSPACE_DIR"), "experiments", args.experiment_name, "documents")

    result_docs = []

    for i, doc in enumerate(docs_list):
        doc_path = os.path.join(documents_dir, doc)

        if os.path.isdir(doc_path):
            print(f"Directory found: {doc_path}. Fetching all files in the directory...")
            for file in os.listdir(doc_path):
                file_full_path = os.path.join(doc_path, file)
                if not os.path.isdir(file_full_path):
                    try:
                        # Process based on file extension
                        if file_full_path.lower().endswith(".pdf"):
                            # Handle PDF files with PyMuPDF
                            content = extract_text_from_pdf(file_full_path)
                            result_docs.append({"text": content, "source": file})
                        else:
                            # Handle as text file
                            with open(file_full_path, "r", encoding="utf-8") as f:
                                content = f.read()
                                result_docs.append({"text": content, "source": file})
                    except Exception as e:
                        print(f"Error reading file {file_full_path}: {e}")
        else:
            full_path = os.path.join(documents_dir, doc)
            try:
                # Process based on file extension
                if full_path.lower().endswith(".pdf"):
                    # Handle PDF files with PyMuPDF
                    content = extract_text_from_pdf(full_path)
                    result_docs.append({"text": content, "source": doc})
                else:
                    # Handle as text file
                    with open(full_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        result_docs.append({"text": content, "source": doc})
            except Exception as e:
                print(f"Error reading file {full_path}: {e}")

    return result_docs


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from PDF files using PyMuPDF"""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text += page.get_text()
        return text
    except Exception as e:
        print(f"Error extracting text from PDF {pdf_path}: {e}")
        return ""


def generate_tflab_dataset(output_file_path: str, dataset_id: str = args.run_name):
    """Upload the generated dataset to TransformerLab"""
    try:
        api_url = "http://localhost:8338/"
        # Create a new dataset
        params = {"dataset_id": dataset_id, "generated": True}
        response = requests.get(api_url + "data/new", params=params)
        if response.status_code != 200:
            print(f"Error creating a new dataset: {response.json()}")
            job.set_job_completion_status("failed", f"Error creating a new dataset: {response.json()}")
            sys.exit(1)
        with open(output_file_path, "rb") as json_file:
            files = {"files": json_file}
            response = requests.post(api_url + "data/fileupload", params=params, files=files)

        if response.status_code != 200:
            print(f"Error uploading the dataset: {response.json()}")
            job.set_job_completion_status("failed", f"Error uploading the dataset: {response.json()}")
            sys.exit(1)

        print("Dataset uploaded successfully")

    except Exception as e:
        print(f"An error occurred while generating data: {e}")
        job.set_job_completion_status("failed", f"An error occurred while generating data: {e}")
        sys.exit(1)


def run_generation():
    try:
        if not args.docs:
            print("Docs must be provided for generating QA pairs.")
            job.set_job_completion_status("failed", "Docs must be provided for generating QA pairs.")
            sys.exit(1)

        # Initialize model
        if "local" not in args.generation_model.lower():
            if "openai" in args.generation_model.lower() or "gpt" in args.generation_model.lower():
                model = CustomCommercialModel("openai", args.generation_model)
            elif "claude" in args.generation_model.lower() or "anthropic" in args.generation_model.lower():
                model = CustomCommercialModel("claude", args.generation_model)
            elif "custom" in args.generation_model.lower():
                model = CustomCommercialModel("custom", "")
        else:
            check_local_server()
            custom_model = ChatOpenAI(
                api_key="dummy",
                base_url="http://localhost:8338/v1",
                model=args.model_name,
            )
            model = TRLAB_MODEL(model=custom_model)

        print("Model loaded successfully")
        job.update_progress(25)

        # Load and process documents
        doc_data = get_docs_list(args.docs)
        if len(doc_data) == 0:
            print("No valid documents found. Exiting...")
            job.set_job_completion_status("failed", "No valid documents found.")
            sys.exit(1)

        print(f"Processing {len(doc_data)} documents")

        # Convert to LangChain documents
        langchain_docs = [
            LangchainDocument(page_content=doc["text"], metadata={"source": doc["source"]}) for doc in doc_data
        ]

        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            add_start_index=True,
            separators=["\n\n", "\n", ".", " ", ""],
        )

        docs_processed = []
        for doc in langchain_docs:
            docs_processed += text_splitter.split_documents([doc])

        job.update_progress(50)
        print(f"Split into {len(docs_processed)} chunks. Generating QA pairs...")

        # QA generation prompt
        QA_generation_prompt = """
        Your task is to write a factoid question and an answer given a context.
        Your factoid question should be answerable with a specific, concise piece of factual information from the context.
        Your factoid question should be formulated in the same style as questions users could ask in a search engine.
        This means that your factoid question MUST NOT mention something like "according to the passage" or "context".

        Provide your answer as follows:

        Output:::
        Factoid question: (your factoid question)
        Answer: (your answer to the factoid question)

        Now here is the context.

        Context: {context}
        Output:::"""

        # Set the number of generations
        n_samples = min(args.n_generations, len(docs_processed))

        outputs = []
        for i, sampled_context in enumerate(tqdm(random.sample(docs_processed, n_samples))):
            try:
                # Generate QA couple
                output_QA_couple = model.generate(QA_generation_prompt.format(context=sampled_context.page_content))

                question = output_QA_couple.split("Factoid question: ")[-1].split("Answer: ")[0].strip()
                answer = output_QA_couple.split("Answer: ")[-1].strip()

                assert len(answer) < 300, "Answer is too long"

                outputs.append(
                    {
                        "context": sampled_context.page_content,
                        "input": question,
                        "expected_output": answer,
                        "source_doc": sampled_context.metadata["source"],
                    }
                )

                # Update progress
                progress = 50 + (i + 1) / n_samples * 40
                job.update_progress(progress)

            except Exception as e:
                print(f"Error processing chunk: {e}")
                continue

        # Save the generated data
        output_dir = os.path.join(
            os.environ.get("_TFL_WORKSPACE_DIR"),
            "experiments",
            args.experiment_name,
            "datasets",
        )
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        output_file = os.path.join(
            output_dir,
            f"{args.run_name}_{args.job_id}.json",
        )

        # Save as JSON
        print(f"Saving {len(outputs)} QA pairs to {output_file}")
        with open(output_file, "w") as f:
            json.dump(outputs, f, indent=2)

        job.update_progress(90)

        # Upload to TransformerLab
        print("Mounting the dataset to the Transformer Lab workspace...")
        generate_tflab_dataset(output_file)

        job.update_progress(100)
        job.set_job_completion_status(
            "success",
            f"QA dataset generated successfully as dataset {args.run_name}",
            additional_output_path=output_file,
        )

    except Exception as e:
        job.set_job_completion_status("failed", f"An error occurred while generating data: {str(e)}")
        print(f"An error occurred while generating data: {e}")
        traceback.print_exc()
        sys.exit(1)


print("Starting RAG evaluation dataset generation...")
run_generation()
