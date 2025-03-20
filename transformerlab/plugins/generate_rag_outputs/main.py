import argparse
import asyncio
import json
import os
import sys
import traceback
from datetime import datetime
from typing import Any, Dict, List

import requests
from datasets import load_dataset

import transformerlab.plugin

try:
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run RAG evaluation on dataset")
    parser.add_argument("--run_name", default="rag_eval_results", type=str)
    parser.add_argument("--dataset_name", default=None, type=str, help="Dataset to evaluate")
    parser.add_argument("--experiment_name", default="test", type=str)
    parser.add_argument("--job_id", default=None, type=str)
    parser.add_argument("--model_name", default=None, type=str, help="Model to use for RAG")
    parser.add_argument("--input_field", default="input", type=str, help="Field in dataset containing queries")
    parser.add_argument("--response_mode", default="compact", type=str, help="Field in dataset containing queries")
    parser.add_argument("--number_of_search_results", default="2", type=str, help="Number of search results to return")
    parser.add_argument("--temperature", default="0.7", type=str, help="Temperature for sampling")
    parser.add_argument("--context_window", default="4096", type=str, help="Context window size")
    parser.add_argument("--num_output", default="256", type=str, help="Output Length")
    parser.add_argument("--chunk_size", default="512", type=str, help="Chunk size")
    parser.add_argument("--chunk_overlap", default="100", type=str, help="Chunk overlap")
    parser.add_argument("--use_reranker", default=None, type=bool, help="Use reranker")
    parser.add_argument(
        "--reranker_model", default="cross-encoder/ms-marco-MiniLM-L-6-v2", type=str, help="Reranker model"
    )
    parser.add_argument("--reranker_top_n", default="20", type=str, help="Reranker top n")
except Exception as e:
    print(f"Error parsing command line arguments: {str(e)}")
    sys.exit(1)


# parser.add_argument("--rag_settings", default="{}", type=str, help="Additional RAG settings as JSON string")

args, other = parser.parse_known_args()

# Initialize job tracking
if args.job_id:
    job = transformerlab.plugin.Job(args.job_id)
    job.update_progress(0)
else:
    print("Job ID not provided.")
    sys.exit(1)


async def run_rag_query(experiment_id, rag_settings, query: str) -> Dict[str, Any]:
    """Run a RAG query using the configured RAG engine"""
    try:
        # Construct the API URL
        api_url = f"http://localhost:8338/experiment/{experiment_id}/rag/query"

        # Prepare parameters
        params = {"experimentId": experiment_id, "query": query, "settings": rag_settings}

        # Make the request
        response = requests.get(api_url, params=params)

        if response.status_code != 200:
            print(f"RAG query failed for query: {query}")
            print(f"Error: {response.text}")
            return {
                "query": query,
                "answer": "Error: RAG query failed",
                "context": [],
                "sources": [],
                "error": response.text,
            }

        # Parse the response
        try:
            result = response.json() if response.headers.get("content-type") == "application/json" else response.text
        except Exception:
            result = response.text

        # Extract relevant information (format may vary by RAG engine)
        if isinstance(result, dict):
            context_list = []
            scores_list = []
            for context in result.get("context", []):
                context_list.append(context.split("Text: ")[1].split("\nScore")[0].strip())
                scores_list.append(context.split("Score: ")[1].split("\n")[0].strip())
            return {
                "query": query,
                "answer": result.get("response", ""),
                "context": context_list,
                "scores": scores_list,
                "raw_response": result,
                "prompt": result.get("template", ""),
            }
        else:
            return {"query": query, "answer": result, "context": [], "sources": [], "raw_response": result}

    except Exception as e:
        print(f"Error running RAG query: {str(e)}")
        return {"query": query, "answer": f"Error: {str(e)}", "context": [], "sources": [], "error": str(e)}


def get_tflab_dataset():
    try:
        dataset_target = transformerlab.plugin.get_dataset_path(args.dataset_name)
    except Exception as e:
        job.add_to_job_data("end_time", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        job.set_job_completion_status("failed", "Failure to get dataset")
        raise e
    dataset = {}
    dataset_types = ["train"]
    for dataset_type in dataset_types:
        try:
            dataset[dataset_type] = load_dataset(dataset_target, split=dataset_type, trust_remote_code=True)

        except Exception as e:
            job.add_to_job_data("end_time", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            job.set_job_completion_status("failed", "Failure to load dataset")
            raise e
    # Convert the dataset to a pandas dataframe
    df = dataset["train"].to_pandas()
    return df


async def process_dataset(experiment_id, rag_settings, dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Process each item in the dataset with RAG"""
    results = []

    print(f"Processing {len(dataset)} queries with RAG...")

    for i, row in dataset.iterrows():
        # Update progress
        progress = int((i / len(dataset)) * 90)
        job.update_progress(progress)

        # Extract the query from the specified field
        # query = item.get(args.input_field, "")
        query = row[args.input_field]
        if not query:
            print(f"Warning: No query found in item {i} using field '{args.input_field}'")
            query = ""
            continue

        # Run RAG on the query
        rag_result = await run_rag_query(experiment_id, rag_settings, query)

        # Combine original item with RAG results
        combined_result = row.to_dict()
        combined_result.update(
            {
                "output": rag_result["answer"],
                "context": rag_result.get("context", []),
                "rag_scores": rag_result.get("scores", []),
                "rag_prompt": rag_result.get("prompt", ""),
                "rag_raw_response": rag_result.get("raw_response", ""),
            }
        )

        results.append(combined_result)

    return results


def generate_tflab_dataset(output_file_path: str, dataset_id: str = f"{args.run_name}_{args.job_id}"):
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


def check_local_server():
    response = requests.get("http://localhost:8338/server/worker_healthz")
    if response.status_code != 200 or not isinstance(response.json(), list) or len(response.json()) == 0:
        print("Local Model Server is not running. Please start it before running the evaluation.")
        sys.exit(1)


async def run_evaluation():
    try:
        check_local_server()
        # Validate required arguments
        if not args.dataset_name:
            raise ValueError("Dataset name is required")

        # Configure experiment with the specified RAG engine
        experiment_config, experiment_id = transformerlab.plugin.get_experiment_config(args.experiment_name)

        if experiment_config:
            plugin = experiment_config.get("rag_engine")
            if plugin is None or plugin == "":
                print(
                    "No RAG engine has been assigned to this experiment. Please install a RAG plugin from the Plugins Tab."
                )
                job.set_job_completion_status("failed", "No RAG engine has been assigned to this experiment.")
                sys.exit(1)
            rag_settings = experiment_config.get("rag_engine_settings", {})
            if args.use_reranker is None or args.use_reranker == "":
                args.use_reranker = False
            # if not rag_settings or rag_settings == {}:
            rag_settings = {
                "response_mode": args.response_mode,
                "number_of_search_results": args.number_of_search_results,
                "temperature": args.temperature,
                "context_window": args.context_window,
                "num_output": args.num_output,
                "chunk_size": args.chunk_size,
                "chunk_overlap": args.chunk_overlap,
                "use_reranker": args.use_reranker,
                "reranker_model": args.reranker_model,
                "reranker_top_n": args.reranker_top_n,
            }
            print(f"RAG settings: {rag_settings}")
            rag_settings = json.dumps(rag_settings)

        # Load the input dataset
        print(f"Loading dataset '{args.dataset_name}'...")
        dataset = get_tflab_dataset()
        job.update_progress(10)

        # Process the dataset
        print("Processing dataset with RAG...")
        results = await process_dataset(experiment_id, rag_settings, dataset)
        job.update_progress(90)

        # Save the results
        output_dir = os.path.join(os.environ.get("_TFL_WORKSPACE_DIR"), "experiments", args.experiment_name, "datasets")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        output_file = os.path.join(output_dir, f"{args.run_name}_{args.job_id}.json")

        # Save as JSON
        print(f"Saving {len(results)} results to {output_file}")
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        # Upload to TransformerLab
        print("Mounting the dataset to the TransformerLab workspace...")
        generate_tflab_dataset(output_file)

        job.update_progress(100)
        job.set_job_completion_status(
            "success",
            f"RAG evaluation completed successfully. Results saved as dataset '{args.run_name}_{args.job_id}'.",
            additional_output_path=output_file,
        )

    except Exception as e:
        traceback.print_exc()
        job.set_job_completion_status("failed", f"An error occurred during evaluation: {str(e)}")
        sys.exit(1)


# Run the evaluation
print("Starting RAG dataset evaluation...")
asyncio.run(run_evaluation())
