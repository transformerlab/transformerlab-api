"""
Synthetic Dataset Generator Plugin for TransformerLab

This plugin integrates the synthetic-data-kit CLI tool by Meta into the TransformerLab platform.
It supports local model-based generation of question-answer pairs, summaries, and chain-of-thought content
using structured documents like PDFs, DOCX, and others. Outputs are automatically ingested, curated,
converted to JSONL/Alpaca/ChatML, and uploaded.

Main steps:
- Ingest input documents into plain text
- Generate synthetic data using vLLM server
- Curate outputs based on rating threshold
- Export into desired format and return final dataset
"""

import os
import sys
import subprocess
import uuid
import yaml
import json
import time
import shutil
import pandas as pd
from pathlib import Path

from transformerlab.sdk.v1.generate import tlab_gen


import os
import sys
from pathlib import Path


def get_synthetic_kit_cli_path():
    # sys.executable points to: /path/to/venv/bin/python
    venv_bin_dir = Path(sys.executable).parent
    cli_path = venv_bin_dir / "synthetic-data-kit"

    if cli_path.exists():
        return str(cli_path)
    else:
        raise FileNotFoundError(f"'synthetic-data-kit' CLI not found at {cli_path}")


def is_server_alive(sys_path, api_base):
    try:
        subprocess.run(
            [sys_path, "system-check", f"--api-base={api_base}"],
            check=True,
        )
        return True
    except subprocess.CalledProcessError as e:
        print("STDOUT:\n", e.stdout)
        print("STDERR:\n", e.stderr)
        return False


def launch_vllm(model_path, port):
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            model_path,
            "--port",
            str(port),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def wait_for_server(api_base, timeout=30):
    for _ in range(timeout):
        if is_server_alive(api_base):
            return True
        time.sleep(1)
    return False


def stop_process(proc):
    if proc:
        try:
            proc.terminate()
            proc.wait(timeout=10)
        except Exception:
            proc.kill()


@tlab_gen.job_wrapper()
def run_generation():
    """
    Main entry point for TransformerLab plugin job.

    Parses documents, generates content using synthetic-data-kit, curates and formats the output,
    and returns a list of final structured rows to be saved as a dataset.

    Saves the final dataset using tlab_gen.save_generated_dataset() which makes it accessible via the UI.
    """
    sys_path = get_synthetic_kit_cli_path()
    docs_str = tlab_gen.params.docs
    generation_type = tlab_gen.params.get("task_type", "qa")
    num_pairs = tlab_gen.params.get("num_pairs", 25)
    threshold = tlab_gen.params.get("curation_threshold", 7.0)
    output_format = tlab_gen.params.get("output_format", "jsonl")
    prompt_template = tlab_gen.params.get("prompt_template", "")
    model_name = tlab_gen.params.get("model_name", "meta-llama/Llama-3-8B-Instruct")
    model_path = tlab_gen.params.get("model_path", model_name)
    api_base = tlab_gen.params.get("vllm_api_base", "http://localhost:8000/v1")
    port = int(api_base.rsplit(":", 1)[-1].rstrip("/v1"))
    workspace = os.environ["_TFL_WORKSPACE_DIR"]
    experiment = tlab_gen.params.experiment_name
    documents_dir = os.path.join(workspace, "experiments", experiment, "documents")
    doc_filenames = [d.strip() for d in docs_str.split(",") if d.strip()]
    full_paths = [os.path.join(documents_dir, name) for name in doc_filenames]
    tmp_dir = "transformerlab/plugins/synthetic_dataset_kit/data"

    # Prompt selector based on generation_type
    DEFAULT_PROMPTS = {
        "summary": "Summarize this document in 3-5 sentences, focusing on the main topic and key concepts.",
        "qa_generation": """               
            Create {num_pairs} question-answer pairs from this text for LLM training.

            Rules:
            1. Questions must be about important facts in the text
            2. Answers must be directly supported by the text
            3. Return JSON format only:
            
            [
            {{
                "question": "Question 1?",
                "answer": "Answer 1."
            }},
            {{
                "question": "Question 2?",
                "answer": "Answer 2."
            }}
            ]
            
            Text:
            {text}
        """,
        "qa_rating": """
            Rate each question-answer pair on a scale from 1-10, based on:
            - Accuracy (0-3): factual correctness
            - Relevance (0-2): relevance to content
            - Clarity (0-2): clear language
            - Usefulness (0-3): value for model learning
            
            YOU MUST RETURN A VALID JSON OBJECT OR ARRAY WITH THIS EXACT SCHEMA:
            {{
            "question": "Exact question text",
            "answer": "Exact answer text",
            "rating": 8
            }}
            
            OR FOR MULTIPLE PAIRS:
            [
            {{"question": "Q1", "answer": "A1", "rating": 8}},
            {{"question": "Q2", "answer": "A2", "rating": 9}}
            ]
            
            *** YOUR RESPONSE MUST BE VALID JSON AND NOTHING ELSE - NO EXPLANATION, NO MARKDOWN ***
            
            QA pairs to rate:
            {pairs}        
        """,
        "cot_generation": """
            Create {num_examples} complex reasoning examples from this text that demonstrate chain-of-thought thinking.
            
            Each example should have:
            1. A challenging question that requires step-by-step reasoning
            2. Detailed reasoning steps that break down the problem
            3. A concise final answer
            
            Return JSON format only:
            
            [
            {{
                "question": "Complex question about the text?",
                "reasoning": "Step 1: First, I need to consider...\nStep 2: Then, I analyze...\nStep 3: Finally, I can conclude...",
                "answer": "Final answer based on the reasoning."
            }},
            {{
                "question": "Another complex question?",
                "reasoning": "Step 1: First, I'll analyze...\nStep 2: Next, I need to determine...\nStep 3: Based on this analysis...",
                "answer": "Final answer drawn from the reasoning."
            }}
            ]
            
            Text:
            {text}
        """,
        "cot_enhancement": """
            You are an expert reasoning assistant. Your task is to enhance the given conversations by adding chain-of-thought reasoning.
            
            For each conversation, add detailed step-by-step reasoning to the assistant's responses while preserving the original answer.
            
            {include_simple_steps} = Whether to add reasoning to simple responses too. If false, only add reasoning to complex responses.
            
            Return the enhanced conversations as a JSON array matching this format:
            [
            [
                {{"role": "system", "content": "System message"}},
                {{"role": "user", "content": "User question"}},
                {{"role": "assistant", "content": "Let me think through this step by step:\n\n1. First, I need to consider...\n2. Then...\n\nTherefore, [original answer]"}}
            ],
            [
                {{"role": "system", "content": "System message"}},
                {{"role": "user", "content": "Another user question"}},
                {{"role": "assistant", "content": "Let me work through this:\n\n1. I'll start by...\n2. Next...\n\nIn conclusion, [original answer]"}}
            ]
            ]
            
            Original conversations:
            {conversations}  
            """,
    }
    prompt_lookup = {"qa": "qa_generation", "cot": "cot_generation", "summary": "summary"}
    prompt_key = prompt_lookup.get(generation_type, "qa_generation")
    if prompt_template != "":
        for k in DEFAULT_PROMPTS.keys():
            if prompt_key == k:
                DEFAULT_PROMPTS[k] = prompt_template

    sub_folder = str(uuid.uuid4().hex)
    paths = {
        "input": {
            "pdf": f"transformerlab/plugins/synthetic_dataset_kit/{sub_folder}/data/pdf/",
            "html": f"transformerlab/plugins/synthetic_dataset_kit/{sub_folder}/data/html/",
            "youtube": f"transformerlab/plugins/synthetic_dataset_kit/{sub_folder}/data/youtube/",
            "docx": f"transformerlab/plugins/synthetic_dataset_kit/{sub_folder}/data/docx/",
            "ppt": f"transformerlab/plugins/synthetic_dataset_kit/{sub_folder}/data/ppt/",
            "txt": f"transformerlab/plugins/synthetic_dataset_kit/{sub_folder}/data/txt/",
        },
        "output": {
            "parsed": f"transformerlab/plugins/synthetic_dataset_kit/data/{sub_folder}/output/",
            "generated": f"transformerlab/plugins/synthetic_dataset_kit/data/{sub_folder}/generated/",
            "cleaned": f"transformerlab/plugins/synthetic_dataset_kit/data/{sub_folder}/cleaned/",
            "final": f"transformerlab/plugins/synthetic_dataset_kit/data/{sub_folder}/final/",
        },
    }

    # Construct synthetic-data-kit config object dynamically
    # This config controls:
    # - input/output folder mappings
    # - vLLM server configuration
    # - generation + curation parameters
    # - formatting options
    # - LLM prompts

    config = {
        "paths": paths,
        "vllm": {
            "api_base": api_base,
            "port": port,
            "model": model_path,
            "max_retries": 3,
            "retry_delay": 1.0,
        },
        "generation": {
            "num_pairs": num_pairs,
            "temperature": 0.7,
            "chunk_size": 3000,
            "overlap": 300,
        },
        "curate": {"threshold": threshold},
        "format": {
            "default": output_format,
            "include_metadata": True,
            "pretty_json": True,
        },
        "prompts": DEFAULT_PROMPTS,
    }

    # Ensure temporary config directory exists
    os.makedirs(tmp_dir, exist_ok=True)
    config_path = os.path.join(tmp_dir, f"tmp_config_{sub_folder}.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    # Backup global config if exists, then overwrite it with plugin-specific one
    global_config_path = Path.home() / ".synthetic_data" / "config.yaml"
    backup_path = global_config_path.with_name("config.yaml.bak")

    # Make a backup if it doesn't already exist
    if global_config_path.exists() and not backup_path.exists():
        global_config_path.rename(backup_path)

    # Ensure the folder exists
    global_config_path.parent.mkdir(parents=True, exist_ok=True)

    # Write new config
    global_config_path.write_text(yaml.dump(config))

    started_vllm = False
    vllm_process = None

    # Check if vLLM is already running. If not, try launching it locally.
    if not is_server_alive(sys_path, api_base):
        try:
            # Attempt to start local vLLM using user-defined model path
            vllm_process = launch_vllm(model_path, port)
            if wait_for_server(api_base):
                started_vllm = True
                print("vLLM server started.")
            else:
                raise RuntimeError("vLLM did not start or is not reachable at expected API base.")
        except Exception as e:
            raise RuntimeError(f"Failed to start vLLM server: {e}")

    final_outputs = []
    total_docs = len(full_paths)

    # Process each uploaded document in the TransformerLab session
    for i, path in enumerate(full_paths):
        # Construct output filenames based on document basename and generation type
        base = Path(path).stem
        output_txt = f"transformerlab/plugins/synthetic_dataset_kit/data/{sub_folder}/output/{base}.txt"
        gen_json = f"transformerlab/plugins/synthetic_dataset_kit/data/{sub_folder}/generated/{base}_{generation_type}_pairs.json"
        clean_json = f"transformerlab/plugins/synthetic_dataset_kit/data/{sub_folder}/cleaned/{base}_{generation_type}_pairs_cleaned.json"
        final_jsonl = f"transformerlab/plugins/synthetic_dataset_kit/data/{sub_folder}/final/{base}_{generation_type}_pairs_cleaned.jsonl"

        try:
            # 1. Ingest: convert input file to plain text
            subprocess.run(
                [
                    sys_path,
                    "-c",
                    str(config_path),
                    "ingest",
                    path,
                    "-o",
                    f"transformerlab/plugins/synthetic_dataset_kit/data/{sub_folder}/output/",
                ],
                check=True,
            )

            # 2. Create: generate synthetic data based on document content
            subprocess.run(
                [
                    sys_path,
                    "create",
                    output_txt,
                    "--type",
                    generation_type,
                    "-o",
                    f"transformerlab/plugins/synthetic_dataset_kit/data/{sub_folder}/generated/",
                ],
                check=True,
            )

            # 3. Curate: filter QA pairs based on quality threshold
            subprocess.run(
                [
                    sys_path,
                    "curate",
                    gen_json,
                    "-t",
                    threshold,
                    "-o",
                    clean_json,
                ],
                check=True,
            )

            # 4. Save-as: convert result to desired output format (jsonl, alpaca, chatml, etc.)
            subprocess.run(
                [
                    sys_path,
                    "save-as",
                    clean_json,
                    "-f",
                    output_format,
                    "-o",
                    final_jsonl,
                ],
                check=True,
            )

            # Read final output file and parse each line to return to TransformerLab
            with open(final_jsonl) as f:
                if output_format in {"jsonl", "chatml"}:
                    for line in f:
                        final_outputs.append(json.loads(line))
                elif output_format == "alpaca":
                    final_outputs = json.load(f)  # Only valid if file is a single JSON array

            tlab_gen.progress_update((i + 1) / total_docs * 100)

        except subprocess.CalledProcessError as e:
            print("Error running command:")
            print("STDOUT:\n", e.stdout)
            print("STDERR:\n", e.stderr)
            if os.path.exists(config_path):
                os.remove(config_path)
            raise RuntimeError(f"Subprocess failed with code {e.returncode}") from e

    # Clean up temporary config file and restore original config
    if backup_path.exists():
        global_config_path.unlink(missing_ok=True)  # remove current modified file
        backup_path.rename(global_config_path)

    if os.path.exists(config_path):
        os.remove(config_path)

    # Stop the vLLM server if started here.
    if started_vllm:
        stop_process(vllm_process)

    df = pd.DataFrame(final_outputs)
    output_path, dataset_name = tlab_gen.save_generated_dataset(df)
    print(f"Dataset saved to {output_path}")

    # Clean up all synthetic-data-kit working folders created in the plugin root (if any)
    system_generated_folders = [
        "data/cleaned",
        "data/final",
        "data/html",
        "data/pdf",
        "data/txt",
        "data/docx",
        "data/generated",
        "data/output",
        "data/ppt",
        "data/youtube",
    ]

    for folder in system_generated_folders:
        if os.path.isdir(folder):
            try:
                shutil.rmtree(folder)
                print(f"[INFO] Deleted system folder: {folder}")
            except Exception as e:
                print(f"[WARN] Failed to delete {folder}: {e}")

    # If 'data/' is now empty, delete it too
    if os.path.isdir("data") and not os.listdir("data"):
        try:
            os.rmdir("data")
            print("[INFO] Deleted empty root 'data/' directory.")
        except Exception as e:
            print(f"[WARN] Failed to delete 'data/' directory: {e}")
    return True


run_generation()
