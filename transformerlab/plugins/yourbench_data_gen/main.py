import os
import yaml
import subprocess
from huggingface_hub import get_token, HfApi
from datasets import load_from_disk
from transformerlab.sdk.v1.generate import tlab_gen


def generate_config():
    """Generate YourBench configuration based on tlab_gen params"""
    hf_token = get_token()
    hf_username = get_huggingface_username(hf_token)
    docs = tlab_gen.params.docs.split(",")[0]

    # Load generation model config
    trlab_model = tlab_gen.load_evaluation_model(field_name="generation_model")
    print("Model loaded successfully")
    tlab_gen.progress_update(30)

    tlab_gen.params.documents_dir = os.path.join(
        os.environ.get("_TFL_WORKSPACE_DIR"), "experiments", tlab_gen.params.experiment_name, "documents", docs
    )
    if not os.path.isdir(tlab_gen.params.documents_dir):
        raise FileNotFoundError("Please provide a directory containing all your files instead of individual files")
    config = {
        "settings": {"debug": True},
        "hf_configuration": {
            "token": hf_token,
            "hf_organization": hf_username,
            "private": True,
            "hf_dataset_name": f"{tlab_gen.params.template_name}_{tlab_gen.params.job_id}",
            "concat_if_exist": False,
        },
        "local_dataset_dir": tlab_gen.params.local_dataset_dir,
        "model_list": [
            {
                "model_name": trlab_model.generation_model_name,
                "api_key": trlab_model.api_key,
                "max_concurrent_requests": 8,
                "base_url": trlab_model.base_url,
            },
        ],
        "model_roles": {
            "ingestion": [trlab_model.generation_model_name],
            "summarization": [trlab_model.generation_model_name],
            "chunking": [trlab_model.generation_model_name],
            "single_shot_question_generation": [trlab_model.generation_model_name],
            "multi_hop_question_generation": [trlab_model.generation_model_name],
        },
        "pipeline": {
            "ingestion": {
                "run": True,
                "source_documents_dir": tlab_gen.params.documents_dir,
                "output_dir": tlab_gen.params.output_dir,
            },
            "upload_ingest_to_hub": {"run": True, "source_documents_dir": tlab_gen.params.output_dir},
            "summarization": {"run": True},
            "chunking": {
                "run": True,
                "chunking_configuration": {
                    "l_min_tokens": int(tlab_gen.params.l_min_tokens),
                    "l_max_tokens": int(tlab_gen.params.l_max_tokens),
                    "tau_threshold": float(tlab_gen.params.tau_threshold),
                    "h_min": int(tlab_gen.params.h_min),
                    "h_max": int(tlab_gen.params.h_max),
                    "num_multihops_factor": int(tlab_gen.params.num_multihops_factor),
                },
            },
            "single_shot_question_generation": {
                "run": True,
                "additional_instructions": tlab_gen.params.single_shot_instructions,
                "chunk_sampling": {
                    "mode": tlab_gen.params.single_shot_sampling_mode,
                    "value": tlab_gen.params.single_shot_sampling_value,
                    "random_seed": tlab_gen.params.single_shot_random_seed,
                },
            },
            "multi_hop_question_generation": {
                "run": True,
                "additional_instructions": tlab_gen.params.multi_hop_instructions,
                "chunk_sampling": {
                    "mode": tlab_gen.params.multi_hop_sampling_mode,
                    "value": tlab_gen.params.multi_hop_sampling_value,
                    "random_seed": tlab_gen.params.multi_hop_random_seed,
                },
            },
            "lighteval": {"run": True},
        },
    }

    return config


def get_huggingface_username(token):
    api = HfApi()
    user_info = api.whoami(token=get_token())
    return user_info["name"]


def save_generated_datasets(output_dir):
    dataset_types = ["chunked", "lighteval", "ingested", "multi_hop_questions", "single_shot_questions", "summarized"]
    for data_split in dataset_types:
        dataset = load_from_disk(os.path.join(output_dir, data_split))
        df = dataset[data_split].to_pandas()
        # Save the generated data and upload to TransformerLab
        additional_metadata = {"source_docs": tlab_gen.params.documents_dir}
        # Save the dataset using tlab_gen
        custom_name = tlab_gen.params.get("output_dataset_name")
        if custom_name and data_split != "train":
            # For non-train splits, append the split name to avoid conflicts
            custom_name = f"{custom_name}_{data_split}"
        tlab_gen.save_generated_dataset(
            df, additional_metadata=additional_metadata, suffix=data_split, dataset_id=custom_name
        )


@tlab_gen.job_wrapper(progress_start=0, progress_end=100)
def run_yourbench():
    """Run YourBench with generated configuration"""
    # Ensure arguments are parsed
    tlab_gen._ensure_args_parsed()

    # Get output directory for the config file
    output_dir = tlab_gen.get_output_file_path(dir_only=True)
    tlab_gen.params.local_dataset_dir = output_dir
    tlab_gen.params.output_dir = os.path.join(output_dir, "temp")
    if not os.path.exists(tlab_gen.params.output_dir):
        os.makedirs(tlab_gen.params.output_dir)
    config_path = os.path.join(output_dir, f"yourbench_config_{tlab_gen.params.job_id}.yaml")

    # Generate the configuration
    tlab_gen.progress_update(10)
    config = generate_config()

    tlab_gen.progress_update(20)

    # Write the configuration to a file
    with open(config_path, "w") as config_file:
        yaml.dump(config, config_file, default_flow_style=False)

    print(f"Configuration written to {config_path}")
    tlab_gen.add_job_data("config_file", config_path)

    tlab_gen.progress_update(30)

    # Get the yourbench directory path
    current_dir = os.path.join(os.environ["_TFL_WORKSPACE_DIR"], "plugins", "yourbench_data_gen")
    yourbench_dir = os.path.join(current_dir, "yourbench")

    # Run yourbench with the configuration
    try:
        print(f"Executing YourBench with config: {config_path}")
        tlab_gen.progress_update(40)

        command = f"""
            yourbench run --config {config_path}
            """

        process = subprocess.Popen(
            command,
            cwd=yourbench_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            shell=True,
            executable="/bin/bash",  # ensure bash is used
        )

        # Monitor progress
        for line in process.stdout:
            print(line.strip())
            # Update progress based on output (simplified approach)
            if "Ingestion" in line:
                tlab_gen.progress_update(50)
            elif "Summarization" in line:
                tlab_gen.progress_update(60)
            elif "Chunking" in line:
                tlab_gen.progress_update(70)
            elif "Single shot question" in line:
                tlab_gen.progress_update(80)
            elif "Multi hop question" in line:
                tlab_gen.progress_update(90)

        process.wait()

        if process.returncode == 0:
            print("YourBench execution completed successfully!")
            tlab_gen.progress_update(95)
            print("Saving generated datasets now...")
            save_generated_datasets(output_dir)

            return config_path
        else:
            error_msg = f"YourBench process exited with error code: {process.returncode}"
            print(error_msg)
            raise ValueError("Error in process")

    except subprocess.CalledProcessError as e:
        error_msg = f"Error running YourBench: {e}"
        print(error_msg)
        raise ValueError("Error in process")
    except FileNotFoundError:
        error_msg = "Error: The 'yourbench' command was not found. Please ensure it's installed and in your PATH."
        print(error_msg)
        raise ValueError("Error in process")


run_yourbench()
