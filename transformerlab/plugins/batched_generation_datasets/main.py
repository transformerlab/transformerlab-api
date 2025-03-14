import pandas as pd
from requests_batching import process_dataset

from transformerlab.tfl_decorators import tfl_gen


async def generate_batched(trlab_model, df: pd.DataFrame, sys_prompt_col=None) -> pd.DataFrame:
    updated_df = await process_dataset(
        df,
        batch_size=tfl_gen.batch_size,
        model=trlab_model.generation_model_name,
        inference_url=trlab_model.chat_completions_url,
        api_key=trlab_model.api_key,
        sys_prompt_col=sys_prompt_col,
        input_col=tfl_gen.input_column,
        output_col=tfl_gen.output_column,
        temperature=float(tfl_gen.temperature),
        max_tokens=int(tfl_gen.max_tokens),
        top_p=float(tfl_gen.top_p),
    )
    return updated_df


@tfl_gen.async_job_wrapper(progress_start=0, progress_end=100)
async def run_generation():
    """Main function for batched generation"""
    print(f"Generation type: {tfl_gen.generation_type}")
    print(f"Model Name: {tfl_gen.generation_model}")
    print(f"Dataset Name: {tfl_gen.dataset_name}")

    # Load the dataset ('train' split) using tfl_gen's built-in method
    dataset = tfl_gen.load_dataset()
    df = dataset["train"].to_pandas()
    print(f"Dataset loaded successfully with {len(df)} rows")
    tfl_gen.progress_update(20)

    # Apply system prompt if provided
    sys_prompt_col = None
    if tfl_gen.system_prompt:
        print(f"Using system prompt: {tfl_gen.system_prompt}")
        df["system_prompt"] = tfl_gen.system_prompt
        sys_prompt_col = "system_prompt"

    # Check if we're using a local model and verify the server is running
    if "local" in tfl_gen.generation_model.lower():
        tfl_gen.check_local_server()

    # Load the model for generation
    trlab_model = tfl_gen.load_evaluation_model(field_name="generation_model")
    print("Model loaded successfully")
    tfl_gen.progress_update(30)

    # Run batched generation
    print("Running batched generation...")
    updated_df = await generate_batched(trlab_model, df, sys_prompt_col=sys_prompt_col)
    print("Batched generation completed successfully")
    tfl_gen.progress_update(80)

    # Save the results as a new dataset
    metadata = {
        "generation_method": "batched",
        "input_column": tfl_gen.input_column,
        "output_column": tfl_gen.output_column,
        "system_prompt": getattr(tfl_gen, "system_prompt", None),
        "batch_size": getattr(tfl_gen, "batch_size", 128),
        "temperature": getattr(tfl_gen, "temperature", 0.01),
        "max_tokens": getattr(tfl_gen, "max_tokens", 1024),
        "top_p": getattr(tfl_gen, "top_p", 1.0),
        "source_dataset": tfl_gen.dataset_name,
        "dataset_split": getattr(tfl_gen, "dataset_split", "train"),
    }

    output_file, dataset_name = tfl_gen.save_generated_dataset(updated_df, metadata)
    tfl_gen.progress_update(100)

    print(f"Dataset processed successfully as {dataset_name}")
    print(f"Saved to {output_file}")

    return updated_df


run_generation()
