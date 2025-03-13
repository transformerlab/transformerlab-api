import pandas as pd
import requests
from requests_batching import process_dataset

from transformerlab.tfl_decorators import tfl_evals

# Metrics mapping
metrics_map = {
    "Time to First Token (TTFT)": "time_to_first_token",
    "Total Time": "time_total",
    "Prompt Tokens": "prompt_tokens",
    "Completion Tokens": "completion_tokens",
    "Total Tokens": "total_tokens",
    "Tokens per Second": "tokens_per_second",
}


def check_local_server():
    """Check if the local model server is running"""
    response = requests.get("http://localhost:8338/server/worker_healthz")
    if response.status_code != 200 or not isinstance(response.json(), list) or len(response.json()) == 0:
        print("Local Model Server is not running. Please start it before running the evaluation.")
        raise RuntimeError("Local Model Server is not running. Please start it before running the evaluation.")


async def generate_batched(trlab_model, df: pd.DataFrame, sys_prompt_col=None) -> pd.DataFrame:
    updated_df = await process_dataset(
        df,
        batch_size=tfl_evals.batch_size,
        model=trlab_model.generation_model_name,
        inference_url=trlab_model.chat_completions_url,
        api_key="dummy",
        sys_prompt_col=sys_prompt_col,
        input_col=tfl_evals.input_column,
        output_col=tfl_evals.output_column,
        temperature=float(tfl_evals.temperature),
        max_tokens=int(tfl_evals.max_tokens),
        top_p=float(tfl_evals.top_p),
    )
    return updated_df


@tfl_evals.async_job_wrapper(progress_start=0, progress_end=100)
async def run_evaluation():
    """Run the inference evaluation"""

    # Setup evaluation logging
    tfl_evals.setup_eval_logging()

    # Type casting for avoiding errors
    tfl_evals.batch_size = int(tfl_evals.batch_size)
    tfl_evals.temperature = float(tfl_evals.temperature)
    tfl_evals.max_tokens = int(tfl_evals.max_tokens)
    tfl_evals.top_p = float(tfl_evals.top_p)

    # Parse the tasks
    tasks = tfl_evals.tasks.split(",")

    tfl_evals.progress_update(10)

    # Load the appropriate model
    try:
        trlab_model = tfl_evals.load_evaluation_model(field_name="generation_model")

        print("Model loaded successfully")
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        raise

    tfl_evals.progress_update(20)

    # Load the dataset
    dataset = tfl_evals.load_dataset()
    df = dataset["train"].to_pandas()

    print("Dataset fetched successfully")

    sys_prompt_col = None
    if hasattr(tfl_evals, "system_prompt") and tfl_evals.system_prompt:
        df["system_prompt"] = tfl_evals.system_prompt
        sys_prompt_col = "system_prompt"

    tfl_evals.progress_update(30)

    # Run batch generation
    df = await generate_batched(trlab_model, df, sys_prompt_col=sys_prompt_col)
    print("Batched generation completed successfully")

    tfl_evals.progress_update(70)

    # Process metrics
    metrics = []
    for metric in tasks:
        metric_value = df[metrics_map[metric]].mean()
        tfl_evals.log_metric(metric, metric_value)

    # Create metrics DataFrame
    for idx, row in df.iterrows():
        for metric in tasks:
            if row[metrics_map[metric]] is not None:
                score = round(float(row[metrics_map[metric]]), 4)
            else:
                score = 0.0
            metrics.append(
                {
                    "test_case_id": f"test_case_{idx}",
                    "metric_name": metric,
                    "score": score,
                    "input": row[tfl_evals.input_column],
                    "output": row[tfl_evals.output_column],
                }
            )

    tfl_evals.progress_update(80)

    # Save results using EvalsTFLPlugin
    metrics_df = pd.DataFrame(metrics)
    output_path, plot_data_path = tfl_evals.save_evaluation_results(metrics_df)

    print(f"Metrics saved to {output_path}")
    print(f"Plotting data saved to {plot_data_path}")
    print("Evaluation completed.")

    return True


run_evaluation()
