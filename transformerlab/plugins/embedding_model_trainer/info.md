# Embedding Model Trainer (Matryoshka Representation Learning)

## Overview

The **Embedding Model Trainer** plugin is designed to train or fine-tune embedding models using [Sentence Transformers v3](https://www.sbert.net/), optionally leveraging **Matryoshka Representation Learning (MRL)**. This technique allows your model to produce embeddings that can be **truncated** to various sizes with minimal performance loss. 

### Key Features
1. **Flexible Dataset Handling**  
   - Supports both local (custom user data) and Hugging Face-hosted datasets.
   - For simple pair-based data, expects columns `(anchor, positive)` or `(id, anchor, positive)`.
2. **Powerful Loss Functions**  
   - Wraps `MultipleNegativesRankingLoss` with `MatryoshkaLoss` for multi-dimension embedding learning.
3. **Evaluation**  
   - Automatically creates multiple `InformationRetrievalEvaluator` instances with truncated embeddings if `(id, anchor, positive)` columns exist in your dataset.
   - Provides IR metrics such as `NDCG@10` across each truncated dimension.
4. **Ease of Integration**  
   - Compatible with your Electron/React + FastAPI environment.
   - Monitors training progress and errors, logging them into the LLM Lab database.
   - Optionally integrates with Weights & Biases (W&B) if a W&B key is configured.


## Parameters

| **Name**             | **Type**             | **Default**             | **Description**                                                                                                                                     |
|----------------------|----------------------|-------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------|
| **num_train_epochs** | integer              | 3                       | Number of epochs for fine-tuning the embedding model.                                                                                               |
| **batch_size**       | integer              | 16                      | Batch size per device (GPU/CPU). Larger values can speed up training but require more memory.                                                       |
| **learning_rate**    | number               | 2e-5                    | Learning rate for the optimizer. Typical values range from 1e-5 to 1e-4.                                                                            |
| **warmup_ratio**     | number               | 0.1                     | Fraction of total training steps used for linear LR warmup before switching to the main schedule.                                                   |
| **fp16**             | boolean              | true                    | Enable half-precision floating point training if your GPU supports it, for faster throughput.                                                       |
| **bf16**             | boolean              | false                   | Use bfloat16 if your GPU supports it (e.g., certain Ampere architectures). Often used on TPUs or new NVIDIA GPUs.                                   |
| **max_samples**      | integer              | -1                      | Limit the number of samples used in training for faster iteration or testing. `-1` uses all samples.                                               |
| **log_to_wandb**     | boolean              | false                   | Log metrics to Weights & Biases if you have a W&B key configured in your environment settings.                                                      |
| **matryoshka_dims**  | array of integers    | `[768, 512, 256, 128, 64]` | (Optional) Dimensions for [Matryoshka Representation Learning](https://www.sbert.net/docs/package_reference/losses.html#matryoshkaloss). Must be in descending order. |


*\[Note\]: `matryoshka_dims` is not explicitly listed in the default `index.json` but can be added if desired. The sample `main.py` demonstrates how to handle it.*


## Usage

1. **Setup and Installation**
   - Ensure the plugin folder contains `index.json`, `setup.sh`, and `main.py`.
   - On application start, dependencies in `setup.sh` are installed automatically.

2. **Preparing Your Dataset**
   - Provide columns: either `(anchor, positive)` or `(id, anchor, positive)` if you want automated evaluation.
   - If your dataset is local, place it under the `datasets/` folder in the workspace or install it from a Hugging Face dataset ID.

3. **Configuring Parameters**
   - Select the embedding base model (`model_name`) and dataset (`dataset_name`).
   - Set training parameters (epochs, batch size, etc.) to suit your hardware and task.
   - Enable `log_to_wandb` if you want to track training in Weights & Biases.

4. **Running Training**
   - Initiate the training from the app. The plugin will:
     1. Load your dataset.
     2. Initialize a `SentenceTransformer` model.
     3. Apply `MultipleNegativesRankingLoss` combined with `MatryoshkaLoss`.
     4. If `(id, anchor, positive)` columns are found, create a sequence of IR evaluators for multi-dimension evaluation.
     5. Monitor training progress and store logs in LLM Lab’s database.
     6. Optionally log metrics to W&B.

5. **Evaluating Multi-Dimension Embeddings**
   - During training, the plugin truncates embedding layers to the various `matryoshka_dims` sizes and evaluates them.
   - After completion, you’ll see metrics such as `NDCG@10` or `Recall@k` for each truncated dimension.

6. **Final Model Artifacts**
   - Upon success, the plugin saves the trained model and any final artifacts to the specified `output_dir`.
   - The job status is marked `success` if it completes, or `failed` with an error message otherwise.


## Example Scenario

**Fine-tuning a Base Embedding Model for Financial Documents**  
- **Dataset**: (id, anchor, positive) pairs of questions and relevant document context.  
- **Model**: Start from `BAAI/bge-base-en-v1.5`.  
- **Matryoshka**: `[768, 512, 256, 128, 64]`.  
- **Epochs**: 3.  
- **batch_size**: 16.  
- **Output**: A single model that can produce embeddings of multiple sizes, validated at each dimension with IR metrics. You store the smaller embeddings (e.g., 128) to handle large-scale retrieval with minimal performance drop.


## Troubleshooting & Tips

- **No `id` Column?** Add one programmatically:
    ```python
    ds = ds.add_column("id", range(len(ds)))
    ``` 
    This is needed if you want automatic IR evaluation.

- **Mismatch in Columns?** Adjust the columns or the loss/evaluation logic in `main.py` to match your dataset format.

- **GPU Compatibility:**
    - Use `fp16` if you have an NVIDIA GPU with half-precision support (e.g., RTX 20 series or later).
    - `bf16` is typically used on newer GPUs or TPUs.

- **Large Datasets:** Increase batch size only if you have enough GPU memory.
- **Logs:** If `log_to_wandb` is `true`, ensure your W&B key is set in the platform settings.

Feel free to customize the plugin code for advanced features or additional losses. For support, reach out on Discord.

Happy Training!