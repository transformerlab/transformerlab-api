# Data Synthesizer

## Overview

The Data Synthesizer plugin uses Large Language Models (LLMs) to create synthetic data for various use cases. This plugin supports different generation types and models, allowing users to generate data based on reference documents.

## Parameters

### Generation Model

- **Type:** string
- **Description:** Select the model to be used for generation. The available options are:
  - **Claude 3.5 Haiku**
  - **Claude 3.5 Sonnet**
  - **OpenAI GPT 4o**
  - **OpenAI GPT 4o Mini**
  - **Local**

### Reference Documents

- **Type:** string
- **Description:** Provide a comma-separated list of document paths for 'docs' generation only.

### Embedding Model

- **Type:** string
- **Description:** Provide the name of the embedding model from Huggingface or its local path for 'docs' generation only.

## Usage

1. **Select the Generation Model:** Choose the model to be used for generation from the `generation_model` parameter.
2. **Provide Reference Documents:** If using 'docs' generation, provide a comma-separated list of document paths in the `docs` parameter.
3. **Specify the Embedding Model:** If using 'docs' generation, provide the name or path of the embedding model in the `embedding_model` parameter.
4. **Generate Synthetic Data:** Save this task create synthetic data based on the selected parameters.
