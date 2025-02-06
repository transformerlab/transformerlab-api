# Data Synthesizer

## Overview

The Data Synthesizer plugin uses Large Language Models (LLMs) to create synthetic data for various use cases. This plugin supports different generation types and models, allowing users to generate data based on reference documents, raw reference text, or from scratch based on a scenario.

## Parameters

### Generation Type

- **Type:** string
- **Description:** Select the type of generation you want to use. The available options are:
  - **docs:** Generate using reference documents.
  - **context:** Generate using raw reference text.
  - **scratch:** Generate from scratch based on a scenario.

### Generation Model

- **Type:** string
- **Description:** Select the model to be used for generation. The available options are:
  - **Claude 3.5 Haiku**
  - **Claude 3.5 Sonnet**
  - **OpenAI GPT 4o**
  - **OpenAI GPT 4o Mini**
  - **Local**

### Number of Samples to Generate

- **Type:** integer
- **Minimum:** 1
- **Maximum:** 1000
- **Default:** 10
- **Description:** Specify the number of samples to generate.

### Reference Documents

- **Type:** string
- **Description:** Provide a comma-separated list of document paths for 'docs' generation only.

### Embedding Model

- **Type:** string
- **Description:** Provide the name of the embedding model from Huggingface or its local path for 'docs' generation only.

### Context

- **Type:** string
- **Description:** Paste all your reference context here for 'context' generation only.
- **Example:** ""

### Scenario

- **Type:** string
- **Description:** Describe the scenario for which you want to generate data for 'scratch' generation only.

### Task

- **Type:** string
- **Description:** Describe the task for which you want to generate data for 'scratch' generation only.

### Input Format

- **Type:** string
- **Description:** Describe the format of the input data which will be sent to the model for 'scratch' generation only.

### Expected Output Format

- **Type:** string
- **Description:** Describe the format of the output data which will be received from the model for 'scratch' generation only.

## Usage

1. **Select the Generation Type:** Choose the type of generation that best fits your needs from the `generation_type` parameter.
2. **Select the Generation Model:** Choose the model to be used for generation from the `generation_model` parameter.
3. **Specify the Number of Samples:** Enter the number of samples to generate in the `num_goldens` parameter.
4. **Provide Reference Documents:** If using 'docs' generation, provide a comma-separated list of document paths in the `docs` parameter.
5. **Specify the Embedding Model:** If using 'docs' generation, provide the name or path of the embedding model in the `embedding_model` parameter.
6. **Provide Context:** If using 'context' generation, paste your reference context in the `context` parameter.
7. **Describe the Scenario:** If using 'scratch' generation, describe the scenario in the `scenario` parameter.
8. **Describe the Task:** If using 'scratch' generation, describe the task in the `task` parameter.
9. **Specify the Input Format:** If using 'scratch' generation, describe the input format in the `input_format` parameter.
10. **Specify the Expected Output Format:** If using 'scratch' generation, describe the expected output format in the `expected_output_format` parameter.
