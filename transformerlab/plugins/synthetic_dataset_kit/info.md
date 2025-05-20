# Synthetic Dataset Generator (Meta synthetic-data-kit)

## Overview

The Synthetic Dataset Generator plugin creates synthetic question-answer pairs, reasoning traces, or summaries from uploaded documents using Meta's [`synthetic-data-kit`](https://github.com/meta-llama/synthetic-data-kit). It is designed to help users bootstrap high-quality datasets for fine-tuning or evaluation by generating natural language examples directly from content.

The plugin:
- creates a vLLM server using the selected model in the foundation.
- Uses Meta’s structured CLI pipeline: `ingest`, `create`, `curate`, `save-as`
- Supports generating QA, CoT (chain-of-thought), or summaries
- Accepts various input formats (PDF, DOCX, PPTX, HTML, YouTube, plain text)
- Works with models already running in Transformer Lab or launched locally via vLLM
- Produces datasets in formats ready for Alpaca, ChatML, or JSONL finetuning workflows

This tool is ideal for researchers or ML engineers looking to prepare high-quality datasets without manual labeling or scripting.

## Parameters

### Generation Type (`generation_type`)

- **Type:** string
- **Default:** `qa`
- **Options:** `qa`, `cot`, `summary`
- **Description:**  
  Controls what type of examples are generated from the input documents.

### Number of Pairs (`num_pairs`)

- **Type:** integer
- **Default:** 25
- **Range:** 1–500
- **Description:**  
  The number of examples to generate per document. The actual number may vary depending on input size and curation threshold.

### Curation Threshold (`curation_threshold`)

- **Type:** float
- **Default:** 7.0
- **Range:** 1.0–10.0
- **Description:**  
  Score threshold used during the `curate` step to filter low-quality outputs. Higher values retain only top-scoring examples.

### Output Format (`output_format`)

- **Type:** string
- **Default:** `jsonl`
- **Options:** `jsonl`, `alpaca`, `chatml`
- **Description:**  
  The desired format of the final dataset file.

### Prompt Template (`prompt_template`)

- **Type:** string (optional)
- **Description:**  
  Allows overriding the default prompt template used during generation by writing a YAML-compatible prompt string.

## Usage
1. **Run a desired model inside the app** Choose the model you'll want to use as the generator and run it.
1. **Provide Input Documents:** Upload PDFs, TXT, DOCX, etc., in the Documents tab.
2. **Choose Generation Type:** Select whether to generate QA pairs, reasoning examples, or summaries.
3. **Tune Parameters:** Adjust number of outputs, output format, and optional curation quality threshold.
4. **(Optional) Provide Prompt Template:** If needed, insert a custom prompt to guide the generation.
5. **Run the Plugin:** The final dataset will be automatically saved and visible inside Transformer Lab.

## Output Format

The final output format depends on your selection. A sample JSONL structure might look like:

```json
[
  {
    "input": "What is the purpose of Transformer Lab plugins?",
    "output": "They allow users to extend model training, generation, and evaluation functionality.",
    "metadata": {
      "source": "plugin_doc.pdf"
    }
  }
]
```

## Notes

- This plugin does not require you to install or configure `synthetic-data-kit` manually.
- All processing is orchestrated via Transformer Lab.
- Temporary files and configurations are automatically cleaned after job completion.
