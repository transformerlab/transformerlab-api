# Evaluating LLM Performance with DeepEval Objective Metrics

The DeepEval objective metrics are a set of metrics that evaluate the performance of a language model on a set of tasks. These metrics are designed to provide a comprehensive evaluation of the model's capabilities across a range of tasks, including text generation, question answering, and summarization.

We currently support the following objective metrics:
- "Rouge",
- "BLEU",
- "Exact Match",
- "Quasi Exact Match",
- "Quasi Contains",
- "BERT Score"


## Parameters

### Run Name (Assign a name to your eval): 
This parameter allows you to assign a name to your evaluation run. It helps in identifying and organizing different evaluation runs.
- **Default**: `anonymous-llama`

### Evaluation Metric: 
This parameter allows you to select an evaluation metric from the provided list. Each metric has its own method of evaluating the performance of your model.
- **Options**: `Rouge`, `BLEU`, `Exact Match`, `Quasi Exact Match`, `Quasi Contains`, `BERT Score`

### Dataset Path (with file name): 
This parameter specifies the local path to the dataset file. Ensure that this is a CSV file with columns: `input`, `output`, `expected_output`. The `context` column is optional if using metrics which don't require it.

### Output Path: 
This parameter specifies the path where the evaluation results will be saved.

