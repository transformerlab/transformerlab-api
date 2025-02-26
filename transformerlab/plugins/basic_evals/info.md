# Basic Evaluation Metrics

## Overview

The Basic Evaluation Metrics plugin is designed to evaluate the outputs of Language Models (LLMs) using objective metrics. It allows you to define custom evaluation criteria using regular expressions and supports both boolean and numerical evaluation outputs.

## Parameters

### tasks

- **Title:** Evaluation Metrics
- **Type:** array of objects
- **Description:** Define multiple evaluation tasks, each containing:
  - **name:** Name of the evaluation metric
  - **expression:** Regular expression pattern to match
  - **return_type:** Output type ("boolean" or "number")

### limit

- **Title:** Fraction of samples to evaluate
- **Type:** number
- **Default:** 1.0
- **Range:** 0.0 to 1.0 (increments of 0.1)
- **Description:** Controls what fraction of the dataset to evaluate

### input_col

- **Title:** Input Column
- **Type:** string
- **Default:** "input"
- **Description:** Specifies the column name containing the input data

### output_col

- **Title:** Output Column
- **Type:** string
- **Default:** "output"
- **Description:** Specifies the column name containing the output data to evaluate

## Usage

1. **Prepare Your Dataset:**
   - Ensure your dataset has clearly defined input and output columns
   - Verify that column names match the configured `input_col` and `output_col` parameters
   - Make sure your data is in a compatible format (CSV, JSON, etc.)

2. **Configure Evaluation Tasks:**
   - Define evaluation metrics using the `tasks` parameter:

     ```json
     {
       "name": "Contains Number",
       "expression": "\\d+",
       "return_type": "boolean"
     }
     ```

   - Create multiple tasks to evaluate different aspects of the output
   - Choose appropriate return types:
     - `boolean`: For yes/no evaluations
     - `number`: For scoring or counting matches

3. **Set Evaluation Parameters:**
   - Adjust the `limit` parameter to control the fraction of data to evaluate
   - Configure input and output column names to match your dataset

4. **Run the Evaluation:**
   - Monitor the progress as the plugin processes your dataset
   - Results will be stored in the detailed report for further analysis
