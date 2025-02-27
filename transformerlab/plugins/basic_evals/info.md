# Basic Evaluation Metrics

## Overview

This plugin helps you evaluate Language Model (LLM) outputs using predefined metrics and custom evaluation criteria. You can use ready-made metrics or create your own using regular expressions.

## Features

### Pre-defined Metrics

Choose from a variety of built-in evaluation metrics:

- Content Structure (headings, lists, tables)
- Text Formatting (bold, italic, underline)
- Special Elements (code blocks, URLs, images)
- Data Validation (JSON, numbers, dates)
- Text Analysis (word count, emojis)

### Custom Evaluation

Create your own metrics with:

- Custom names for each metric
- Regular expressions for pattern matching
- Multiple return types:
  - boolean: Yes/No answers
  - number: Count occurrences
  - contains: Check if text contains a pattern
  - isequal: Exact match comparison

## Getting Started

### 1. Prepare Your Dataset

- Ensure your dataset has input and output columns
- Default column names are "input" and "output"
- You can customize column names if needed

### 2. Choose Evaluation Methods

#### Using Pre-defined Metrics
Simply select from the dropdown list of available metrics like:

```json
[
  "Is Valid JSON",
  "Word Count",
  "Contains URLs",
  "Contains code blocks"
]
```

#### Creating Custom Metrics
Define your own evaluation tasks:

```json
{
  "name": "Contains Numbers",
  "expression": "\\d+",
  "return_type": "boolean"
}
```

### 3. Configure Evaluation Settings

- **Data Sample Size**: Choose how much of your dataset to evaluate (0.1 to 1.0)
- **Column Names**: Specify input and output column names if different from defaults

### Example Usage

Here's a complete evaluation setup example:

```json
{
  "predefined_tasks": ["Word Count", "Contains URLs"],
  "tasks": [
    {
      "name": "Has Date",
      "expression": "\\d{2}/\\d{2}/\\d{4}",
      "return_type": "boolean"
    }
  ],
  "limit": 0.5,
  "input_col": "prompt",
  "output_col": "response"
}
```

This configuration will:

- Use two predefined metrics
- Add a custom date detection metric
- Evaluate 50% of the dataset
- Use "prompt" and "response" as column names