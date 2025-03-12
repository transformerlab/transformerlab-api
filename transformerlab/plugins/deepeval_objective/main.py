from datetime import datetime

import nltk
import pandas as pd
from deepeval.metrics import BaseMetric
from deepeval.scorer import Scorer
from deepeval.test_case import LLMTestCase

from transformerlab.tfl_decorators import tfl_evals

nltk.download("punkt_tab")


# Define the metric classes
class RougeMetric(BaseMetric):
    def __init__(self, threshold: float = 0.5, score_type: str = "rouge1"):
        self.threshold = threshold
        self.score_type = score_type
        self.scorer = Scorer()

    def measure(self, test_case: LLMTestCase):
        self.score = self.scorer.rouge_score(
            prediction=test_case.actual_output, target=test_case.expected_output, score_type=self.score_type
        )
        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case: LLMTestCase):
        return self.measure(test_case)

    def is_successful(self):
        return self.success

    @property
    def __name__(self):
        return "Rouge Metric"


class SentenceBleuMetric(BaseMetric):
    def __init__(self, threshold: float = 0.5, score_type: str = "bleu1"):
        self.threshold = threshold
        self.score_type = score_type
        self.scorer = Scorer()

    def measure(self, test_case: LLMTestCase):
        self.score = self.scorer.sentence_bleu_score(
            prediction=test_case.actual_output, references=test_case.expected_output, bleu_type=self.score_type
        )
        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case: LLMTestCase):
        return self.measure(test_case)

    def is_successful(self):
        return self.success

    @property
    def __name__(self):
        return "Sentence Bleu Metric"


class ExactMatchScore(BaseMetric):
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.scorer = Scorer()

    def measure(self, test_case: LLMTestCase):
        self.score = self.scorer.exact_match_score(
            prediction=test_case.actual_output,
            target=test_case.expected_output,
        )
        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case: LLMTestCase):
        return self.measure(test_case)

    def is_successful(self):
        return self.success

    @property
    def __name__(self):
        return "Exact Match Score Metric"


class QuasiExactMatchScore(BaseMetric):
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.scorer = Scorer()

    def measure(self, test_case: LLMTestCase):
        self.score = self.scorer.quasi_exact_match_score(
            prediction=test_case.actual_output,
            target=test_case.expected_output,
        )
        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case: LLMTestCase):
        return self.measure(test_case)

    def is_successful(self):
        return self.success

    @property
    def __name__(self):
        return "Quasi Exact Match Score Metric"


class QuasiContainsScore(BaseMetric):
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.scorer = Scorer()

    def measure(self, test_case: LLMTestCase):
        self.score = self.scorer.quasi_contains_score(
            prediction=test_case.actual_output,
            targets=test_case.expected_output,
        )
        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case: LLMTestCase):
        return self.measure(test_case)

    def is_successful(self):
        return self.success

    @property
    def __name__(self):
        return "Quasi Contains Score Metric"


class BertScoreMetric(BaseMetric):
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.scorer = Scorer()

    def measure(self, test_case: LLMTestCase):
        self.score = self.scorer.bertscore_score(
            prediction=test_case.actual_output,
            reference=test_case.expected_output,
        )
        self.success = self.score["bert-f1"] >= self.threshold
        return self.score

    async def a_measure(self, test_case: LLMTestCase):
        return self.measure(test_case)

    def is_successful(self):
        return self.success

    @property
    def __name__(self):
        return "BertScore Metric"


# Define the metric classes dictionary
metric_classes = {
    "rouge": RougeMetric,
    "bleu": SentenceBleuMetric,
    "exact_match": ExactMatchScore,
    "quasi_exact_match": QuasiExactMatchScore,
    "quasi_contains": QuasiContainsScore,
    "bert_score": BertScoreMetric,
}


# Use the job_wrapper decorator to handle job status updates
@tfl_evals.job_wrapper(progress_start=0, progress_end=100)
def run_evaluation():
    # Setup evaluation logging
    tfl_evals.setup_eval_logging()

    # Record start time
    tfl_evals.add_job_data("start_time", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    tfl_evals.add_job_data("template_name", tfl_evals.run_name)
    tfl_evals.add_job_data("model_name", tfl_evals.model_name)
    tfl_evals.add_job_data("config", str(vars(tfl_evals)))

    # Parse tasks
    tasks = tfl_evals.tasks.split(",")
    tasks = [metric.lower().replace(" ", "_") for metric in tasks]

    # Load the dataset
    dataset = tfl_evals.load_dataset()
    df = dataset["train"].to_pandas()

    # Check required columns
    assert "input" in df.columns, "Input column not found in the dataset"
    assert "output" in df.columns, "Output column not found in the dataset"
    assert "expected_output" in df.columns, "Expected output column not found in the dataset"

    # Create a list of test cases
    test_cases = []
    for _, row in df.iterrows():
        test_case = LLMTestCase(input=row["input"], actual_output=row["output"], expected_output=row["expected_output"])
        test_cases.append(test_case)

    if hasattr(tfl_evals, "limit") and tfl_evals.limit and float(tfl_evals.limit) < 1.0:
        num_samples = int(len(test_cases) * float(tfl_evals.limit))
        if num_samples < 1:
            num_samples = 1
        test_cases = test_cases[:num_samples]

    print(f"Test cases loaded successfully: {len(test_cases)}")
    tfl_evals.progress_update(20)

    if not hasattr(tfl_evals, "threshold"):
        tfl_evals.threshold = 0.5
    # Calculate metrics for each test case
    metrics = []
    for metric_name in tasks:
        metric = metric_classes[metric_name](threshold=float(tfl_evals.threshold))
        for idx, test_case in enumerate(test_cases):
            score = metric.measure(test_case)
            if metric_name == "bert_score":
                metrics.append(
                    {
                        "test_case_id": f"test_case_{idx}",
                        "metric_name": metric_name,
                        "score": score["bert-f1"],
                        "bert_precision": score["bert-precision"],
                        "bert_recall": score["bert-recall"],
                        "bert_f1": score["bert-f1"],
                        "input": test_case.input,
                        "actual_output": test_case.actual_output,
                        "expected_output": test_case.expected_output,
                    }
                )
            else:
                metrics.append(
                    {
                        "test_case_id": f"test_case_{idx}",
                        "metric_name": metric_name,
                        "score": score,
                        "input": test_case.input,
                        "actual_output": test_case.actual_output,
                        "expected_output": test_case.expected_output,
                    }
                )

    tfl_evals.progress_update(60)

    # Create DataFrame and save results
    metrics_df = pd.DataFrame(metrics)
    tfl_evals.save_evaluation_results(metrics_df)

    tfl_evals.progress_update(80)

    # Log metrics to TensorBoard
    for metric_name in tasks:
        avg_score = metrics_df[metrics_df["metric_name"] == metric_name]["score"].mean()
        tfl_evals.log_metric(metric_name, avg_score)

    return True


# Run the evaluation when script is executed
run_evaluation()
