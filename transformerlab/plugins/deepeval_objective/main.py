from deepeval.metrics import BaseMetric
from deepeval.scorer import Scorer
from deepeval.test_case import LLMTestCase
import pandas as pd
import argparse
import sys
import os
import traceback
import time
from datetime import datetime
from tensorboardX import SummaryWriter
import nltk
import transformerlab.plugin
from datasets import load_dataset

nltk.download("punkt_tab")

parser = argparse.ArgumentParser(description="Run DeepEval metrics for LLM-as-judge evaluation.")
parser.add_argument("--run_name", default="evaluation", type=str)
parser.add_argument("--model_name", default="gpt-j-6b", type=str, help="Model to use for evaluation.")
parser.add_argument("--experiment_name", default="", type=str)
parser.add_argument("--eval_name", default="", type=str)
parser.add_argument("--tasks", default="", type=str)
parser.add_argument(
    "--model_adapter",
    default=None,
    type=str,
)
parser.add_argument(
    "--dataset_name",
    default=None,
    type=str,
)
parser.add_argument("--threshold", default=0.5, type=float)
parser.add_argument("--job_id", default=None, type=str)
parser.add_argument("--limit", default=None, type=float)

args, other = parser.parse_known_args()

if args.job_id:
    job = transformerlab.plugin.Job(args.job_id)
    job.update_progress(0)
else:
    print("Job ID not provided.")
    sys.exit(1)


today = time.strftime("%Y%m%d-%H%M%S")
tensorboard_dir = os.path.join(os.environ["_TFL_WORKSPACE_DIR"], "experiments", args.experiment_name, "tensorboards")
# Find directory to put in based on eval name
combined_tensorboard_dir = None
for dir in os.listdir(tensorboard_dir):
    if args.run_name == dir or args.run_name == dir.lower():
        combined_tensorboard_dir = os.path.join(tensorboard_dir, dir)
if combined_tensorboard_dir is None:
    combined_tensorboard_dir = os.path.join(tensorboard_dir, args.run_name)
output_dir = os.path.join(combined_tensorboard_dir, f"evaljob_{args.job_id}_{today}")
os.makedirs(output_dir, exist_ok=True)
writer = SummaryWriter(output_dir)
job.set_tensorboard_output_dir(output_dir)
print("Writing tensorboard logs to", output_dir)

score_colour_sensitivity = {
    "Rouge": 1,
    "BLEU": 1,
    "Exact Match": 1,
    "Quasi Exact Match": 1,
    "Quasi Contains": 1,
    "BERT Score": 1,
}

print("ARGS", args)
try:
    job.add_to_job_data("config", str(args))
    job.add_to_job_data("template_name", args.run_name)
    job.add_to_job_data("model_name", args.model_name)
    job.add_to_job_data("start_time", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
except Exception as e:
    print(f"An error occurred while adding job data: {e}")
    job.add_to_job_data("end_time", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    job.set_job_completion_status("failed", "An error occurred while adding job data.")
    sys.exit(1)


args.tasks = args.tasks.split(",")
original_metric_names = args.tasks
args.tasks = [metric.lower().replace(" ", "_") for metric in args.tasks]


# # Set experiment name if None
# if args.experiment_name is None or args.experiment_name == "":
#     # Set experiment name to current timestamp

#     args.experiment_name = f"experiment_eval_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"


def get_tflab_dataset():
    try:
        dataset_target = transformerlab.plugin.get_dataset_path(args.dataset_name)
    except Exception as e:
        job.add_to_job_data("end_time", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        job.set_job_completion_status("failed", "Failure to get dataset")
        raise e
    dataset = {}
    dataset_types = ["train"]
    for dataset_type in dataset_types:
        try:
            dataset[dataset_type] = load_dataset(dataset_target, split=dataset_type, trust_remote_code=True)

        except Exception as e:
            job.add_to_job_data("end_time", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            job.set_job_completion_status("failed", "Failure to load dataset")
            raise e
    # Convert the dataset to a pandas dataframe
    df = dataset["train"].to_pandas()
    return df


def get_output_file_path():
    experiment_dir = os.path.join(os.environ["_TFL_WORKSPACE_DIR"], "experiments", args.experiment_name)
    p = os.path.join(experiment_dir, "evals", args.eval_name)
    os.makedirs(p, exist_ok=True)
    return os.path.join(p, f"detailed_output_{args.job_id}.csv")


def get_plotting_data(metrics_df):
    file_path = get_output_file_path()
    file_path = file_path.replace(".csv", "_plotting.json")
    metrics_data = metrics_df[["test_case_id", "metric_name", "score"]].copy()
    metrics_data["metric_name"] = metrics_data["metric_name"].apply(lambda x: x.replace("_", " ").title())
    metrics_data.to_json(file_path, orient="records", lines=False)

    return file_path


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

    # Async implementation of measure(). If async version for
    # scoring method does not exist, just reuse the measure method.
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

    # Async implementation of measure(). If async version for
    # scoring method does not exist, just reuse the measure method.
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

    # Async implementation of measure(). If async version for
    # scoring method does not exist, just reuse the measure method.
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

    # Async implementation of measure(). If async version for
    # scoring method does not exist, just reuse the measure method.
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

    # Async implementation of measure(). If async version for
    # scoring method does not exist, just reuse the measure method.
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
        # Only for this a dictionary is returned with keys: `bert-precision`, `bert-recall`, `bert-f1`
        self.score = self.scorer.bert_score(
            predictions=test_case.actual_output,
            references=test_case.expected_output,
        )
        self.score["bert-f1"] = self.score["bert-f1"][0]
        self.score["bert-precision"] = self.score["bert-precision"][0]
        self.score["bert-recall"] = self.score["bert-recall"][0]
        return self.score

    # Async implementation of measure(). If async version for
    # scoring method does not exist, just reuse the measure method.
    async def a_measure(self, test_case: LLMTestCase):
        return self.measure(test_case)

    def is_successful(self):
        return self.success

    @property
    def __name__(self):
        return "BertScore Metric"


metric_classes = {
    "rouge": RougeMetric,
    "bleu": SentenceBleuMetric,
    "exact_match": ExactMatchScore,
    "quasi_exact_match": QuasiExactMatchScore,
    "quasi_contains": QuasiContainsScore,
    "bert_score": BertScoreMetric,
}


def print_fancy_df(df):
    separator = "🟡" + "━" * 60 + "🟡"

    print(separator)
    print("📊  **Evaluation Results**  📊")
    print(separator)

    for _, row in df.iterrows():
        print(f"🔹 **Input:** {row['input']}")
        print(f"  ↳ 📝 **Output:** {row['actual_output']}")
        print(f"  ✅ **Expected:** {row['expected_output']}")
        if "context" in row:
            print(f"  🏷 **Context:** {row['context']}")
        elif "retrieval_context" in row:
            print(f"  🏷 **Retrieval Context:** {row['retrieval_context']}")
        print(f"  📊 **Score:** {row['score']:.2f} 🔥")
        print(f"  📐 **Metric:** {row['metric_name']} 📏")
        print(separator)


def run_evaluation():
    try:
        # Load the csv file
        # df = pd.read_csv(args.dataset_path)
        df = get_tflab_dataset()
        # Check if `input`, `output` and `expected_output` columns exist
        assert "input" in df.columns, (
            "Input column not found in the dataset. Please make sure the column name is `input`"
        )
        assert "output" in df.columns, (
            "Output column not found in the dataset. Please make sure the column name is `output`"
        )
        assert "expected_output" in df.columns, (
            "Expected output column not found in the dataset. Please make sure the column name is `expected_output`"
        )

        # Create a list of test cases
        test_cases = []
        for index, row in df.iterrows():
            test_case = LLMTestCase(
                input=row["input"], actual_output=row["output"], expected_output=row["expected_output"]
            )
            test_cases.append(test_case)

        if args.limit and float(args.limit) != 1.0:
            num_samples = int(len(test_cases) * float(args.limit))
            if num_samples < 1:
                num_samples = 1
            test_cases = test_cases[:num_samples]

        print(f"Test cases loaded successfully: {len(test_cases)}")

        job.update_progress(20)

        # Calculate metrics for each test case
        metrics = []
        for metric_name in args.tasks:
            metric = metric_classes[metric_name]()
            for idx, test_case in enumerate(test_cases):
                score = metric.measure(test_case)
                if metric_name == "bert_score":
                    metrics.append(
                        {
                            "test_case_id": f"test_case_{idx}",
                            "metric_name": metric_name,
                            "score": [score["bert-f1"], 1],
                            "bert_precision": score["bert-precision"],
                            "bert_recall": score["bert-recall"],
                            "bert_f1": score["bert-f1"],
                            "input": test_case.input,
                            "actual_output": test_case.actual_output,
                            "expected_output": test_case.expected_output,
                        }
                    )
                else:
                    score_arr = None
                    for key, value in score_colour_sensitivity.items():
                        if key.lower() == metric_name.lower():
                            score_arr = [score, value]
                            break
                    if score_arr is None:
                        score_arr = [score, value]

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
        job.update_progress(60)
        # Save the metrics to a csv file
        metrics_df = pd.DataFrame(metrics)
        output_path = get_output_file_path()
        metrics_df.to_csv(output_path, index=False)
        job.update_progress(80)
        plot_data_path = get_plotting_data(metrics_df)
        # print_fancy_df(metrics_df)

        for idx, metric in enumerate(args.tasks):
            print(
                f"Average {original_metric_names[idx]} score: {metrics_df[metrics_df['metric_name'] == metric]['score'].mean()}"
            )
        print(f"Metrics saved to {output_path}")
        print(f"Plotting data saved to {plot_data_path}")
        print("Evaluation completed.")
        job.update_progress(100)
        score_list = []
        for metric in args.tasks:
            writer.add_scalar(f"eval/{metric}", metrics_df[metrics_df["metric_name"] == metric]["score"].mean(), 1)
            score_list.append(
                {"type": metric, "score": round(metrics_df[metrics_df["metric_name"] == metric]["score"].mean(), 4)}
            )
        job.add_to_job_data("end_time", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        job.set_job_completion_status(
            "success",
            "Evaluation completed successfully.",
            score=score_list,
            additional_output_path=output_path,
            plot_data_path=plot_data_path,
        )
    except Exception as e:
        print("Error occurred while running the evaluation.")
        job.add_to_job_data("end_time", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        job.set_job_completion_status("failed", str(e))
        print(e)
        traceback.print_exc()


try:
    run_evaluation()
except Exception as e:
    print("Error occurred while running the evaluation.")
    job.add_to_job_data("end_time", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    job.set_job_completion_status("failed", str(e))
    print(e)
    traceback.print_exc()
