import json
import os
import time

from transformerlab.plugin import test_wandb_login
from transformerlab.sdk.v1.tlab_plugin import TLabPlugin


class EvalsTLabPlugin(TLabPlugin):
    """Enhanced decorator class for TransformerLab evaluation plugins"""

    def __init__(self):
        super().__init__()
        # Add common evaluation-specific arguments
        self._parser.add_argument("--run_name", default="evaluation", type=str, help="Name for the evaluation run")
        self._parser.add_argument("--template_name", default="evaluation", type=str, help="Name for the evaluation run")
        self._parser.add_argument("--experiment_name", default="", type=str, help="Name of the experiment")
        self._parser.add_argument("--eval_name", default="", type=str, help="Name of the evaluation")
        self.tlab_plugin_type = "evals"

    def _ensure_args_parsed(self):
        """Parse arguments if not already done"""
        if not self._args_parsed:
            args, unknown_args = self._parser.parse_known_args()

            # Transfer all known arguments to attributes of self
            for key, value in vars(args).items():
                self.params[key] = value

            self._parse_unknown_args(unknown_args)
            self._args_parsed = True

    def _parse_unknown_args(self, unknown_args):
        """Parse unknown arguments which change with each eval"""
        key = None
        for arg in unknown_args:
            if arg.startswith("--"):  # Argument key
                key = arg.lstrip("-")
                self.params[key] = True
                setattr(self, key, True)
            elif key:  # Argument value
                self.params[key] = arg
                key = None

    def setup_eval_logging(self, wandb_project_name: str = "TLab_Evaluations", manual_logging=False):
        """Setup Weights and Biases and TensorBoard logging for evaluations

        Returns:
            str: Path to the TensorBoard output directory
        """
        from tensorboardX import SummaryWriter

        self._ensure_args_parsed()

        today = time.strftime("%Y%m%d-%H%M%S")
        workspace_dir = os.environ.get("_TFL_WORKSPACE_DIR", "./")

        # Create tensorboard directory structure
        tensorboard_dir = os.path.join(workspace_dir, "experiments", self.params.experiment_name, "tensorboards")
        os.makedirs(tensorboard_dir, exist_ok=True)

        # Find directory based on eval name
        combined_dir = None
        for dir_name in os.listdir(tensorboard_dir):
            if self.params.run_name == dir_name or self.params.run_name == dir_name.lower():
                combined_dir = os.path.join(tensorboard_dir, dir_name)
                break

        if combined_dir is None:
            combined_dir = os.path.join(tensorboard_dir, self.params.run_name)

        output_dir = os.path.join(combined_dir, f"evaljob_{self.params.job_id}_{today}")
        os.makedirs(output_dir, exist_ok=True)

        # Store the writer and output directory as instance variables
        self.params["tensorboard_output_dir"] = output_dir

        # Create writer and store it
        self.writer = SummaryWriter(output_dir)

        # Store the output directory in the job
        self.add_job_data("tensorboard_output_dir", self.params.tensorboard_output_dir)

        print(f"Writing tensorboard logs to {self.params.tensorboard_output_dir}")

        # Check for wandb logging preference
        log_to_wandb = getattr(self, "log_to_wandb", False)

        report_to = ["tensorboard"]

        if log_to_wandb:
            try:
                wandb_success, report_to = test_wandb_login(wandb_project_name)

                if wandb_success:
                    print(f"W&B logging enabled (project: {wandb_project_name})")
                    try:
                        import wandb

                        report_to.append("wandb")

                    except ImportError:
                        raise ImportError("Could not import wandb. Skipping W&B logging.")

                else:
                    print("W&B API key not found. W&B logging disabled.")
                    self.add_job_data("wandb_logging", False)

            except Exception as e:
                print(f"Error setting up W&B: {str(e)}. Continuing without W&B.")
                self.add_job_data("wandb_logging", False)
                report_to = ["tensorboard"]

        if "wandb" in report_to and manual_logging:
            self.wandb_run = wandb.init(
                project=wandb_project_name, config=self.params, name=f"{self.params.template_name}_{self.params.job_id}"
            )

        self.report_to = report_to

    def log_metric(self, metric_name, value, step=1):
        """Log a metric to TensorBoard

        Args:
            metric_name: Name of the metric
            value: Value of the metric
            step: Step number for TensorBoard
        """
        if "tensorboard" in self.report_to:
            self.writer.add_scalar(f"eval/{metric_name}", value, step)

        if "wandb" in self.report_to and getattr(self, "wandb_run") is not None:
            self.wandb_run.log({metric_name: value}, step=step)

    def get_output_file_path(self, suffix="", is_plotting=False, dir_only=False):
        """Get path for saving evaluation outputs

        Args:
            suffix: Optional suffix for the filename
            is_plotting: Whether this is for plotting data (uses .json extension)

        Returns:
            str: Full path for output file
        """

        self._ensure_args_parsed()

        workspace_dir = os.environ.get("_TFL_WORKSPACE_DIR", "./")
        experiment_dir = os.path.join(workspace_dir, "experiments", self.params.experiment_name)
        eval_dir = os.path.join(experiment_dir, "evals", self.params.eval_name, self.params.job_id)

        os.makedirs(eval_dir, exist_ok=True)

        if dir_only:
            return eval_dir

        if is_plotting:
            # For plotting data, we use a JSON file with a specific naming convention
            plotting_suffix = suffix if suffix else "plotting"
            if not plotting_suffix.endswith(".json"):
                plotting_suffix += ".json"
            return os.path.join(eval_dir, f"plot_data_{self.params.job_id}_{plotting_suffix}")
        else:
            # For regular outputs
            if suffix:
                if not any(suffix.endswith(ext) for ext in (".csv", ".json", ".txt")):
                    suffix += ".csv"
                return os.path.join(eval_dir, f"output_{self.params.job_id}_{suffix}")
            else:
                return os.path.join(eval_dir, f"output_{self.params.job_id}.csv")

    def save_evaluation_results(self, metrics_df):
        """Save evaluation results and generate plotting data

        Args:
            metrics_df: DataFrame containing evaluation metrics with
                       required columns "test_case_id", "metric_name", "score"

        Returns:
            tuple: Paths to the saved files (output_path, plot_data_path)

        Raises:
            ValueError: If required columns are missing from the DataFrame
        """
        # Validate that required columns exist
        required_columns = ["test_case_id", "metric_name", "score"]
        missing_columns = [col for col in required_columns if col not in metrics_df.columns]

        if missing_columns:
            raise ValueError(f"Missing required columns in metrics DataFrame: {missing_columns}")

        # Save full DataFrame to CSV
        output_path = self.get_output_file_path()
        metrics_df.to_csv(output_path, index=False)
        print(f"Saved detailed evaluation results to {output_path}")

        # Create and save plotting data
        plot_data_path = self.get_output_file_path(is_plotting=True)

        # Extract and format plotting data
        plotting_data = metrics_df[["test_case_id", "metric_name", "score"]].copy()

        # Format metric names for better display (replace underscores with spaces and capitalize)
        plotting_data["metric_name"] = plotting_data["metric_name"].apply(lambda x: x.replace("_", " ").title())

        # Save as JSON
        plotting_data.to_json(plot_data_path, orient="records", lines=False)
        print(f"Saved plotting data to {plot_data_path}")

        self.job.add_to_job_data("additional_output_path", output_path)
        self.job.add_to_job_data("plot_data_path", plot_data_path)

        # Create evaluation provenance file
        self.create_evaluation_provenance_file(metrics_df)

        # Print average scores by metric
        print("\n===== Evaluation Results =====")
        metrics = metrics_df["metric_name"].unique()
        score_list = []
        for metric in metrics:
            avg_score = metrics_df[metrics_df["metric_name"] == metric]["score"].mean()
            print(f"Average {metric}: {avg_score:.4f}")
            score_list.append({"type": metric, "score": avg_score})

        self.add_job_data("score", json.dumps(score_list))

        return output_path, plot_data_path

    def create_evaluation_provenance_file(self, metrics_df):
        """Add evaluation data to the existing _tlab_provenance.json file"""

        # Get evaluation parameters and metadata
        model_name = self.params.get("model_name", None)
        if not model_name:
            model_name = self.params.get("generation_model", "unknown_model")

        # Extract just the model name if it's a full path
        if model_name and "/" in model_name:
            model_name = os.path.basename(model_name)

        # Calculate average scores for each metric
        metrics_summary = {}
        for metric in metrics_df["metric_name"].unique():
            avg_score = metrics_df[metrics_df["metric_name"] == metric]["score"].mean()
            metrics_summary[metric] = avg_score

        evaluation_data = {
            "job_id": self.params.get("job_id", None),
            "model_name": model_name,
            "evaluation_type": self.params.get("eval_name", "evaluation"),
            "parameters": {
                "generation_model": self.params.get("generation_model", None),
                "tasks": self.params.get("tasks", None),
                "predefined_tasks": self.params.get("predefined_tasks", None),
                "input_column": self.params.get("input_column", None),
                "output_column": self.params.get("output_column", None),
            },
            "metrics_summary": metrics_summary,
            "total_test_cases": len(metrics_df["test_case_id"].unique()),
            "start_time": self.params.get("start_time", ""),
            "end_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        # Add evaluation data to the existing provenance file in the model directory
        if model_name and model_name != "unknown_model":
            # Try to find the model directory using environment variables
            workspace_dir = os.environ.get("_TFL_WORKSPACE_DIR", "./")
            models_dir = os.path.join(workspace_dir, "models")

            # Look for the model directory
            model_dir = None
            for entry in os.listdir(models_dir):
                if os.path.isdir(os.path.join(models_dir, entry)):
                    if entry == model_name or entry.endswith(f"_{model_name}") or model_name in entry:
                        model_dir = os.path.join(models_dir, entry)
                        break

            if model_dir and os.path.exists(model_dir):
                provenance_path = os.path.join(model_dir, "_tlab_provenance.json")

                # Load existing provenance data
                existing_provenance = {}
                if os.path.exists(provenance_path):
                    try:
                        with open(provenance_path, "r") as f:
                            existing_provenance = json.load(f)
                    except Exception as e:
                        print(f"Error loading existing provenance: {e}")
                        existing_provenance = {}

                # Initialize evaluations list if it doesn't exist
                if "evaluations" not in existing_provenance:
                    existing_provenance["evaluations"] = []

                # Add new evaluation to the list
                existing_provenance["evaluations"].append(evaluation_data)

                # Write updated provenance file
                with open(provenance_path, "w") as f:
                    json.dump(existing_provenance, f, indent=2)

                print(f"Evaluation data added to provenance file: {provenance_path}")
            else:
                print(f"Could not find model directory for {model_name}, skipping evaluation provenance update")
        else:
            print("No model name available, skipping evaluation provenance update")


tlab_evals = EvalsTLabPlugin()
