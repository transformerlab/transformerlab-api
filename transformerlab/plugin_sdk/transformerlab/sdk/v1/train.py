from transformerlab.sdk.v1.tlab_plugin import TLabPlugin
import os
import time
import traceback
from transformerlab.plugin import test_wandb_login


class TrainerTLabPlugin(TLabPlugin):
    """Enhanced decorator class for TransformerLab training plugins"""

    def __init__(self):
        super().__init__()
        self.tlab_plugin_type = "trainer"
        # Add training-specific default arguments
        self._parser.add_argument("--input_file", type=str, help="Path to configuration file")

        # Training state tracking
        self._config_parsed = False

    def _ensure_args_parsed(self):
        """Parse arguments if not already done"""
        if not self._args_parsed:
            args, _ = self._parser.parse_known_args()
            # Transfer all arguments to attributes of self
            for key, value in vars(args).items():
                self.params[key] = value
            self._args_parsed = True

        if self._args_parsed and not self._config_parsed:
            if getattr(self.params, "input_file") is not None:
                self.load_config()
                self._config_parsed = True

    def create_progress_callback(self, framework="huggingface", **kwargs):
        """
        Create a progress callback for various ML frameworks.

        Args:
            framework: The framework to create a callback for (e.g., "huggingface")
            **kwargs: Additional arguments specific to the callback

        Returns:
            A callback object compatible with the specified framework
        """
        self._ensure_args_parsed()

        if framework.lower() == "huggingface" or framework.lower() == "hf":
            # Import here to avoid dependency issues if HF isn't installed
            try:
                from transformers import TrainerCallback

                class TLabProgressCallback(TrainerCallback):
                    """Callback that updates progress in TransformerLab DB during HuggingFace training"""

                    def __init__(self, tlab_instance):
                        self.tlab = tlab_instance

                    def on_step_end(self, args, state, control, **callback_kwargs):
                        if state.is_local_process_zero:
                            if state.max_steps > 0:
                                progress = state.global_step / state.max_steps
                                progress = int(progress * 100)
                                self.tlab.progress_update(progress)

                                # Check if job should be stopped
                                if self.tlab.job.should_stop:
                                    control.should_training_stop = True

                        return control

                return TLabProgressCallback(self)

            except ImportError:
                raise ImportError("Could not create HuggingFace callback. Please install transformers package.")

        else:
            raise ValueError(f"Unsupported framework: {framework}. Supported frameworks: huggingface")

    def load_config(self):
        """Decorator for loading configuration from input file"""

        try:
            import json

            # Load configuration from file
            with open(self.params.input_file) as json_file:
                input_config = json.load(json_file)
            
            if "config" in input_config:
                self.params._config = input_config["config"]
            else:
                self.params._config = input_config

            # Transfer config values to instance attributes for easy access
            for key, value in self.params._config.items():
                if getattr(self.params, key) is None:
                    self.params[key] = value
            
        except Exception as e:
            error_msg = f"Error loading configuration: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            self.job.set_job_completion_status("failed", "Error loading configuration")
            self.add_job_data("end_time", time.strftime("%Y-%m-%d %H:%M:%S"))
            raise

    def setup_train_logging(self, wandb_project_name: str = "TLab_Training", manual_logging=False):
        """Setup Weights and Biases and TensorBoard logging

        Args:
            wandb_project_name: Name of the W&B project

        Returns:
            List of reporting targets (e.g. ["tensorboard", "wandb"])
        """
        from tensorboardX import SummaryWriter

        self._ensure_args_parsed()
        if not self.params.template_name:
            self.params.template_name = "default"
        # Add tensorboard_output_dir
        self.params.tensorboard_output_dir = os.path.join(
            self.params.output_dir, f"job_{self.params.job_id}_{self.params.template_name}"
        )
        self.writer = SummaryWriter(self.params.tensorboard_output_dir)
        print("Writing tensorboard logs to:", self.params.output_dir)

        # Store the tensorboard output dir in the job
        self.add_job_data("tensorboard_output_dir", self.params.output_dir)

        # Check config or direct attribute for wandb logging preference
        log_to_wandb = False
        if getattr(self.params, "_config") is not None:
            log_to_wandb = self.params._config.get("log_to_wandb", False)
        elif getattr(self.params, "log_to_wandb") is not None:
            log_to_wandb = self.params.log_to_wandb

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
                project=wandb_project_name,
                config=self.params._config,
                name=f"{self.params.template_name}_{self.params.job_id}",
            )

        self.report_to = report_to

    def log_metric(self, metric_name: str, metric_value: float, step: int = None):
        """Log a metric to all reporting targets"""
        if "tensorboard" in self.report_to:
            self.writer.add_scalar(metric_name, metric_value, step)
        if "wandb" in self.report_to and getattr(self, "wandb_run") is not None:
            self.wandb_run.log({metric_name: metric_value}, step=step)


# Create an instance of the TrainerTLabPlugin class
tlab_trainer = TrainerTLabPlugin()
