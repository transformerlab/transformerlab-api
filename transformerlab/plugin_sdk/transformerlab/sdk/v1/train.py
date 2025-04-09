import hashlib
import json
import os
import time
import traceback

try:
    from transformerlab.plugin import WORKSPACE_DIR, generate_model_json, test_wandb_login
    from transformerlab.sdk.v1.tlab_plugin import TLabPlugin

except ModuleNotFoundError:
    from transformerlab.plugin_sdk.transformerlab.plugin import WORKSPACE_DIR, generate_model_json, test_wandb_login
    from transformerlab.plugin_sdk.transformerlab.sdk.v1.tlab_plugin import TLabPlugin


class TrainerTLabPlugin(TLabPlugin):
    """Enhanced decorator class for TransformerLab training plugins"""

    def __init__(self):
        super().__init__()
        self.tlab_plugin_type = "trainer"
        # Add training-specific default arguments
        self._parser.add_argument("--input_file", default=None, type=str, help="Path to configuration file")

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


    def setup_train_logging(self, wandb_project_name: str = "TLab_Training", manual_logging=False, output_dir = None):
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
        if output_dir is None:
            self.params.tensorboard_output_dir = os.path.join(
                self.params.output_dir, f"job_{self.params.job_id}_{self.params.template_name}"
            )
            self.add_job_data("tensorboard_output_dir", self.params.output_dir)
        else:
            self.params.tensorboard_output_dir = os.path.join(
                output_dir, f"job_{self.params.job_id}_{self.params.template_name}"
            )
            self.add_job_data("tensorboard_output_dir", output_dir)

        self.writer = SummaryWriter(self.params.tensorboard_output_dir)
        print("Writing tensorboard logs to:", self.params.output_dir)


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
            

    def create_transformerlab_model(self, fused_model_name, model_architecture, json_data, output_dir=None):
        generate_model_json(fused_model_name, model_architecture, json_data=json_data, output_directory=output_dir)
        print("Generated Model JSON File")
        if output_dir is None:
            fused_model_location = os.path.join(WORKSPACE_DIR, "models", fused_model_name)
        else:
            fused_model_location = os.path.join(output_dir, fused_model_name)

        # Create the hash files for the model
        md5_objects = self.create_md5_checksum_model_files(fused_model_location)

        print("Model Checksum Computed")

        # Create the _tlab_provenance.json file
        provenance_file = self.create_provenance_file(
            model_location=fused_model_location,
            model_name=fused_model_name,
            model_architecture=model_architecture,
            md5_objects=md5_objects,
        )
        print(f"Provenance file created at: {provenance_file}")

    def create_md5_checksum_model_files(self, fused_model_location):
        def compute_md5(file_path):
            md5 = hashlib.md5()
            with open(file_path, "rb") as f:
                while chunk := f.read(8192):
                    md5.update(chunk)
            return md5.hexdigest()

        md5_objects = []

        if not os.path.isdir(fused_model_location):
            print("Fused model location is not a directory, skipping md5 within provenance")
            return md5_objects

        for root, _, files in os.walk(fused_model_location):
            for file in files:
                file_path = os.path.join(root, file)
                md5_hash = compute_md5(file_path)
                md5_objects.append({"file_path": file_path, "md5_hash": md5_hash})

        return md5_objects

    def create_provenance_file(self, model_location, model_name, model_architecture, md5_objects):
        """Create a _tlab_provenance.json file containing model provenance data"""

        # Get training parameters and metadata
        provenance_data = {
            "model_name": model_name,
            "model_architecture": model_architecture,
            "job_id": self.params.get("job_id", None),
            "input_model": self.params.get("model_name", None),
            "dataset": self.params.get("dataset_name", None),
            "adaptor_name": self.params.get("adaptor_name", None),
            "parameters": self.params.get("_config", None),
            "start_time": self.params.get("start_time", ""),
            "end_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "md5_checksums": md5_objects,
        }

        # Write provenance to file
        provenance_path = os.path.join(model_location, "_tlab_provenance.json")
        with open(provenance_path, "w") as f:
            json.dump(provenance_data, f, indent=2)

        print(f"Created provenance file at {provenance_path}")
        return provenance_path


# Create an instance of the TrainerTLabPlugin class
tlab_trainer = TrainerTLabPlugin()
