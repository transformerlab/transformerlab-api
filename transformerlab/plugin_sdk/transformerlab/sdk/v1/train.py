import hashlib
import json
import os
import time
import traceback
import copy

try:
    from transformerlab.plugin import WORKSPACE_DIR, generate_model_json, test_wandb_login
    from transformerlab.sdk.v1.tlab_plugin import TLabPlugin

except ModuleNotFoundError:
    from transformerlab.plugin_sdk.transformerlab.plugin import WORKSPACE_DIR, generate_model_json, test_wandb_login
    from transformerlab.plugin_sdk.transformerlab.sdk.v1.tlab_plugin import TLabPlugin


class DotDict(dict):
    """Dictionary subclass that allows attribute access to dictionary keys"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


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

        if framework.lower() in ("huggingface", "hf"):
            try:
                from transformers import TrainerCallback
            except ImportError:
                raise ImportError("Could not create HuggingFace callback. Please install transformers package.")

            class TLabProgressCallback(TrainerCallback):
                """Callback that updates progress and logs metrics to metrics.json"""

                def __init__(self, tlab_instance):
                    self.tlab = tlab_instance

                def on_step_end(self, args, state, control, **cb_kwargs):
                    if state.is_local_process_zero and state.max_steps > 0:
                        progress = int(state.global_step / state.max_steps * 100)
                        self.tlab.progress_update(progress)
                        if self.tlab.job.should_stop:
                            control.should_training_stop = True
                    return control

                def on_log(self, args, state, control, logs=None, **cb_kwargs):
                    # Called whenever Trainer.log() is called
                    if state.is_local_process_zero and logs:
                        step = logs.get("step", state.global_step)
                        for name, val in logs.items():
                            # skip the step counter itself
                            if name == "step":
                                continue
                            try:
                                self.tlab.log_metric(name.replace("train_", "train/").replace("eval_", "eval/"), float(val), step, logging_platforms=False)
                            except Exception:
                                pass
                    return control

                def on_evaluate(self, args, state, control, metrics=None, **cb_kwargs):
                    # Called at end of evaluation
                    if state.is_local_process_zero and metrics:
                        for name, val in metrics.items():
                            try:
                                self.tlab.log_metric(name.replace("train_", "train/").replace("eval_", "eval/"), float(val), state.global_step, logging_platforms=False)
                            except Exception:
                                pass
                    return control

            return TLabProgressCallback(self)

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

    def setup_train_logging(self, wandb_project_name: str = "TLab_Training", manual_logging=False, output_dir=None):
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
            print("Writing tensorboard logs to:", self.params.output_dir)
        else:
            self.params.tensorboard_output_dir = os.path.join(
                output_dir, f"job_{self.params.job_id}_{self.params.template_name}"
            )
            self.add_job_data("tensorboard_output_dir", output_dir)
            print("Writing tensorboard logs to:", output_dir)

        self.writer = SummaryWriter(self.params.tensorboard_output_dir)

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

    def log_metric(self, metric_name: str, metric_value: float, step: int = None, logging_platforms: bool = True):
        """Log a metric to all reporting targets"""
        if logging_platforms:
            if "tensorboard" in self.report_to:
                self.writer.add_scalar(metric_name, metric_value, step)
            if "wandb" in self.report_to and getattr(self, "wandb_run") is not None:
                self.wandb_run.log({metric_name: metric_value}, step=step)

        # Store metrics in memory
        if not hasattr(self, "_metrics"):
            self._metrics = {}

        # Store the latest value for each metric
        self._metrics[metric_name] = metric_value

        # Save metrics to a file in the output directory
        try:
            # Ensure output_dir exists
            output_dir = self.params.get("output_dir", "")
            if output_dir and os.path.exists(output_dir):
                # Save metrics to a JSON file
                metrics_path = os.path.join(output_dir, "metrics.json")
                with open(metrics_path, "w") as f:
                    json.dump(self._metrics, f, indent=2)
            else:
                print(f"Output directory not found or not specified: {output_dir}")
        except Exception as e:
            print(f"Error saving metrics to file: {str(e)}")

    def create_transformerlab_model(
        self, fused_model_name, model_architecture, json_data, output_dir=None, generate_json=True
    ):
        if generate_json:
            generate_model_json(fused_model_name, model_architecture, json_data=json_data, output_directory=output_dir)

        if output_dir is None:
            fused_model_location = os.path.join(WORKSPACE_DIR, "models", fused_model_name)
        else:
            fused_model_location = os.path.join(output_dir, fused_model_name)

        # Create the hash files for the model
        md5_objects = self.create_md5_checksum_model_files(fused_model_location)

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
        dataset_name = self.params.get("dataset_name", None)
        if dataset_name is None:
            dataset_name = self.params.get("dataset", None)
        provenance_data = {
            "model_name": model_name,
            "model_architecture": model_architecture,
            "job_id": self.params.get("job_id", None),
            "input_model": self.params.get("model_name", None),
            "dataset": dataset_name,
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

        return provenance_path

    def run_sweep(self, train_function, sweep_config=None):
        """Run a hyperparameter sweep with the specified train function

        Args:
            train_function: Function that performs the training with given parameters
            sweep_config: Dictionary of parameters to sweep (or None to use defaults)

        Returns:
            Dictionary with sweep results and best configuration
        """
        print("Starting hyperparameter sweep")

        # Load dataset once for all runs
        datasets = self.load_dataset()

        # Get sweep parameter definitions
        sweep_config = self.params.get("sweep_config", {})
        if isinstance(sweep_config, str):
            # If sweep_config is a string, assume it's a JSON object
            try:
                sweep_config = json.loads(sweep_config)
            except json.JSONDecodeError:
                print(f"Error decoding JSON: {sweep_config}. Using default sweep configuration.")
                sweep_config = None
        if not sweep_config:
            # Default sweep parameters if none specified
            sweep_config = {
                "learning_rate": [1e-4, 3e-4, 5e-4],
                "lora_alpha": [8, 16, 32],
                "lora_r": [8, 16, 32],
                "lora_dropout": [0.1, 0.2],
            }
            print(f"Using default sweep configuration: {json.dumps(sweep_config, indent=2)}")
        else:
            print(f"Using provided sweep configuration: {json.dumps(sweep_config, indent=2)}")

        # Import here to avoid circular imports
        from transformerlab.sdk.v1.sweep import HyperparameterSweep

        # Setup the sweeper with a copy of current parameters
        base_params = copy.deepcopy(self.params)
        print("Base parameters for sweep:")
        print(json.dumps({k: v for k, v in base_params.items() if not k.startswith("_")}, indent=2))
        sweeper = HyperparameterSweep(base_params)

        # Add parameters to sweep
        for param_name, values in sweep_config.items():
            sweeper.add_parameter(param_name, values)

        # Generate all configurations
        configs = sweeper.generate_configs()
        total_configs = len(configs)

        print(f"Generated {total_configs} configurations for sweep")

        # Create a directory for sweep results
        sweep_dir = os.path.join(self.params.output_dir, f"sweep_{self.params.job_id}")
        os.makedirs(sweep_dir, exist_ok=True)

        # Setup logging for the sweep
        sweep_log_path = os.path.join(sweep_dir, "sweep_results.json")
        self.add_job_data("sweep_log_path", str(sweep_log_path))
        self.add_job_data("sweep_configs", str(total_configs))

        # Store original params
        original_params = self.params

        # Run each configuration
        for i, config in enumerate(configs):
            run_id = f"run_{i + 1}_of_{total_configs}"

            # Update job progress based on sweep progress
            overall_progress = int((i / total_configs) * 100)
            self.progress_update(overall_progress)

            print(f"\n--- Starting sweep {run_id} ---")
            print(f"Configuration: {json.dumps({k: config[k] for k in sweep_config.keys()}, indent=2)}")

            # Create run-specific output directories
            run_output_dir = os.path.join(sweep_dir, run_id)
            run_adaptor_dir = os.path.join(run_output_dir, "adaptor")
            os.makedirs(run_output_dir, exist_ok=True)
            os.makedirs(run_adaptor_dir, exist_ok=True)

            try:
                # Create a new params object for this run
                run_params = copy.deepcopy(config)
                run_params["output_dir"] = run_output_dir
                run_params["adaptor_output_dir"] = run_adaptor_dir
                run_params["run_id"] = run_id
                run_params["datasets"] = datasets

                # Replace the params temporarily
                self.params = DotDict(run_params)

                # Run training with this configuration
                metrics = train_function(**run_params)

                # Add result to sweeper
                sweeper.add_result(config, metrics)

                print(f"Run {i + 1} completed with metrics: {json.dumps(metrics, indent=2)}")

            except Exception as e:
                error_msg = f"Error in sweep run {i + 1}: {str(e)}\n{traceback.format_exc()}"
                print(error_msg)
                sweeper.add_result(config, {}, "failed")
            finally:
                # Restore original params
                self.params = original_params

            # Save intermediate sweep results
            with open(sweep_log_path, "w") as f:
                json.dump(
                    {
                        "sweep_config": sweep_config,
                        "results": sweeper.results,
                        "completed_runs": i + 1,
                        "total_runs": total_configs,
                    },
                    f,
                    indent=2,
                )

        # Find best configuration
        metric_name = self.params.get("sweep_metric", "eval/loss")
        lower_is_better = self.params.get("lower_is_better") is not None
        best_result = sweeper.get_best_config(metric_name, lower_is_better)

        self.params["train_final_model"] = True

        if best_result:
            print("\n--- Sweep completed ---")
            print("Best configuration:")
            print(json.dumps(best_result["params"], indent=2))
            print("Metrics:")
            print(json.dumps(best_result["metrics"], indent=2))

            # Add best result to job data
            self.add_job_data("best_config", str(best_result["params"]))
            self.add_job_data("best_metrics", str(best_result["metrics"]))

        # Return all results
        return {
            "sweep_results": sweeper.results,
            "best_config": best_result["params"] if best_result else None,
            "best_metrics": best_result["metrics"] if best_result else None,
            "sweep_log_path": sweep_log_path,
        }


# Create an instance of the TrainerTLabPlugin class
tlab_trainer = TrainerTLabPlugin()
