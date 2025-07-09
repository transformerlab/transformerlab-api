import os
import json
from transformerlab.sdk.v1.tlab_plugin import TLabPlugin


class DiffusionTLabPlugin(TLabPlugin):
    """Enhanced decorator class for TransformerLab diffusion plugins"""

    def __init__(self):
        super().__init__()
        self._parser.add_argument("--run_name", default="diffused", type=str, help="Name for the diffusion output")
        self._parser.add_argument("--experiment_name", default="default", type=str, help="Name of the experiment")
        self._parser.add_argument("--diffusion_model", default="local", type=str, help="Diffusion model to use")
        self._parser.add_argument(
            "--diffusion_type", default="txt2img", type=str, help="Type of diffusion task (txt2img, img2img, etc.)"
        )

        self.tlab_plugin_type = "diffusion"

    def _ensure_args_parsed(self):
        if not self._args_parsed:
            args, unknown_args = self._parser.parse_known_args()
            for key, value in vars(args).items():
                self.params[key] = value
            self._parse_unknown_args(unknown_args)
            self._args_parsed = True

    def _parse_unknown_args(self, unknown_args):
        key = None
        for arg in unknown_args:
            if arg.startswith("--"):
                key = arg.lstrip("-")
                self.params[key] = True
            elif key:
                self.params[key] = arg
                key = None

    def save_generated_images(self, metadata: dict = None, suffix: str = None):
        """
        Save metadata file (e.g. for a diffusion image folder run)

        Args:
            metadata: Optional metadata dict
            suffix: Optional file suffix
        """
        self._ensure_args_parsed()

        output_dir = self.get_output_file_path(dir_only=True)
        os.makedirs(output_dir, exist_ok=True)

        if suffix:
            metadata_file = os.path.join(output_dir, f"{self.params.run_name}_{self.params.job_id}_{suffix}.json")
        else:
            metadata_file = os.path.join(output_dir, f"{self.params.run_name}_{self.params.job_id}_metadata.json")

        with open(metadata_file, "w") as f:
            json.dump(metadata or {}, f, indent=2)

        print(f"[DIFFUSION] Metadata saved to: {metadata_file}")
        return metadata_file

    def get_output_file_path(self, suffix="", dir_only=False):
        self._ensure_args_parsed()

        workspace_dir = os.environ.get("_TFL_WORKSPACE_DIR", "./")
        experiment_dir = os.path.join(workspace_dir, "experiments", self.params.experiment_name)
        diffusion_dir = os.path.join(experiment_dir, "diffusion")

        output_dir = os.path.join(diffusion_dir, f"{self.params.run_name}_{self.params.job_id}")
        os.makedirs(output_dir, exist_ok=True)

        if dir_only:
            return output_dir

        if suffix:
            return os.path.join(output_dir, f"{self.params.run_name}_{suffix}")
        else:
            return os.path.join(output_dir, f"{self.params.run_name}.json")


# Global instance (like tlab_gen in generate.py)
tlab_diffusion = DiffusionTLabPlugin()
