import copy
import itertools
from typing import Dict, List, Any


class HyperparameterSweep:
    """Class to handle hyperparameter sweep configurations and results"""

    def __init__(self, base_config):
        """Initialize with base configuration that will be used for all runs"""
        self.base_config = base_config
        self.parameters = {}
        self.results = []

    def add_parameter(self, name: str, values: List[Any]):
        """Add a parameter to sweep with its possible values"""
        self.parameters[name] = values

    def generate_configs(self) -> List[Dict[str, Any]]:
        """Generate all possible configurations based on parameters"""
        # Get all parameter combinations
        param_names = list(self.parameters.keys())
        param_values = list(self.parameters.values())

        # Generate all combinations of parameter values
        combinations = list(itertools.product(*param_values))

        # Create configurations with each combination
        configs = []
        for combo in combinations:
            config = copy.deepcopy(self.base_config)
            # Apply this combination of parameters
            for i, name in enumerate(param_names):
                config[name] = combo[i]
            configs.append(config)

        return configs

    def add_result(self, config: Dict[str, Any], metrics: Dict[str, float], status: str = "success"):
        """Add a result from a configuration run"""
        # Create result entry
        result = {"params": {k: config[k] for k in self.parameters.keys()}, "metrics": metrics, "status": status}
        self.results.append(result)

    def get_best_config(self, metric_name: str, lower_is_better: bool = True):
        """Get the best configuration based on a specific metric"""
        # Filter out failed runs
        successful_results = [r for r in self.results if r["status"] == "success"]

        if not successful_results:
            return None

        # Filter out results that don't have the specified metric
        valid_results = [r for r in successful_results if metric_name in r["metrics"]]

        if not valid_results:
            return None

        # Find the best result based on the metric
        if lower_is_better:
            best_result = min(valid_results, key=lambda x: x["metrics"][metric_name])
        else:
            best_result = max(valid_results, key=lambda x: x["metrics"][metric_name])

        return best_result
