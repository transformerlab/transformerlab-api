import itertools
import copy
import json
import os
import time
from typing import Dict, List, Any, Optional, Union

class HyperparameterSweep:
    """Simple hyperparameter sweep manager for TransformerLab"""
    
    def __init__(self, base_params: Dict[str, Any]):
        """
        Initialize with base configuration
        
        Args:
            base_params: Dictionary containing the base training parameters
        """
        self.base_params = base_params
        self.sweep_params = {}
        self.results = []
        
    def add_parameter(self, param_name: str, values: List[Any]):
        """
        Add a parameter to sweep over
        
        Args:
            param_name: Name of the parameter in the config
            values: List of values to try for this parameter
        """
        self.sweep_params[param_name] = values
        return self
        
    def generate_configs(self):
        """
        Generate all parameter combinations
        
        Returns:
            List of configuration dictionaries
        """
        param_names = list(self.sweep_params.keys())
        param_values = list(self.sweep_params.values())
        
        configs = []
        for combo in itertools.product(*param_values):
            config = copy.deepcopy(self.base_params)
            
            # Create unique name for this run
            param_str = "_".join([f"{name}={value}" for name, value in zip(param_names, combo)])
            
            # Apply this combination of parameters
            for name, value in zip(param_names, combo):
                config[name] = value
                
            if config.get("template_name"):
                config["template_name"] = f"{config['template_name']}_sweep_{param_str}"
            else:
                config["template_name"] = f"sweep_{param_str}"
                
            configs.append(config)
            
        return configs
    
    def add_result(self, config: Dict[str, Any], metrics: Dict[str, float], status: str = "success"):
        """
        Add a result from a sweep run
        
        Args:
            config: Configuration used for this run
            metrics: Dictionary of evaluation metrics
            status: Status of the run ('success' or 'failed')
        """
        result = {
            "params": {k: config[k] for k in self.sweep_params.keys() if k in config},
            "metrics": metrics,
            "status": status,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        self.results.append(result)
        
    def get_best_config(self, metric_name: str = "eval/loss", lower_is_better: bool = True):
        """
        Get the best configuration based on a metric
        
        Args:
            metric_name: Name of the metric to optimize
            lower_is_better: Whether lower values are better (True for loss, False for accuracy)
            
        Returns:
            Dictionary with the best configuration or None if no valid results
        """
        valid_results = [r for r in self.results if r["status"] == "success" and 
                         metric_name in r["metrics"]]
        
        if not valid_results:
            return None
            
        if lower_is_better:
            best_result = min(valid_results, key=lambda x: x["metrics"][metric_name])
        else:
            best_result = max(valid_results, key=lambda x: x["metrics"][metric_name])
            
        return best_result
        
    def save_results(self, output_path: str):
        """
        Save the sweep results to a JSON file
        
        Args:
            output_path: Path to save the results file
        """
        with open(output_path, "w") as f:
            json.dump({
                "sweep_params": self.sweep_params,
                "results": self.results,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }, f, indent=2)