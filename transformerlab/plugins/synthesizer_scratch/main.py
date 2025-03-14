import json

from deepeval.synthesizer import Evolution, Synthesizer
from deepeval.synthesizer.config import EvolutionConfig, StylingConfig

# Import tfl_gen instead of using argparse
from transformerlab.tfl_decorators import tfl_gen

tfl_gen.add_argument("--num_goldens", default=5, type=int)


def scratch_generation(model, styling_config: dict, evolution_config: dict = None):
    """Generate synthetic data from scratch"""
    # Validate configs
    if not all(key in styling_config for key in ["input_format", "expected_output_format", "task", "scenario"]):
        raise ValueError(
            "Styling config dictionary must have the keys `input_format`, `expected_output_format`, `task`, `scenario`"
        )

    if evolution_config is not None and not all(
        key in evolution_config for key in ["REASONING", "CONCRETIZING", "CONSTRAINED"]
    ):
        raise ValueError("Evolution config dictionary must have the keys `REASONING`, `CONCRETIZING`, `CONSTRAINED`")

    print("Generating data from scratch...")
    tfl_gen.progress_update(35)

    try:
        # Create StylingConfig
        styling_config = StylingConfig(**styling_config)

        # Create EvolutionConfig
        if not evolution_config:
            evolution_config = EvolutionConfig(
                evolutions={Evolution.REASONING: 1 / 3, Evolution.CONCRETIZING: 1 / 3, Evolution.CONSTRAINED: 1 / 3},
                num_evolutions=3,
            )
        else:
            evolution_config = EvolutionConfig(
                evolutions={
                    Evolution.REASONING: evolution_config["REASONING"],
                    Evolution.CONCRETIZING: evolution_config["CONCRETIZING"],
                    Evolution.CONSTRAINED: evolution_config["CONSTRAINED"],
                },
                num_evolutions=3,
            )

        # Initialize synthesizer
        synthesizer = Synthesizer(styling_config=styling_config, model=model, evolution_config=evolution_config)
        tfl_gen.progress_update(45)
        print("Synthesizer initialized successfully")

        # Generate data
        synthesizer.generate_goldens_from_scratch(num_goldens=tfl_gen.num_goldens)
        tfl_gen.progress_update(60)

        # Convert to DataFrame
        df = synthesizer.to_pandas()
        return df

    except Exception as e:
        print(f"An error occurred while generating data from scratch: {e}")
        raise


@tfl_gen.job_wrapper(progress_start=0, progress_end=100)
def run_generation():
    """Run data generation using Synthesizer"""
    # Setup and initialize

    # Load model
    try:
        trlab_model = tfl_gen.load_evaluation_model(field_name="generation_model")
        print(f"Model loaded successfully: {trlab_model.get_model_name()}")
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        raise

    tfl_gen.progress_update(20)

    # Check required parameters
    if not tfl_gen.input_format or not tfl_gen.expected_output_format or not tfl_gen.task or not tfl_gen.scenario:
        raise ValueError("Input format, expected output format, task and scenario must be provided for generation.")

    # Create styling config
    styling_config = {
        "input_format": tfl_gen.input_format,
        "expected_output_format": tfl_gen.expected_output_format,
        "task": tfl_gen.task,
        "scenario": tfl_gen.scenario,
    }

    # Parse evolution config if provided
    evolution_config = None
    if hasattr(tfl_gen, "evolution_config") and tfl_gen.evolution_config:
        try:
            evolution_config = json.loads(tfl_gen.evolution_config)
        except json.JSONDecodeError:
            print("Warning: Invalid JSON in evolution_config, using default")

    tfl_gen.progress_update(30)

    # Call synthesizer function
    df = scratch_generation(trlab_model, styling_config, evolution_config)

    tfl_gen.progress_update(70)

    # Generate expected outputs if requested
    if getattr(tfl_gen, "generate_expected_output", "Yes").lower() == "yes":
        input_values = df["input"].tolist()
        expected_outputs = tfl_gen.generate_expected_outputs(
            input_values,
            styling_config["task"],
            styling_config["scenario"],
            styling_config["input_format"],
            styling_config["expected_output_format"],
        )
        df["expected_output"] = expected_outputs

    # Rename columns for consistency
    df.rename(columns={"actual_output": "output"}, inplace=True)

    tfl_gen.progress_update(90)

    # Save dataset
    output_file, dataset_name = tfl_gen.save_generated_dataset(
        df, {"styling_config": styling_config, "evolution_config": evolution_config}
    )

    tfl_gen.progress_update(100)
    print(f"Data generated successfully as dataset {dataset_name}")

    return True


print("Running generation...")
run_generation()
