import traceback

from deepeval.synthesizer import Synthesizer
from langchain.text_splitter import RecursiveCharacterTextSplitter

from transformerlab.tfl_decorators import tfl_gen

# Add custom arguments specific to the synthesizer plugin
tfl_gen.add_argument("--num_goldens", default=5, type=int, help="Number of golden examples to generate")


def context_generation(context: str, model, num_goldens: int):
    """Generate data from context using the Synthesizer"""
    print("Splitting context into sentences...")
    # Break the context into sentences
    splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", ". ", " ", ""], chunk_size=256, chunk_overlap=0)
    sentences = splitter.split_text(context)
    sentences = [[sentence] for sentence in sentences]
    print(f"Number of sentences in the context: {len(sentences)}")

    tfl_gen.progress_update(20)

    # Generate goldens from contexts
    print("Generating data from contexts...")
    try:
        synthesizer = Synthesizer(model=model)
        print("Synthesizer initialized successfully")
        tfl_gen.progress_update(30)

        max_goldens_per_context = num_goldens // max(len(sentences), 1)
        synthesizer.generate_goldens_from_contexts(
            contexts=sentences, max_goldens_per_context=max(max_goldens_per_context, 2), include_expected_output=True
        )
        tfl_gen.progress_update(80)

        # Convert the generated data to a pandas dataframe
        df = synthesizer.to_pandas()

        # Rename the column `actual_output` to `output` for consistency
        if "actual_output" in df.columns:
            df.rename(columns={"actual_output": "output"}, inplace=True)

        return df

    except Exception as e:
        print(f"An error occurred while generating data from context: {e}")
        traceback.print_exc()
        raise


@tfl_gen.job_wrapper(progress_start=0, progress_end=100)
def run_generation():
    """Main function to run the synthesizer plugin"""
    print(f"Generation type: {tfl_gen.generation_type}")
    print(f"Model Name: {tfl_gen.generation_model}")

    # Check for context
    if not tfl_gen.context or len(tfl_gen.context.strip()) <= 1:
        print("Context must be provided for generation.")
        raise ValueError("Context must be provided for generation.")

    # Load the model for generation using tfl_gen helper
    trlab_model = tfl_gen.load_evaluation_model()

    print("Model loaded successfully")
    tfl_gen.progress_update(10)

    # Generate data from context
    df = context_generation(tfl_gen.context, trlab_model, getattr(tfl_gen, "num_goldens", 5))

    # Save dataset using tfl_gen helper
    output_file, dataset_name = tfl_gen.save_generated_dataset(
        df,
        {
            "generation_method": "context",
            "num_goldens": getattr(tfl_gen, "num_goldens", 5),
        },
    )

    print(f"Data generated successfully as dataset {dataset_name}")
    print(f"Saved to {output_file}")

    return df


run_generation()
