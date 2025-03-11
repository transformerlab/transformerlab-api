from transformerlab.tfl_decorators import tfl

# Add any custom arguments your plugin needs
tfl.add_argument("--sample_text", default="Hello, Transformer Lab!", help="Sample text parameter")
tfl.add_argument("--additional_info", default="Sample info", help="Additional information")


@tfl.job_wrapper(progress_start=0, progress_end=100)
@tfl.load_dataset(dataset_types=["train"])
def main(datasets):
    # Convert the dataset to a pandas dataframe
    df = datasets["train"].to_pandas()

    tfl.progress_update(50)

    # Access arguments directly from tfl
    print("Job ID:", tfl.job_id)
    print("Dataset Name:", tfl.dataset_name)

    print("\nDataset Contents:")
    print(df)

    tfl.progress_update(75)

    # Access custom arguments
    print("\nSample Text Parameter:", tfl.sample_text)
    print("Additional Info Parameter:", tfl.additional_info)

    # Optional: Add any job data you want to store
    tfl.add_job_data("sample_data", "This is sample data added to the job")

    return df


if __name__ == "__main__":
    main()
