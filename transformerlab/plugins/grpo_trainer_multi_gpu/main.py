import os
import time
import re
import subprocess

from transformerlab.tfl_decorators import tfl_trainer

# Add custom arguments
tfl_trainer.add_argument(
    "--launched_with_accelerate", action="store_true", help="Flag to prevent recursive subprocess launching"
)


def setup_accelerate_environment():
    """Set up the environment for the accelerate launch subprocess"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    api_dir = os.path.abspath(os.path.join(current_dir, "../../.."))
    env = os.environ.copy()
    tfl_source_dir = os.environ.get("_TFL_SOURCE_CODE_DIR")
    python_path = env.get("PYTHONPATH", "")
    paths_to_include = [api_dir]

    if tfl_source_dir:
        tflab_sdk_path = os.path.join(tfl_source_dir, "transformerlab", "plugin_sdk")
        paths_to_include.append(tflab_sdk_path)
        plugin_parent = os.path.join(tfl_source_dir, "transformerlab")
        paths_to_include.append(plugin_parent)

    if python_path:
        paths_to_include.append(python_path)

    env["PYTHONPATH"] = ":".join(paths_to_include)
    return env


def extract_answer(text: str, start_answer_string, end_answer_string) -> str:
    """Extract the answer from the text between start and end tags"""
    answer = text.split(f"{start_answer_string}")[-1]
    answer = answer.split(f"{end_answer_string}")[0]
    return answer.strip()


def count_xml(text, start_thinking_string, end_thinking_string, start_answer_string, end_answer_string) -> float:
    """Count XML tags in the response"""
    count = 0.0
    if text.count(f"{start_thinking_string}\n") == 1:
        count += 0.125
    if text.count(f"\n{end_thinking_string}\n") == 1:
        count += 0.125
    if text.count(f"\n{start_answer_string}\n") == 1:
        count += 0.125
        count -= len(text.split(f"\n{end_answer_string}\n")[-1]) * 0.001
    if text.count(f"\n{end_answer_string}") == 1:
        count += 0.125
        count -= (len(text.split(f"\n{end_answer_string}")[-1]) - 1) * 0.001
    return count


@tfl_trainer.job_wrapper(progress_start=0, progress_end=100)
def train_model():
    """Main training function using TrainerTFLPlugin"""
    # Get the dataset from the datasets dict loaded by the decorator
    datasets = tfl_trainer.load_dataset(dataset_types=["train"])
    dataset = datasets["train"]

    # Set up accelerate configuration
    accelerate_config = {
        "cuda": "multi_gpu",
        "cpu": "cpu",
        "tpu": "tpu",
    }

    train_device = accelerate_config.get(tfl_trainer.train_device, "multi_gpu")
    print(f"Training setup for accelerate launch: {train_device}")

    # Configure GPU IDs
    gpu_ids = None
    if train_device == "multi_gpu":
        gpu_ids = tfl_trainer.gpu_ids
        if gpu_ids and gpu_ids != "auto":
            gpu_ids = str(gpu_ids)
        if gpu_ids == "auto":
            gpu_ids = None

    # Check if we need to launch with accelerate
    if not getattr(tfl_trainer, "launched_with_accelerate", False):
        print("Launching training with accelerate for multi-GPU...")
        env = setup_accelerate_environment()

        cmd = [
            "accelerate",
            "launch",
            f"--{train_device}",
            __file__,
            "--input_file",
            tfl_trainer.input_file,
            "--launched_with_accelerate",
        ]
        if gpu_ids:
            cmd.extend(["--gpu_ids", gpu_ids])

        result = subprocess.run(cmd, env=env)
        print(f"Subprocess completed with return code: {result.returncode}")
        return "Training process launched"

    # Import dependencies after the subprocess check
    import torch
    from jinja2 import Environment
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer
    from accelerate import Accelerator

    # Initialize Accelerator
    accelerator = Accelerator()
    print(f"Running with accelerate on {accelerator.num_processes} processes")

    jinja_environment = Environment()

    # Get configuration values from tfl_trainer
    model_id = tfl_trainer.model_name
    max_completion_length = int(tfl_trainer.maximum_completion_length)
    learning_rate = float(tfl_trainer.learning_rate)
    learning_rate_schedule = tfl_trainer.learning_rate_schedule
    max_grad_norm = float(tfl_trainer.max_grad_norm)
    batch_size = int(tfl_trainer.batch_size)
    num_epochs = int(tfl_trainer.num_train_epochs)
    weight_decay = float(tfl_trainer.weight_decay)
    adam_beta1 = float(tfl_trainer.adam_beta1)
    adam_beta2 = float(tfl_trainer.adam_beta2)
    adam_epsilon = float(tfl_trainer.adam_epsilon)
    output_dir = tfl_trainer.output_dir

    # Get the template strings
    question_formatting_template = getattr(tfl_trainer, "input_template", "")
    answer_formatting_template = getattr(tfl_trainer, "output_template", "")
    instruction_template = getattr(tfl_trainer, "instruction_template", "")

    start_thinking_string = getattr(tfl_trainer, "start_thinking_string", "<reasoning>")
    end_thinking_string = getattr(tfl_trainer, "end_thinking_string", "</reasoning>")
    start_answer_string = getattr(tfl_trainer, "start_answer_string", "<answer>")
    end_answer_string = getattr(tfl_trainer, "end_answer_string", "</answer>")

    # Determine if the instruction template is missing the necessary strings
    if start_thinking_string not in instruction_template or start_answer_string not in instruction_template:
        system_prompt = f"""
        Respond in the following format:
            {start_thinking_string}
            ...
            {end_thinking_string}
            {start_answer_string}
            ...
            {end_answer_string}
        """
    else:
        system_prompt = instruction_template

    # Define format_instruction function
    def format_instruction(template, mapping):
        return template.render(mapping)

    question_template = jinja_environment.from_string(question_formatting_template)
    answer_template = jinja_environment.from_string(answer_formatting_template)

    # Process the dataset
    dataset = dataset.map(
        lambda x: {
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": format_instruction(question_template, x)},
            ],
            "answer": format_instruction(answer_template, x).split("#### ")[-1],
        }
    )

    # Define reward functions with closure to capture config variables
    def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
        responses = [completion[0]["content"] for completion in completions]
        extracted_responses = [extract_answer(r, start_answer_string, end_answer_string) for r in responses]
        return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

    def xmlcount_reward_func(completions, **kwargs) -> list[float]:
        contents = [completion[0]["content"] for completion in completions]
        return [
            count_xml(c, start_thinking_string, end_thinking_string, start_answer_string, end_answer_string)
            for c in contents
        ]

    def extract_xml_answer(text: str) -> str:
        return extract_answer(text, start_answer_string, end_answer_string)

    def int_reward_func(completions, **kwargs) -> list[float]:
        """Reward function that checks if the answer is a number"""
        responses = [completion[0]["content"] for completion in completions]
        extracted_responses = [extract_xml_answer(r) for r in responses]
        return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]

    def strict_format_reward_func(completions, **kwargs) -> list[float]:
        """Reward function that checks strictly if the completion has a specific format."""
        pattern = (
            rf"^{start_thinking_string}\n.*?\n{end_thinking_string}\n{start_answer_string}\n.*?\n{end_answer_string}\n$"
        )
        responses = [completion[0]["content"] for completion in completions]
        matches = [re.match(pattern, r) for r in responses]
        return [0.5 if match else 0.0 for match in matches]

    def soft_format_reward_func(completions, **kwargs) -> list[float]:
        """Reward function that checks if the completion has a specific format."""
        pattern = rf"{start_thinking_string}.*?{end_thinking_string}\s*{start_answer_string}.*?{end_answer_string}"
        responses = [completion[0]["content"] for completion in completions]
        matches = [re.match(pattern, r) for r in responses]
        return [0.5 if match else 0.0 for match in matches]

    # Load model and tokenizer
    try:
        device_map = None if accelerator.num_processes > 1 else "auto"

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.padding_side = "right"
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map=device_map)
    except Exception as e:
        return f"Failed to load model: {str(e)}"

    # Training run name
    today = time.strftime("%Y%m%d-%H%M%S")
    run_suffix = getattr(tfl_trainer, "template_name", today)

    # GRPO training configuration
    args = GRPOConfig(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        weight_decay=weight_decay,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=2,
        gradient_checkpointing=True,
        optim="paged_adamw_32bit",
        logging_steps=10,
        save_strategy="epoch",
        learning_rate=learning_rate,
        bf16=True,
        tf32=True,
        max_grad_norm=max_grad_norm,
        warmup_ratio=0.03,
        max_completion_length=max_completion_length,
        lr_scheduler_type=learning_rate_schedule,
        adam_beta1=adam_beta1,
        adam_beta2=adam_beta2,
        adam_epsilon=adam_epsilon,
        disable_tqdm=False,
        run_name=f"job_{tfl_trainer.job_id}_{run_suffix}",
        report_to=tfl_trainer.report_to,
        ddp_find_unused_parameters=False,
        dataloader_pin_memory=True,
        no_cuda=False,
    )

    # Create progress callback
    progress_callback = tfl_trainer.create_progress_callback(framework="huggingface")

    # Initialize GRPO trainer
    trainer = GRPOTrainer(
        model=model,
        train_dataset=dataset,
        reward_funcs=[
            xmlcount_reward_func,
            correctness_reward_func,
            int_reward_func,
            strict_format_reward_func,
            soft_format_reward_func,
        ],
        args=args,
        processing_class=tokenizer,
        callbacks=[progress_callback],
    )

    # Train the model
    try:
        trainer.train()
    except Exception as e:
        return f"Training failed: {str(e)}"

    # Save the model
    try:
        trainer.save_model(output_dir=tfl_trainer.adaptor_output_dir)
    except Exception as e:
        return f"Failed to save model: {str(e)}"

    # Return success message
    return "Adaptor trained successfully"


train_model()
