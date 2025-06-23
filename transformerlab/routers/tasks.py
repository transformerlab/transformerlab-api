import json
from fastapi import APIRouter, Body

import transformerlab.db as db
from transformerlab.models import model_helper

router = APIRouter(prefix="/tasks", tags=["tasks"])


@router.get("/list", summary="Returns all the tasks")
async def tasks_get_all():
    tasks = await db.tasks_get_all()
    return tasks


@router.get("/{task_id}/get", summary="Gets all the data for a single task")
async def tasks_get_by_id(task_id: int):
    tasks = await db.tasks_get_all()
    for task in tasks:
        if task["id"] == task_id:
            return task
    return {"message": "NOT FOUND"}


@router.get("/list_by_type", summary="Returns all the tasks of a certain type, e.g TRAIN")
async def tasks_get_by_type(type: str):
    tasks = await db.tasks_get_by_type(type)
    return tasks


@router.get(
    "/list_by_type_in_experiment", summary="Returns all the tasks of a certain type in a certain experiment, e.g TRAIN"
)
async def tasks_get_by_type_in_experiment(type: str, experiment_id: int):
    tasks = await db.tasks_get_by_type_in_experiment(type, experiment_id)
    return tasks


@router.put("/{task_id}/update", summary="Updates a task with new information")
async def update_task(task_id: int, new_task: dict = Body()):
    await db.update_task(task_id, new_task)
    return {"message": "OK"}


@router.get("/{task_id}/delete", summary="Deletes a task")
async def delete_task(task_id: int):
    await db.delete_task(task_id)
    return {"message": "OK"}


@router.put("/new_task", summary="Create a new task")
async def add_task(new_task: dict = Body()):
    await db.add_task(
        new_task["name"],
        new_task["type"],
        new_task["inputs"],
        new_task["config"],
        new_task["plugin"],
        new_task["outputs"],
        new_task["experiment_id"],
    )
    if new_task["type"] == "TRAIN":
        config = json.loads(new_task["config"])
        # Get the dataset info from the config
        datasets = config.get("_tlab_recipe_datasets", {})
        datasets = datasets.get("path", "")

        # Get the model info from the config
        model = config.get("_tlab_recipe_models", {})
        model_path = model.get("path", "")

        if datasets == "" and model_path == "":
            return {"message": "OK"}

        # Check if the model and dataset are installed
        # For model: get a list of local models to determine what has been downloaded already
        model_downloaded = await model_helper.is_model_installed(model_path)

        # Repeat for dataset
        dataset_downloaded = False
        local_datasets = await db.get_datasets()
        for dataset in local_datasets:
            if dataset["dataset_id"] == datasets:
                dataset_downloaded = True

        # generate a repsonse to tell if model and dataset need to be downloaded
        response = {}

        # Dataset info - including whether it needs to be downloaded or not
        dataset_status = {}
        dataset_status["path"] = datasets
        dataset_status["downloaded"] = dataset_downloaded
        response["dataset"] = dataset_status

        # Model info - including whether it needs to be downloaded or not
        model_status = {}
        model_status["path"] = model_path
        model_status["downloaded"] = model_downloaded
        response["model"] = model_status

        return {"status": "OK", "data": response}

    return {"message": "OK"}


@router.get("/delete_all", summary="Wipe the task table")
async def tasks_delete_all():
    await db.tasks_delete_all()
    return {"message": "OK"}


# These functions convert templates to tasks so that we can do a migration in dev without breaking main for users
# Right now it can do trains, evals, and generates
@router.get("/convert_training_template_to_task", summary="Convert a specific training template to a task")
async def convert_training_template_to_task(template_id: int, experiment_id: int):
    template = await db.get_training_template(template_id)
    template_config = json.loads(template["config"])
    inputs = {}
    if "model_name" in template_config.keys():
        inputs = {
            "model_name": template_config["model_name"],
            "model_architecture": template_config["model_architecture"],
            "dataset_name": template_config["dataset_name"],
        }
    if "embedding_model_name" in template_config.keys():
        inputs = {
            "embedding_model_name": template_config["embedding_model_name"],
            "embedding_model_architecture": template_config["embedding_model_architecture"],
            "dataset_name": template_config["dataset_name"],
        }

    outputs = {}
    if "adaptor_name" in template_config.keys():
        outputs = {"adaptor_name": template_config.get("adaptor_name", "adaptor")}
    try:
        await db.add_task(
            template["name"],
            "TRAIN",
            json.dumps(inputs),
            template["config"],
            template_config["plugin_name"],
            json.dumps(outputs),
            experiment_id,
        )
    except Exception:
        return {"message": "ERROR: unable to convert template to task."}
    return {"message": "OK"}


@router.get("/convert_eval_to_task", summary="Convert a specific eval template to a task")
async def convert_eval_to_task(eval_name: str, experiment_id: int):
    experiment_evaluations = json.loads(json.loads((await db.experiment_get(experiment_id))["config"])["evaluations"])
    for eval in experiment_evaluations:
        if eval["name"] == eval_name:
            await db.add_task(eval["name"], "EVAL", "{}", json.dumps(eval), eval["plugin"], "{}", experiment_id)
    return {"message": "OK"}


@router.get("/convert_generate_to_task", summary="Convert a specific generation template to a task")
async def convert_generate_to_task(generate_name: str, experiment_id: int):
    experiment_generations = json.loads(json.loads((await db.experiment_get(experiment_id))["config"])["generations"])
    for generation in experiment_generations:
        if generation["name"] == generate_name:
            await db.add_task(
                generation["name"], "GENERATE", "{}", json.dumps(generation), generation["plugin"], "{}", experiment_id
            )
    return {"message": "OK"}


# this function is the "convert all" function so that its just 1 api call
@router.get("/{experiment_id}/convert_all_to_tasks", summary="Convert all templates to tasks")
async def convert_all_to_tasks(experiment_id):
    # train templates
    train_templates = await db.get_training_templates()
    for template in train_templates:
        await convert_training_template_to_task(template[0], experiment_id)
    experiment_config = json.loads((await db.experiment_get(experiment_id))["config"])
    # evals
    if "evaluations" in experiment_config.keys():
        experiment_evaluations = json.loads(experiment_config["evaluations"])
        for eval in experiment_evaluations:
            await convert_eval_to_task(eval["name"], experiment_id)
    # generations
    if "generations" in experiment_config:
        experiment_generations = json.loads(experiment_config["generations"])
        for generation in experiment_generations:
            await convert_generate_to_task(generation["name"], experiment_id)
    return {"message": "OK"}


@router.get("/{task_id}/queue", summary="Queue a task to run")
async def queue_task(task_id: int, input_override: str = "{}", output_override: str = "{}"):
    print(
        f"Queueing\n task {task_id} with\n\n input override: {input_override}\n\n and output override: {output_override}"
    )
    task_to_queue = await db.tasks_get_by_id(task_id)
    job_type = task_to_queue["type"]
    job_status = "QUEUED"
    job_data = {}
    # these are the input and output configs from the task
    inputs = json.loads(task_to_queue["inputs"])
    outputs = json.loads(task_to_queue["outputs"])

    # these are the in runtime changes that will override the input and output config from the task
    input_override = json.loads(input_override)
    output_override = json.loads(output_override)
    if job_type == "TRAIN":
        job_data["config"] = json.loads(task_to_queue["config"])
        job_data["model_name"] = inputs["model_name"]
        job_data["dataset"] = inputs["dataset_name"]
        if "type" not in job_data["config"].keys():
            job_data["config"]["type"] = "LoRA"
        # sets the inputs and outputs from the task
        for key in inputs.keys():
            job_data["config"][key] = inputs[key]
        for key in outputs.keys():
            job_data["config"][key] = outputs[key]

        # overrides the inputs and outputs based on the runtime changes requested
        for key in input_override.keys():
            if key == "model_name":
                job_data["model_name"] = input_override["model_name"]
            if key == "dataset":
                job_data["dataset"] = input_override["dataset_name"]
            job_data["config"][key] = input_override[key]
        for key in output_override.keys():
            job_data["config"][key] = output_override[key]

        job_data["template_id"] = task_to_queue["id"]
        job_data["template_name"] = task_to_queue["name"]
    elif job_type == "EVAL":
        job_data["evaluator"] = task_to_queue["name"]
        job_data["config"] = json.loads(task_to_queue["config"])
        for key in inputs.keys():
            job_data["config"][key] = inputs[key]
            job_data["config"]["script_parameters"][key] = inputs[key]
        for key in input_override.keys():
            job_data["config"][key] = input_override[key]
            job_data["config"]["script_parameters"][key] = input_override[key]

        job_data["plugin"] = task_to_queue["plugin"]
    elif job_type == "GENERATE":
        job_data["generator"] = task_to_queue["name"]
        job_data["config"] = json.loads(task_to_queue["config"])
        for key in inputs.keys():
            job_data["config"][key] = inputs[key]
            job_data["config"]["script_parameters"][key] = inputs[key]
        for key in input_override.keys():
            job_data["config"][key] = input_override[key]
            job_data["config"]["script_parameters"][key] = input_override[key]

        for key in outputs.keys():
            job_data["config"][key] = outputs[key]
            job_data["config"]["script_parameters"][key] = outputs[key]
        for key in output_override.keys():
            job_data["config"][key] = output_override[key]
            job_data["config"]["script_parameters"][key] = output_override[key]
        job_data["plugin"] = task_to_queue["plugin"]
    job_id = await db.job_create(job_type, job_status, json.dumps(job_data), task_to_queue["experiment_id"])
    return {"id": job_id}


@router.get("/{task_id}/queue/{machine_id}", summary="Queue a task to run on a remote machine")
async def queue_task_remote(task_id: int, machine_id: int, input_override: str = "{}", output_override: str = "{}"):
    """Queue a task to run on a specific remote machine. Same as local queue but with target machine."""
    import transformerlab.db as db

    # Validate machine exists
    machine = await db.network_machine_get(machine_id)
    if not machine:
        return {"error": "Target machine not found"}

    # Get the task to queue (same as local queue)
    task_to_queue = await db.tasks_get_by_id(task_id)
    job_type = task_to_queue["type"]
    job_status = "QUEUED"
    job_data = {}

    # these are the input and output configs from the task
    inputs = json.loads(task_to_queue["inputs"])
    outputs = json.loads(task_to_queue["outputs"])

    # these are the in runtime changes that will override the input and output config from the task
    input_override = json.loads(input_override)
    output_override = json.loads(output_override)

    # Build job_data exactly the same way as local queue
    if job_type == "TRAIN":
        job_data["config"] = json.loads(task_to_queue["config"])
        job_data["model_name"] = inputs["model_name"]
        job_data["dataset"] = inputs["dataset_name"]
        if "type" not in job_data["config"].keys():
            job_data["config"]["type"] = "LoRA"
        # sets the inputs and outputs from the task
        for key in inputs.keys():
            job_data["config"][key] = inputs[key]
        for key in outputs.keys():
            job_data["config"][key] = outputs[key]

        # overrides the inputs and outputs based on the runtime changes requested
        for key in input_override.keys():
            if key == "model_name":
                job_data["model_name"] = input_override["model_name"]
            if key == "dataset":
                job_data["dataset"] = input_override["dataset_name"]
            job_data["config"][key] = input_override[key]
        for key in output_override.keys():
            job_data["config"][key] = output_override[key]

        job_data["template_id"] = task_to_queue["id"]
        job_data["template_name"] = task_to_queue["name"]
    elif job_type == "EVAL":
        job_data["evaluator"] = task_to_queue["name"]
        job_data["config"] = json.loads(task_to_queue["config"])
        for key in inputs.keys():
            job_data["config"][key] = inputs[key]
            job_data["config"]["script_parameters"][key] = inputs[key]
        for key in input_override.keys():
            job_data["config"][key] = input_override[key]
            job_data["config"]["script_parameters"][key] = input_override[key]

        job_data["plugin"] = task_to_queue["plugin"]
    elif job_type == "GENERATE":
        job_data["generator"] = task_to_queue["name"]
        job_data["config"] = json.loads(task_to_queue["config"])
        for key in inputs.keys():
            job_data["config"][key] = inputs[key]
            job_data["config"]["script_parameters"][key] = inputs[key]
        for key in input_override.keys():
            job_data["config"][key] = input_override[key]
            job_data["config"]["script_parameters"][key] = input_override[key]

        for key in outputs.keys():
            job_data["config"][key] = outputs[key]
            job_data["config"]["script_parameters"][key] = outputs[key]
        for key in output_override.keys():
            job_data["config"][key] = output_override[key]
            job_data["config"]["script_parameters"][key] = output_override[key]
        job_data["plugin"] = task_to_queue["plugin"]

    # Add remote execution info
    job_data["target_machine_id"] = machine_id
    job_data["target_machine_info"] = {
        "name": machine["name"],
        "host": machine["host"],
        "port": machine["port"],
    }

    job_id = await db.job_create(job_type, job_status, json.dumps(job_data), task_to_queue["experiment_id"])
    return {"id": job_id, "target_machine_id": machine_id, "target_machine_name": machine["name"]}


@router.get("/{task_id}/queue_distributed", summary="Queue a distributed training task across multiple machines")
async def queue_distributed_training_task(
    task_id: int,
    machine_ids: str,  # Comma-separated machine IDs
    master_machine_id: int,
    input_override: str = "{}",
    output_override: str = "{}",
    distributed_config: str = "{}",
):
    """
    Queue a training task to run distributed across multiple machines.

    Args:
        task_id: The training task to execute
        machine_ids: Comma-separated list of machine IDs to use
        master_machine_id: Which machine should coordinate the training
        input_override: JSON string with input parameter overrides
        output_override: JSON string with output parameter overrides
        distributed_config: JSON string with distributed training configuration
    """
    import transformerlab.db as db

    try:
        print("RECEIVING DISTRIBUTED QUEUE REQUEST with parameters:")
        print(f"  task_id: {task_id}")
        print(f"  machine_ids: {machine_ids}")
        print(f"  master_machine_id: {master_machine_id}")
        print(f"  input_override: {input_override}")
        print(f"  output_override: {output_override}")
        print(f"  distributed_config: {distributed_config}")
        # Parse machine IDs
        target_machine_ids = [int(mid.strip()) for mid in machine_ids.split(",")]

        if len(target_machine_ids) < 2:
            return {"error": "At least 2 machines required for distributed training"}

        if master_machine_id not in target_machine_ids:
            return {"error": "Master machine must be included in target machines"}

        # Validate all machines exist and are online
        for machine_id in target_machine_ids:
            machine = await db.network_machine_get(machine_id)
            if not machine:
                return {"error": f"Machine {machine_id} not found"}
            if machine["status"] != "online":
                return {"error": f"Machine {machine_id} is not online"}

        # Get the task to queue
        task_to_queue = await db.tasks_get_by_id(task_id)
        if not task_to_queue:
            return {"error": "Task not found"}

        if task_to_queue["type"] != "TRAIN":
            return {"error": "Only training tasks can be distributed"}

        # Parse overrides and config
        input_override = json.loads(input_override)
        output_override = json.loads(output_override)
        distributed_config = json.loads(distributed_config)

        # Build job data (similar to regular queue but with distributed info)
        job_data = {}
        inputs = json.loads(task_to_queue["inputs"])
        outputs = json.loads(task_to_queue["outputs"])

        job_data["config"] = json.loads(task_to_queue["config"])
        job_data["model_name"] = inputs["model_name"]
        job_data["dataset"] = inputs["dataset_name"]

        if "type" not in job_data["config"].keys():
            job_data["config"]["type"] = "LoRA"

        # Apply inputs and outputs from task
        for key in inputs.keys():
            job_data["config"][key] = inputs[key]
        for key in outputs.keys():
            job_data["config"][key] = outputs[key]

        # Apply runtime overrides
        for key in input_override.keys():
            if key == "model_name":
                job_data["model_name"] = input_override["model_name"]
            if key == "dataset":
                job_data["dataset"] = input_override["dataset_name"]
            job_data["config"][key] = input_override[key]
        for key in output_override.keys():
            job_data["config"][key] = output_override[key]

        job_data["template_id"] = task_to_queue["id"]
        job_data["template_name"] = task_to_queue["name"]

        # Add distributed training specific configuration
        job_data["distributed_training"] = True
        job_data["distributed_config"] = {
            "world_size": len(target_machine_ids),
            "master_machine_id": master_machine_id,
            "target_machine_ids": target_machine_ids,
            "backend": distributed_config.get("backend", "nccl"),
            "master_port": distributed_config.get("master_port", 29500),
            **distributed_config,
        }

        # Extract required dependencies (stored in job_data for later use by orchestrator)
        job_data["plugins_required"] = [task_to_queue["plugin"]] if task_to_queue.get("plugin") else []
        job_data["models_required"] = [job_data["model_name"]] if job_data.get("model_name") else []
        job_data["datasets_required"] = [job_data["dataset"]] if job_data.get("dataset") else []

        # Create the job with distributed type
        job_id = await db.job_create(
            type="DISTRIBUTED_TRAIN",
            status="QUEUED",
            job_data=json.dumps(job_data),
            experiment_id=task_to_queue["experiment_id"],
        )

        return {
            "id": job_id,
            "type": "DISTRIBUTED_TRAIN",
            "target_machines": target_machine_ids,
            "master_machine_id": master_machine_id,
            "world_size": len(target_machine_ids),
            "message": f"Distributed training job queued across {len(target_machine_ids)} machines",
        }

    except ValueError as e:
        return {"error": f"Invalid machine IDs format: {str(e)}"}
    except json.JSONDecodeError as e:
        return {"error": f"Invalid JSON in parameters: {str(e)}"}
    except Exception as e:
        return {"error": f"Failed to queue distributed training task: {str(e)}"}


@router.get("/distributed/suggest_machines", summary="Get machine suggestions for distributed training")
async def suggest_machines_for_distributed_training(
    required_gpus: int = 4, model_size_gb: float = 7.0, dataset_size_gb: float = 1.0
):
    """
    Suggest optimal machine combinations for distributed training based on requirements.
    """
    try:
        import httpx

        # Get all online machines
        machines = await db.network_machine_get_all()
        online_machines = [m for m in machines if m.get("status") == "online"]

        if len(online_machines) < 1:
            return {"suggestions": [], "message": "At least 1 online machines required for distributed training"}

        # Get capabilities for each machine
        machine_capabilities = []
        for machine in online_machines:
            try:
                base_url = f"http://{machine['host']}:{machine['port']}"
                headers = {}
                if machine.get("api_token"):
                    headers["Authorization"] = f"Bearer {machine['api_token']}"

                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.get(f"{base_url}/network/capabilities", headers=headers)
                    if response.status_code == 200:
                        caps = response.json()
                        gpu_count = caps.get("ml_frameworks", {}).get("cuda_device_count", 0)
                        gpu_info = caps.get("gpu", [])

                        machine_capabilities.append(
                            {
                                "machine_id": machine["id"],
                                "machine_name": machine["name"],
                                "host": f"{machine['host']}:{machine['port']}",
                                "gpu_count": gpu_count,
                                "gpu_info": gpu_info,
                                "memory_gb": caps.get("resources", {}).get("memory_total_gb", 0),
                                "current_load": caps.get("current_load", {}),
                                "suitable": gpu_count > 0,  # Basic suitability check
                            }
                        )
            except Exception as e:
                print(f"Failed to get capabilities for machine {machine['id']}: {e}")
                continue

        # Generate suggestions
        suggestions = []

        # Sort machines by GPU count (descending)
        suitable_machines = [m for m in machine_capabilities if m["suitable"]]
        suitable_machines.sort(key=lambda x: x["gpu_count"], reverse=True)

        # Suggestion 1: Best 2 machines
        if len(suitable_machines) >= 2:
            best_two = suitable_machines[:2]
            total_gpus = sum(m["gpu_count"] for m in best_two)
            suggestions.append(
                {
                    "name": "High Performance (2 machines)",
                    "machines": best_two,
                    "total_gpus": total_gpus,
                    "estimated_speedup": min(total_gpus / 2, 1.8),  # Diminishing returns
                    "recommended": total_gpus >= required_gpus,
                    "master_suggestion": best_two[0]["machine_id"],
                }
            )

        # Suggestion 2: Best 4 machines (if available)
        if len(suitable_machines) >= 4:
            best_four = suitable_machines[:4]
            total_gpus = sum(m["gpu_count"] for m in best_four)
            suggestions.append(
                {
                    "name": "Maximum Parallelism (4 machines)",
                    "machines": best_four,
                    "total_gpus": total_gpus,
                    "estimated_speedup": min(total_gpus / 2, 3.5),  # Diminishing returns
                    "recommended": total_gpus >= required_gpus * 1.5,
                    "master_suggestion": best_four[0]["machine_id"],
                }
            )

        # Suggestion 3: All available machines
        if len(suitable_machines) > 4:
            total_gpus = sum(m["gpu_count"] for m in suitable_machines)
            suggestions.append(
                {
                    "name": f"All Available ({len(suitable_machines)} machines)",
                    "machines": suitable_machines,
                    "total_gpus": total_gpus,
                    "estimated_speedup": min(total_gpus / 2, 6.0),  # Heavy diminishing returns
                    "recommended": False,  # Usually not recommended due to communication overhead
                    "master_suggestion": suitable_machines[0]["machine_id"],
                }
            )

        print("AVAILABLE MACHINES: ", len(suitable_machines))
        print("TOTAL AVAILABLE GPUS: ", sum(m["gpu_count"] for m in suitable_machines))

        return {
            "suggestions": suggestions,
            "requirements": {
                "required_gpus": required_gpus,
                "model_size_gb": model_size_gb,
                "dataset_size_gb": dataset_size_gb,
            },
            "available_machines": len(suitable_machines),
            "total_available_gpus": sum(m["gpu_count"] for m in suitable_machines),
        }

    except Exception as e:
        return {"error": f"Failed to generate suggestions: {str(e)}"}
