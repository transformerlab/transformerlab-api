###############
# GENERIC JOBS MODEL
###############
import json
from lab import Job, Experiment

# Allowed job types:
ALLOWED_JOB_TYPES = [
    "TRAIN",
    "DOWNLOAD_MODEL",
    "LOAD_MODEL",
    "TASK",
    "EVAL",
    "EXPORT",
    "UNDEFINED",
    "GENERATE",
    "INSTALL_RECIPE_DEPS",
    "DIFFUSION",
]


async def job_create(type, status, experiment_id, job_data="{}"):
    # check if type is allowed
    if type not in ALLOWED_JOB_TYPES:
        raise ValueError(f"Job type {type} is not allowed")
    
    # Ensure job_data is a dict
    if isinstance(job_data, str):
        try:
            job_data = json.loads(job_data)
        except Exception:
            job_data = {}
    
    # Create experiment if it doesn't exist
    experiment = Experiment(experiment_id)
    
    # Create job through experiment
    job = experiment.create_job()
    job.set_type(type)
    job.update_status(status)
    job.set_job_data(job_data)
    
    return job.id


async def jobs_get_all(experiment_id, type="", status=""):
    exp_obj = Experiment(experiment_id)
    return exp_obj.get_jobs(type, status)


async def jobs_get_all_by_experiment_and_type(experiment_id, job_type):
    exp_obj = Experiment(experiment_id)
    return exp_obj.get_jobs(job_type, "")


async def jobs_get_by_experiment(experiment_id):
    """Get all jobs for a specific experiment"""
    exp_obj = Experiment(experiment_id)
    return exp_obj.get_jobs("", "")


async def job_get(job_id):
    try:
        job = Job.get(job_id)
        return job.get_json_data()
    except Exception:
        return None


async def job_count_running():
    return Job.count_running_jobs()


async def jobs_get_next_queued_job():
    return Job.get_next_queued_job()


async def job_update_status(job_id, status, experiment_id, error_msg=None):
    try:
        job = Job.get(job_id)
        if experiment_id is not None and job.get_experiment_id() != experiment_id:
            return
        job.update_status(status)
        if error_msg:
            job.set_error_message(error_msg)
    except Exception:
        pass


async def job_update(job_id, type, status, experiment_id):
    try:
        job = Job.get(job_id)
        if experiment_id is not None and job.get_experiment_id() != experiment_id:
            return
        job.set_type(type)
        job.update_status(status)
    except Exception:
        pass


async def job_delete_all(experiment_id):
    if experiment_id is not None:
        experiment = Experiment(experiment_id)
        experiment.delete_all_jobs()


async def job_delete(job_id, experiment_id):
    print("Deleting job: " + str(job_id))
    try:
        job = Job.get(job_id)
        if experiment_id is not None and job.get_experiment_id() != experiment_id:
            return
        job.delete()
    except Exception:
        pass


async def job_update_job_data_insert_key_value(job_id, key, value, experiment_id):
    try:
        job = Job.get(job_id)
        if experiment_id is not None and job.get_experiment_id() != experiment_id:
            return
        job.update_job_data_field(key, value)
    except Exception:
        pass


async def job_stop(job_id, experiment_id):
    print("Stopping job: " + str(job_id))
    await job_update_job_data_insert_key_value(job_id, "stop", True, experiment_id)


async def job_update_progress(job_id, progress, experiment_id):
    """
    Update the percent complete for this job.

    progress: int representing percent complete
    """
    try:
        job = Job.get(job_id)
        if experiment_id is not None and job.get_experiment_id() != experiment_id:
            return
        job.update_progress(progress)
    except Exception:
        pass


async def job_update_sweep_progress(job_id, value, experiment_id):
    """
    Update the 'sweep_progress' key in the job_data JSON column for a given job.
    """
    try:
        job = Job.get(job_id)
        if experiment_id is not None and job.get_experiment_id() != experiment_id:
            return
        job.update_sweep_progress(value)
    except Exception:
        pass
