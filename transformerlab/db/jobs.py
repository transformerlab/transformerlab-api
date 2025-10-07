###############
# GENERIC JOBS MODEL
# Have to temporarily leave these here as there is a conflict with anothe rfunction in job service
###############
from lab import Job


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
