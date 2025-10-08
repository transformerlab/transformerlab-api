import os

from lab import Experiment, Job
from lab.dirs import get_jobs_dir


def seed_default_experiments():
    """Create a few default experiments if they do not exist (filesystem-backed)."""
    for name in ["alpha", "beta", "gamma"]:
        try:
            exp = Experiment.get(name)
            if not exp:
                Experiment.create(name)
        except Exception:
            # Best-effort seeding; ignore errors (e.g., partial setups)
            pass


def cancel_in_progress_jobs():
    """On startup, mark any RUNNING jobs as CANCELLED in the filesystem job store."""
    jobs_dir = get_jobs_dir()
    if not os.path.exists(jobs_dir):
        return

    for entry in os.listdir(jobs_dir):
        job_path = os.path.join(jobs_dir, entry)
        if os.path.isdir(job_path):
            try:
                job = Job.get(entry)
                if job.get_status() == "RUNNING":
                    job.update_status("CANCELLED")
                    print(f"Cancelled running job: {entry}")
            except Exception:
                # If we can't access the job, continue to the next one
                pass


