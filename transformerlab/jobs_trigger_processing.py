"""
Logic for processing triggers when jobs are completed.

This module processes triggers for completed jobs using the new per-workflow trigger system.
Each workflow has its own trigger_configs that define which job completion events should activate it.
"""

import logging
import transformerlab.db as db

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def process_job_completion_triggers(job_id: int):
    """
    Process triggers for a single completed job (event-based).
    
    This function is called when a job is marked as COMPLETE and:
    1. Gets the job details
    2. Maps the job type to trigger type
    3. Finds any enabled triggers for that type in the job's experiment
    4. For each enabled trigger, queues the associated workflows
    5. Marks the job as having had its triggers processed
    """
    try:
        # Step 1: Get job details
        job = await db.job_get(job_id)
        if not job:
            logger.warning(f"Job {job_id} not found, skipping trigger processing")
            return

        if job["status"] != "COMPLETE":
            logger.debug(f"Job {job_id} is not complete (status: {job['status']}), skipping trigger processing")
            return

        # Step 2: Check if triggers have already been processed
        job_data = job.get("job_data", {})
        if job_data.get("triggers_processed", False):
            logger.debug(f"Job {job_id} triggers already processed, skipping")
            return

        job_type = job["type"]
        experiment_id = job["experiment_id"]

        if not experiment_id:
            logger.warning(f"Job {job_id} has no experiment_id, skipping trigger processing")
            await db.job_update_job_data_insert_key_value(job_id, "triggers_processed", True)
            return

        # Step 3: Map job type to trigger type (they match for predefined blueprints)
        trigger_type = job_type  # Direct mapping

        logger.info(f"Processing triggers for completed {job_type} job {job_id} in experiment {experiment_id}")

        # Step 4: Find workflows that should be triggered by this job event
        matching_workflows = await db.workflow_get_by_job_event(job_type, experiment_id, job_name=None)

        if not matching_workflows:
            logger.info(f"No workflows found with enabled {trigger_type} triggers in experiment {experiment_id}")
            await db.job_update_job_data_insert_key_value(job_id, "triggers_processed", True)
            return

        logger.info(f"Found {len(matching_workflows)} workflows to trigger for job {job_id}")

        # Step 5: Queue each matching workflow
        for workflow in matching_workflows:
            try:
                await db.workflow_queue(workflow["id"])
                logger.info(f"Job {job_id} queued workflow {workflow['id']} ({workflow['name']})")
            except Exception as e:
                logger.error(f"Error queuing workflow {workflow['id']}: {str(e)}")

        # Step 6: Mark job as processed
        await db.job_update_job_data_insert_key_value(job_id, "triggers_processed", True)
        logger.info(f"Job {job_id} trigger processing completed")

    except Exception as e:
        # Log the error but don't crash the application
        logger.error(f"Error processing triggers for job {job_id}: {str(e)}", exc_info=True)