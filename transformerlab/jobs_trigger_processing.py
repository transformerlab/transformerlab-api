"""
Logic for processing triggers when jobs are completed.

This module processes triggers for completed jobs based on the trigger types defined in db.TRIGGERS_TO_SEED.
"""

import logging
from typing import Dict, Any
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

        # Step 3: Map job type to trigger type (they should match based on TRIGGERS_TO_SEED)
        trigger_type = job_type  # Direct mapping

        logger.info(f"Processing triggers for completed {job_type} job {job_id} in experiment {experiment_id}")

        # Step 4: Find active triggers for this experiment and type
        active_triggers = await db.workflow_trigger_get_enabled_by_experiment_id_and_type(
            experiment_id, trigger_type
        )

        if not active_triggers:
            logger.info(f"No active triggers found for {trigger_type} in experiment {experiment_id}")
            await db.job_update_job_data_insert_key_value(job_id, "triggers_processed", True)
            return

        logger.info(f"Found {len(active_triggers)} active triggers for job {job_id}")

        # Step 5: Process each active trigger
        for trigger in active_triggers:
            await process_trigger_for_job(job_id, trigger)

        # Step 6: Mark job as processed
        await db.job_update_job_data_insert_key_value(job_id, "triggers_processed", True)
        logger.info(f"Job {job_id} trigger processing completed")

    except Exception as e:
        # Log the error but don't crash the application
        logger.error(f"Error processing triggers for job {job_id}: {str(e)}", exc_info=True)


async def process_completed_job_triggers():
    """
    Legacy polling-based trigger processing function.
    
    This function is kept for backward compatibility but should be replaced
    with the event-based process_job_completion_triggers() function.
    
    This function:
    1. Gets all trigger types from TRIGGERS_TO_SEED
    2. For each trigger type, fetches completed jobs of that type that haven't had their triggers processed
    3. For each job, finds any enabled triggers of the matching type in its experiment
    4. For each enabled trigger, queues the associated workflows
    5. Marks the job as having had its triggers processed
    """
    # Get unique trigger types from TRIGGERS_TO_SEED
    trigger_types = {trigger["trigger_type"] for trigger in db.TRIGGERS_TO_SEED}
    
    for trigger_type in trigger_types:
        try:
            # Step 1: Fetch unprocessed completed jobs for this trigger type
            cursor = await db.db.execute(
                "SELECT * FROM job WHERE type = ? AND status = 'COMPLETE' AND (json_extract(job_data, '$.triggers_processed') IS NULL OR json_extract(job_data, '$.triggers_processed') = 0)",
                (trigger_type,)
            )
            jobs = await cursor.fetchall()

            # Convert rows to dicts
            desc = cursor.description
            column_names = [col[0] for col in desc]
            job_dicts = [dict(zip(column_names, job)) for job in jobs]
            await cursor.close()

            if not job_dicts:
                # No jobs to process for this trigger type
                continue

            logger.info(f"Found {len(job_dicts)} completed {trigger_type} jobs with unprocessed triggers")

            # Step 2: Process each job using the new event-based function
            for job in job_dicts:
                await process_job_completion_triggers(job["id"])

        except Exception as e:
            # Log the error but don't crash the application
            logger.error(f"Error processing {trigger_type} job triggers: {str(e)}", exc_info=True)


async def process_trigger_for_job(job_id: int, trigger: Dict[str, Any]):
    """Process a single trigger for a job."""
    trigger_id = trigger["id"]
    
    # Check that the config and workflow_ids exist
    if not trigger["config"] or "workflow_ids" not in trigger["config"]:
        logger.warning(f"Trigger {trigger_id} has no workflow_ids configured, skipping")
        return
    
    workflow_ids = trigger["config"]["workflow_ids"]
    if not isinstance(workflow_ids, list) or not workflow_ids:
        logger.warning(f"Trigger {trigger_id} has invalid workflow_ids: {workflow_ids}, skipping")
        return
    
    # Queue each workflow
    for workflow_id in workflow_ids:
        try:
            await db.workflow_queue(workflow_id)
            logger.info(f"Job {job_id} queued workflow {workflow_id} via trigger {trigger_id}")
        except Exception as e:
            logger.error(f"Error queuing workflow {workflow_id} via trigger {trigger_id}: {str(e)}")
            # Continue processing other workflows 