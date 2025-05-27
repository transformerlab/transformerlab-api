# Workflow Trigger System Implementation Summary

## Overview
This implementation provides a per-workflow trigger configuration system that allows users to toggle 6 predefined trigger types on/off for each workflow independently. When a job completes in an experiment, any workflows in that same experiment with the corresponding trigger enabled will be automatically queued.

## Core Components

### 1. Predefined Trigger Blueprints (`transformerlab/db.py`)
```python
PREDEFINED_TRIGGER_BLUEPRINTS = [
    {
        "trigger_type": "TRAIN",
        "name": "On Training Completed",
        "description": "Starts the workflow when a training job in its experiment completes.",
        "default_is_enabled": False
    },
    # ... 5 more trigger types: DOWNLOAD_MODEL, LOAD_MODEL, EXPORT_MODEL, EVAL, GENERATE
]
```

### 2. Database Schema Changes

#### Workflow Model (`transformerlab/shared/models/models.py`)
- Added `trigger_configs: Mapped[Optional[list[dict]]] = mapped_column(JSON, nullable=True)`

#### Database Migration (`transformerlab/db.py`)
- Added `trigger_configs` column to workflows table
- Migration runs automatically on startup

### 3. Data Structure
Each workflow's `trigger_configs` field contains:
```json
[
    {"trigger_type": "TRAIN", "is_enabled": true},
    {"trigger_type": "DOWNLOAD_MODEL", "is_enabled": false},
    {"trigger_type": "LOAD_MODEL", "is_enabled": false},
    {"trigger_type": "EXPORT_MODEL", "is_enabled": false},
    {"trigger_type": "EVAL", "is_enabled": false},
    {"trigger_type": "GENERATE", "is_enabled": false}
]
```

## API Endpoints

### 1. Get Trigger Blueprints
```
GET /workflows/trigger_blueprints
```
Returns the list of available trigger types with descriptions.

### 2. Get Workflow with Trigger Configs
```
GET /workflows/{workflow_id}
```
Returns workflow details including normalized trigger_configs.

### 3. Update Workflow Trigger Configs
```
PUT /workflows/{workflow_id}/trigger_configs
```
**Request Body:**
```json
{
    "configs": [
        {"trigger_type": "TRAIN", "is_enabled": true},
        {"trigger_type": "DOWNLOAD_MODEL", "is_enabled": false},
        {"trigger_type": "LOAD_MODEL", "is_enabled": false},
        {"trigger_type": "EXPORT_MODEL", "is_enabled": false},
        {"trigger_type": "EVAL", "is_enabled": false},
        {"trigger_type": "GENERATE", "is_enabled": false}
    ]
}
```

## Database Functions

### Core Functions
- `workflow_create()` - Initializes new workflows with default trigger configs
- `workflows_get_by_id()` - Returns workflow with normalized trigger configs
- `workflows_get_all()` - Returns all workflows with normalized trigger configs
- `workflows_get_from_experiment()` - Returns experiment workflows with normalized trigger configs
- `workflow_update_trigger_configs()` - Updates trigger configurations with validation
- `workflow_get_by_job_event()` - Finds workflows to trigger based on job completion

### Helper Functions
- `_normalize_trigger_configs()` - Ensures trigger configs are complete and valid

## Trigger Processing

### Event-Based Processing (`transformerlab/jobs_trigger_processing.py`)
- `process_job_completion_triggers()` - Main function called when jobs complete
- Automatically called from `job_update_status()` and `job_mark_as_complete_if_running()`

### Processing Logic
1. Job completes and status is set to "COMPLETE"
2. System calls `process_job_completion_triggers(job_id)`
3. Function gets job details (type, experiment_id)
4. Calls `workflow_get_by_job_event()` to find matching workflows
5. Queues each matching workflow using `workflow_queue()`
6. Marks job as having triggers processed

### Matching Criteria
A workflow matches a job completion event if:
1. **Type Match**: Workflow has a trigger config with `trigger_type` matching the job type
2. **Enabled**: The trigger config has `is_enabled: true`
3. **Same Experiment**: Workflow is in the same experiment as the completed job

## Key Features

### 1. Self-Contained Configuration
- Each workflow stores its own trigger settings
- No external trigger management needed
- Settings persist with the workflow

### 2. Automatic Normalization
- All workflow retrieval functions ensure complete trigger configs
- Missing trigger types are added with default values
- Handles legacy workflows without trigger configs

### 3. Validation
- API validates all 6 trigger types are provided
- Ensures no duplicate trigger types
- Validates trigger types against predefined blueprints
- Validates boolean values for `is_enabled`

### 4. Backward Compatibility
- Legacy trigger system remains functional during transition
- Existing workflows get default trigger configs automatically
- Database migration is non-destructive

## Usage Examples

### Frontend Integration
```javascript
// Get available trigger types
const blueprints = await fetch('/workflows/trigger_blueprints').then(r => r.json());

// Get workflow with current trigger settings
const workflow = await fetch(`/workflows/${workflowId}`).then(r => r.json());

// Update trigger settings
await fetch(`/workflows/${workflowId}/trigger_configs`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        configs: [
            { trigger_type: 'TRAIN', is_enabled: true },
            { trigger_type: 'DOWNLOAD_MODEL', is_enabled: false },
            // ... other 4 trigger types
        ]
    })
});
```

### Testing Triggers
1. Create a workflow in an experiment
2. Enable a trigger type (e.g., "TRAIN") for that workflow
3. Run a training job in the same experiment
4. When training completes, the workflow should automatically queue

## Migration Notes

### From Legacy System
- Old `workflow_triggers` table remains for backward compatibility
- New system uses per-workflow `trigger_configs` field
- Legacy polling function `process_completed_job_triggers()` still works
- Gradual migration path allows testing both systems

### Database Changes
- Added `trigger_configs` column to `workflows` table
- Column is nullable to support existing workflows
- Automatic normalization ensures all workflows have complete configs

## Error Handling

### API Level
- 400 errors for invalid request data
- 404 errors for non-existent workflows
- 500 errors for server issues

### Database Level
- Graceful handling of missing trigger configs
- Automatic completion of partial configs
- JSON parsing error recovery

### Processing Level
- Continues processing other workflows if one fails
- Logs errors without crashing the system
- Marks jobs as processed even if trigger processing fails

## Performance Considerations

### Efficient Queries
- Single query to get all workflows for trigger matching
- JSON field indexing for trigger_configs (if needed)
- Minimal database calls during trigger processing

### Scalability
- Event-based processing (no polling overhead)
- Per-workflow configuration (no global trigger management)
- Efficient matching algorithm

## Security Considerations

### Input Validation
- Strict validation of trigger types against predefined list
- Boolean validation for enabled flags
- Workflow ownership validation (implicit through workflow_id)

### Data Integrity
- Atomic updates to trigger configurations
- Consistent data structure enforcement
- Graceful handling of malformed data

## Future Enhancements

### Possible Extensions
1. **Conditional Triggers**: Add conditions based on job success/failure
2. **Job Name Patterns**: Trigger based on specific job names or patterns
3. **Cross-Experiment Triggers**: Allow workflows to trigger across experiments
4. **Trigger Delays**: Add configurable delays before triggering
5. **Trigger Limits**: Limit how many times a workflow can be triggered

### API Improvements
1. **Bulk Updates**: Update multiple workflows' triggers at once
2. **Trigger History**: Track when and why workflows were triggered
3. **Trigger Statistics**: Show trigger activation frequency
4. **Trigger Testing**: Dry-run trigger matching without actually queuing

## Testing Strategy

### Unit Tests
- Test trigger config validation
- Test workflow matching logic
- Test normalization functions

### Integration Tests
- Test complete trigger flow (job completion → workflow queuing)
- Test API endpoints with various inputs
- Test database migration

### End-to-End Tests
- Create workflow with triggers enabled
- Run job and verify workflow triggers
- Test multiple workflows with different trigger settings

## Conclusion

This implementation provides a robust, scalable, and user-friendly workflow trigger system that:
- Gives users fine-grained control over when workflows run
- Maintains data consistency and integrity
- Provides backward compatibility during migration
- Offers a clean API for frontend integration
- Handles errors gracefully
- Scales efficiently with the number of workflows and jobs

The system is ready for production use and provides a solid foundation for future enhancements. 