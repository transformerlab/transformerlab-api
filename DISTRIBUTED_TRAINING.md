# Distributed Training System for TransformerLab

This document describes the distributed training system for TransformerLab that enables any existing training plugin to run across multiple machines using HuggingFace Accelerate.

## Architecture Overview

The distributed training system is built on top of the existing plugin architecture and maintains full compatibility with the plugin SDK:

1. **Plugin Harness Integration**: All distributed training jobs are launched through the existing `plugin_harness.py`, ensuring compatibility with the plugin SDK
2. **Accelerate Integration**: Uses HuggingFace Accelerate for distributed coordination and communication
3. **Automatic Configuration**: Automatically adjusts training configurations for distributed settings
4. **Universal Compatibility**: Works with any existing training plugin without modification

## System Components

### 1. Plugin Harness (`plugin_sdk/plugin_harness.py`)
- Extended to support distributed training mode
- Sets up Accelerate environment when `--distributed_mode` flag is provided
- Provides accelerator instance to plugins via `sys.modules['__main__'].accelerator`
- Sets environment variables for distributed training detection

### 2. Distributed Utilities (`plugin_sdk/distributed_utils.py`)
- Helper functions for plugins to detect and work with distributed training
- Provides easy access to distributed training metadata (rank, world size, etc.)
- Utility functions for conditional saving and logging

### 3. Network Router (`routers/network.py`)
- REST API endpoints for planning, dispatching, and monitoring distributed jobs
- Handles machine registration and coordination
- Provides status monitoring across all nodes

### 4. Job Management (`routers/jobs.py`, `routers/tasks.py`)
- Extended task queue system to support distributed job dispatch
- Manages job lifecycle across multiple machines
- Handles job synchronization and status aggregation

### 5. Shared Services (`shared/shared.py`)
- Core logic for launching distributed training jobs
- Automatic configuration adjustment for distributed settings
- Integration with existing plugin execution pipeline

## How It Works

### 1. Job Planning Phase
The coordinator machine analyzes the requested job and determines:
- How many machines are needed
- Resource requirements per machine
- Network configuration (master address/port)
- Plugin compatibility

### 2. Machine Coordination
Available machines register with the coordinator and receive job assignments including:
- Their rank in the distributed job
- Master node connection details
- Plugin and configuration information

### 3. Distributed Launch
Each machine launches the job via the plugin harness with distributed mode enabled:

```bash
accelerate launch --multi_node --num_machines=3 --machine_rank=0 \
  --main_process_ip=192.168.1.100 --main_process_port=29500 \
  plugin_harness.py --plugin_dir=/path/to/plugin \
  --distributed_mode --accelerate_config='{"world_size": 3, "rank": 0}' \
  --config_file=config.json
```

### 4. Plugin Execution
The plugin runs normally, but can optionally use distributed utilities:

```python
from plugin_sdk.distributed_utils import (
    is_distributed_training,
    get_accelerator,
    should_save_outputs,
    print_distributed_info
)

def main():
    if is_distributed_training():
        print_distributed_info()
        accelerator = get_accelerator()
        
        # Use accelerator for distributed training
        model = accelerator.prepare(model)
        optimizer = accelerator.prepare(optimizer)
        dataloader = accelerator.prepare(dataloader)
        
        # Training loop...
        
        # Only save on main process
        if should_save_outputs():
            model.save_pretrained("output_dir")
    else:
        # Regular single-device training
        pass
```

## Configuration Adjustments

The system automatically adjusts training configurations for distributed settings:

- **Batch Size**: Divided by world size to maintain effective batch size
- **Learning Rate**: Scaled by world size (linear scaling rule)
- **Output Directories**: Only rank 0 saves to main directories, others use temporary locations
- **Distributed Metadata**: Adds world size, rank, and coordination info to config

## API Endpoints

### Plan Distributed Job
```http
POST /api/network/distributed_jobs/plan
Content-Type: application/json

{
  "plugin_name": "llama_trainer",
  "config": {...},
  "resource_requirements": {
    "num_machines": 3,
    "gpus_per_machine": 2
  }
}
```

### Dispatch Job to Machine
```http
POST /api/network/distributed_jobs/dispatch
Content-Type: application/json

{
  "job_id": "dist_job_123",
  "target_machine": "192.168.1.101",
  "rank": 1,
  "world_size": 3,
  "master_addr": "192.168.1.100",
  "master_port": 29500,
  "plugin_name": "llama_trainer",
  "config": {...}
}
```

### Monitor Job Status
```http
GET /api/network/distributed_jobs/dist_job_123/status
```

## Frontend Integration

The frontend needs to be updated to support the new distributed training capabilities. This section outlines all the new API routes and UI changes required.

### New API Routes for Distributed Training

#### 1. Network Management Routes (`/api/network/*`)

**Machine Management:**
- `GET /api/network/machines` - List all registered machines
- `POST /api/network/machines` - Add a new machine to the network
- `GET /api/network/machines/{machine_id}` - Get specific machine details
- `DELETE /api/network/machines/{machine_id}` - Remove a machine
- `DELETE /api/network/machines/by-name/{machine_name}` - Remove machine by name

**Network Status:**
- `GET /api/network/status` - Get aggregated status of all machines
- `POST /api/network/machines/{machine_id}/ping` - Ping a specific machine
- `POST /api/network/health-check` - Run health check on all machines
- `GET /api/network/capabilities` - Get capabilities of current machine

**Distributed Training Orchestration:**
- `POST /api/network/distributed/plan` - Plan a distributed training job
- `POST /api/network/distributed/dispatch` - Dispatch job to machines
- `GET /api/network/distributed/status/{job_id}` - Get distributed job status
- `POST /api/network/distributed/stop/{job_id}` - Stop distributed job

#### 2. Task Queue Extensions (`/api/tasks/*`)

**Distributed Task Management:**
- `GET /api/tasks/{task_id}/queue_distributed` - Queue task for distributed training
- `GET /api/tasks/distributed/suggest_machines` - Get machine suggestions for distributed training

#### 3. Job Management Routes (`/api/jobs/*`)

All existing job management routes continue to work and now support distributed jobs.

#### 4. Remote Job Dispatch (Existing Functionality - Preserved)

**Single Machine Remote Execution:**
- `POST /api/network/machines/{machine_id}/dispatch_job` - Dispatch entire job to a remote machine
- `GET /api/tasks/{task_id}/queue/{machine_id}` - Queue task to run on specific remote machine
- `GET /api/network/machines/{machine_id}/job_status/{job_id}` - Get status of job on remote machine
- `POST /api/network/machines/{machine_id}/stop_job/{job_id}` - Stop job on remote machine

**Remote Job Support Infrastructure:**
- `POST /api/network/prepare_dependencies` - Prepare dependencies on remote machine
- `POST /api/network/execute_job` - Execute job on this machine (called by remote)
- `GET /api/network/local_job_status/{job_id}` - Get local job status (for remote polling)
- `GET /api/network/local_job_file/{job_id}/{key}` - Get job files from remote machine
- `GET /api/network/local_job_output/{job_id}` - Get job output from remote machine

### Comparison: Remote Jobs vs Distributed Training

The system now supports two types of multi-machine execution:

#### 1. Remote Job Dispatch (Existing - Preserved)
- **Purpose**: Run an entire job on a single remote machine
- **Use Case**: Utilize a powerful remote machine for training
- **API**: `POST /api/network/machines/{machine_id}/dispatch_job`
- **Frontend**: Single machine selection dropdown
- **Job Execution**: Complete job runs on one machine

#### 2. Distributed Training (New)
- **Purpose**: Split training across multiple machines simultaneously
- **Use Case**: Scale training beyond single machine capabilities
- **API**: `POST /api/network/distributed/plan` + `POST /api/network/distributed/dispatch`
- **Frontend**: Multi-machine selection with coordination
- **Job Execution**: Coordinated training across multiple machines

Both approaches are fully supported and serve different use cases:
- Use **Remote Job Dispatch** when you want to run on a more powerful single machine
- Use **Distributed Training** when you need to scale beyond what a single machine can handle

## Frontend UI Requirements

#### 1. Machine Management Interface

Create a new "Network" or "Machines" section in the UI with:

```typescript
interface Machine {
  id: string;
  name: string;
  host: string;
  port: number;
  status: 'online' | 'offline' | 'busy';
  capabilities: {
    gpu_count: number;
    gpu_memory: number;
    cpu_count: number;
    memory: number;
    disk_space: number;
  };
  current_load: {
    cpu_usage: number;
    memory_usage: number;
    gpu_usage: number[];
  };
}
```

**Required UI Components:**
- Machine list/grid view
- Add machine form (host, port, name)
- Machine status indicators
- Remove machine functionality
- Health check buttons
- Real-time status updates

#### 2. Distributed Training Interface

Extend the existing training interface with distributed options:

```typescript
interface DistributedTrainingConfig {
  plugin_name: string;
  config: any; // existing training config
  resource_requirements: {
    num_machines: number;
    gpus_per_machine?: number;
    min_gpu_memory?: number;
  };
  machine_selection: 'auto' | 'manual';
  selected_machines?: string[]; // machine IDs if manual selection
}
```

**New UI Elements:**
- "Distributed Training" toggle/checkbox
- Machine count selector
- GPU per machine selector
- Machine selection (auto vs manual)
- Manual machine picker (if manual selected)
- Resource requirement sliders
- Distributed job status display

#### 3. Job Monitoring Enhancements

Extend job monitoring to show distributed job details:

```typescript
interface DistributedJobStatus {
  job_id: string;
  status: 'planning' | 'dispatching' | 'running' | 'completed' | 'failed';
  world_size: number;
  nodes: {
    machine_id: string;
    rank: number;
    status: 'pending' | 'running' | 'completed' | 'failed';
    start_time?: string;
    end_time?: string;
    progress?: number;
  }[];
  master_addr: string;
  master_port: number;
}
```

**Enhanced Job Display:**
- Multi-node progress bars
- Per-machine status indicators
- Distributed job topology view
- Aggregated logs from all nodes
- Stop all nodes functionality

### API Usage Examples for Frontend

#### 1. Setting up Distributed Training

```typescript
// 1. Get machine suggestions
const suggestions = await fetch('/api/tasks/distributed/suggest_machines').then(r => r.json());

// 2. Plan distributed job
const planRequest = {
  plugin_name: 'llama_trainer',
  config: trainingConfig,
  resource_requirements: {
    num_machines: 3,
    gpus_per_machine: 2
  }
};

const planResponse = await fetch('/api/network/distributed/plan', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(planRequest)
});

const { job_id, execution_plan } = await planResponse.json();

// 3. Queue distributed task
await fetch(`/api/tasks/${task_id}/queue_distributed`);
```

#### 2. Monitoring Distributed Job

```typescript
// Poll for distributed job status
const pollDistributedJob = async (jobId: string) => {
  const response = await fetch(`/api/network/distributed/status/${jobId}`);
  const status = await response.json();
  
  // Update UI with multi-node status
  updateDistributedJobUI(status);
  
  if (status.status === 'running') {
    setTimeout(() => pollDistributedJob(jobId), 5000);
  }
};
```

#### 3. Machine Management

```typescript
// Add a new machine
const addMachine = async (machine: { name: string, host: string, port: number }) => {
  const response = await fetch('/api/network/machines', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(machine)
  });
  return response.json();
};

// Get all machines
const getMachines = async () => {
  const response = await fetch('/api/network/machines');
  return response.json();
};

// Health check all machines
const healthCheckAll = async () => {
  const response = await fetch('/api/network/health-check', { method: 'POST' });
  return response.json();
};
```

### UI Flow for Distributed Training

1. **Setup Phase:**
   - User navigates to training interface
   - User toggles "Distributed Training" option
   - UI shows machine selection and resource options
   - User configures distributed settings

2. **Planning Phase:**
   - Frontend calls `/api/network/distributed/plan`
   - Shows execution plan to user for confirmation
   - User can modify machine selection if needed

3. **Execution Phase:**
   - Frontend calls `/api/tasks/{task_id}/queue_distributed`
   - UI switches to distributed monitoring view
   - Shows multi-node progress and status

4. **Monitoring Phase:**
   - Real-time updates from `/api/network/distributed/status/{job_id}`
   - Per-machine status indicators
   - Aggregated logs and metrics
   - Option to stop all nodes

### Error Handling

Frontend should handle these distributed-specific errors:

- **Machine Unavailable**: Show alternative machine suggestions
- **Resource Insufficient**: Guide user to adjust requirements
- **Network Issues**: Provide retry options and diagnostics
- **Partial Failures**: Show which nodes succeeded/failed

### Configuration Storage

Frontend should persist distributed training preferences:

```typescript
interface DistributedPreferences {
  preferred_machines: string[];
  default_num_machines: number;
  default_gpus_per_machine: number;
  auto_machine_selection: boolean;
}
```

This comprehensive integration will provide users with a seamless distributed training experience while maintaining the familiar TransformerLab interface.
