import os
from pathlib import Path

# Check for MULTITENANT environment variables
# If MULTITENANT is set, use multi-tenant mode
MULTITENANT = os.getenv("MULTITENANT", "").lower() == "true"

# TFL_HOME_DIR
if "TFL_HOME_DIR" in os.environ:
    HOME_DIR = os.environ["TFL_HOME_DIR"]
    if not os.path.exists(HOME_DIR):
        print(f"Error: Home directory {HOME_DIR} does not exist")
        exit(1)
    print(f"Home directory is set to: {HOME_DIR}")
else:
    HOME_DIR = Path.home() / ".transformerlab"
    os.makedirs(name=HOME_DIR, exist_ok=True)
    print(f"Using default home directory: {HOME_DIR}")

if MULTITENANT:
    WORKSPACE_DIR = os.getenv("REMOTE_WORKSPACE_DIR")
    print(f"🏢 Multi-tenant S3 mode, Using mounted workspace directory: {WORKSPACE_DIR}")
else:
# TFL_WORKSPACE_DIR
    if "TFL_WORKSPACE_DIR" in os.environ:
        WORKSPACE_DIR = os.environ["TFL_WORKSPACE_DIR"]
        if not os.path.exists(WORKSPACE_DIR):
            print(f"Error: Workspace directory {WORKSPACE_DIR} does not exist")
            exit(1)
        print(f"Workspace is set to: {WORKSPACE_DIR}")
    else:
        WORKSPACE_DIR = os.path.join(HOME_DIR, "workspace")
        print(f"Single-tenant local mode: Using workspace directory: {WORKSPACE_DIR}")