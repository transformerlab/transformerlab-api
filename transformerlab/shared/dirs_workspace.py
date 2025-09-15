import os
from pathlib import Path

# Check for SINGLE_TENANT environment variable
# If SINGLE_TENANT is true or not set, use single tenant mode
# If SINGLE_TENANT is false, use multi-tenant mode with org/workspace structure
SINGLE_TENANT = os.getenv("SINGLE_TENANT", "true").lower() in ("true", "1", "yes")

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

# TFL_WORKSPACE_DIR
if "TFL_WORKSPACE_DIR" in os.environ:
    WORKSPACE_DIR = os.environ["TFL_WORKSPACE_DIR"]
    if not os.path.exists(WORKSPACE_DIR):
        print(f"Error: Workspace directory {WORKSPACE_DIR} does not exist")
        exit(1)
    print(f"Workspace is set to: {WORKSPACE_DIR}")
else:
    if SINGLE_TENANT:
        WORKSPACE_DIR = os.path.join(HOME_DIR, "workspace")
        print(f"Single tenant mode: Using workspace directory: {WORKSPACE_DIR}")
    else:
        # Multi-tenant mode: use org/workspace structure
        # QUESTION: what should we use if TFL_DEFAULT_ORG is not set?
        DEFAULT_ORG = os.getenv("TFL_DEFAULT_ORG", "default")
        WORKSPACE_DIR = os.path.join(HOME_DIR, DEFAULT_ORG, "workspace")
        print(f"Multi-tenant mode: Using workspace directory: {WORKSPACE_DIR}")
    os.makedirs(name=WORKSPACE_DIR, exist_ok=True)
