# Single source of truth for workspace directory used across API and plugins
# Currently delegated to lab's WORKSPACE_DIR. Change here to adjust globally.
try:
    from lab import WORKSPACE_DIR
except Exception as e:
    # We intentionally do not fall back to env per requirement.
    raise ImportError("Failed to import WORKSPACE_DIR from lab.") from e


WORKSPACE_DIR = WORKSPACE_DIR