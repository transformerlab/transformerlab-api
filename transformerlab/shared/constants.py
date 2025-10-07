"""Workspace path resolution shared by API and plugins."""

try:  # pragma: no cover - thin wrapper only
    from lab.dirs import get_workspace_dir
except Exception as e:
    # We intentionally do not fall back to env per requirement.
    raise ImportError("Failed to import WORKSPACE_DIR from lab.") from e
