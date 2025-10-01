"""Workspace path resolution shared by API and plugins."""

import os
from typing import Optional

try:  # pragma: no cover - thin wrapper only
    from lab import WORKSPACE_DIR as LAB_WORKSPACE_DIR
    from lab import HOME_DIR as LAB_HOME_DIR
except Exception as e:
    # We intentionally do not fall back to env per requirement.
    raise ImportError("Failed to import WORKSPACE_DIR from lab.") from e


# Legacy constant for existing imports. Prefer calling _get_workspace_dir instead.
WORKSPACE_DIR = LAB_WORKSPACE_DIR


def _get_workspace_dir(organization_id: Optional[str] = None) -> str:
    """
    Compute the effective workspace directory.

    - If TFL_MULTITENANT is true and organization_id is provided, resolve to
      ~/.transformerlab/orgs/<organization_id>/workspace using lab HOME.
    - Otherwise, return the default lab WORKSPACE_DIR.
    """
    if os.getenv("TFL_MULTITENANT") == "true" and organization_id:
        return os.path.join(LAB_HOME_DIR, "orgs", organization_id, "workspace")
    return LAB_WORKSPACE_DIR