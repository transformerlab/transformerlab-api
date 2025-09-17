import importlib
import sys
from types import ModuleType
from typing import Iterable

import pytest


_BASE_MODULES = {
    "transformerlab.services.user_service",
    "transformerlab.db.session",
    "transformerlab.db.constants",
    "transformerlab.shared.dirs_workspace",
}


def import_module_with_temp_workspace(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    module_name: str,
    *,
    extra_modules: Iterable[str] | None = None,
) -> ModuleType:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir(exist_ok=True)
    monkeypatch.setenv("TFL_HOME_DIR", str(tmp_path))
    monkeypatch.setenv("TFL_WORKSPACE_DIR", str(workspace_dir))

    modules_to_clear = set(_BASE_MODULES)
    modules_to_clear.add(module_name)
    if extra_modules:
        modules_to_clear.update(extra_modules)

    for name in modules_to_clear:
        sys.modules.pop(name, None)

    return importlib.import_module(module_name)
