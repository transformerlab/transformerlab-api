import logging
from fastapi import FastAPI
from fastapi.testclient import TestClient

from test.utils import import_module_with_temp_workspace


class DummyLogger:
    def __init__(self):
        self.messages: list[tuple[str, str]] = []

    def info(self, message: str, *args, **kwargs) -> None:
        self.messages.append(("info", message))

    def warning(self, message: str, *args, **kwargs) -> None:
        self.messages.append(("warning", message))

    def debug(self, message: str, *args, **kwargs) -> None:
        self.messages.append(("debug", message))


def _collect_paths(router_module) -> set[str]:
    app = FastAPI()
    app.include_router(router_module.router)
    return {route.path for route in app.routes}


def test_workos_routes_missing_configuration(monkeypatch, tmp_path):
    dummy_logger = DummyLogger()
    original_get_logger = logging.getLogger

    def fake_get_logger(name=None):
        # Only capture logs from transformerlab modules; allow pytest and others to function.
        if name and name.startswith("transformerlab"):
            return dummy_logger
        return original_get_logger(name)

    monkeypatch.setattr(logging, "getLogger", fake_get_logger)

    # Ensure any pre-existing WorkOS-related env vars are cleared so the
    # router logic detects missing configuration deterministically.
    for var in [
        "WORKOS_CLIENT_ID",
        "WORKOS_CLIENT_SECRET",
        "WORKOS_OPENID_CONFIGURATION_URL",
        "WORKOS_ISSUER",
        "WORKOS_REDIRECT_URL",
    ]:
        monkeypatch.delenv(var, raising=False)

    router_module = import_module_with_temp_workspace(
        monkeypatch,
        tmp_path,
        "transformerlab.routers.users",
        extra_modules=["transformerlab.services.user_service"],
    )

    paths = _collect_paths(router_module)

    assert not any(path.endswith("/auth/workos/authorize") for path in paths)
    assert not any(path.endswith("/auth/workos/scope") for path in paths)
    warning_messages = [msg for level, msg in dummy_logger.messages if level == "warning"]
    assert any("Skipping WorkOS OAuth routes" in msg for msg in warning_messages)


def test_workos_routes_enabled_with_configuration(monkeypatch, tmp_path):
    dummy_logger = DummyLogger()
    original_get_logger = logging.getLogger

    def fake_get_logger(name=None):
        if name and name.startswith("transformerlab"):
            return dummy_logger
        return original_get_logger(name)

    monkeypatch.setattr(logging, "getLogger", fake_get_logger)

    monkeypatch.setenv("WORKOS_CLIENT_ID", "client")
    monkeypatch.setenv("WORKOS_CLIENT_SECRET", "secret")
    monkeypatch.setenv(
        "WORKOS_OPENID_CONFIGURATION_URL",
        "https://workos.example/.well-known/openid-configuration",
    )
    monkeypatch.setenv("WORKOS_REDIRECT_URL", "https://backend.example/auth/workos/callback")

    router_module = import_module_with_temp_workspace(
        monkeypatch,
        tmp_path,
        "transformerlab.routers.users",
        extra_modules=["transformerlab.services.user_service"],
    )

    paths = _collect_paths(router_module)

    assert any(path.endswith("/auth/workos/authorize") for path in paths)
    assert any(path.endswith("/auth/workos/scope") for path in paths)
    info_messages = [msg for level, msg in dummy_logger.messages if level == "info"]
    assert any("Mounted WorkOS OAuth routes" in msg for msg in info_messages)


def test_workos_scope_endpoint_returns_tokens(monkeypatch, tmp_path):
    monkeypatch.setenv("WORKOS_CLIENT_ID", "client")
    monkeypatch.setenv("WORKOS_CLIENT_SECRET", "secret")
    monkeypatch.setenv(
        "WORKOS_OPENID_CONFIGURATION_URL",
        "https://workos.example/.well-known/openid-configuration",
    )
    monkeypatch.setenv("WORKOS_REDIRECT_URL", "https://backend.example/auth/workos/callback")

    router_module = import_module_with_temp_workspace(
        monkeypatch,
        tmp_path,
        "transformerlab.routers.users",
        extra_modules=["transformerlab.services.user_service"],
    )

    async def fake_scope(user, organization_id):
        assert organization_id == "org_abc"
        return {
            "tokens": {
                "access_token": "new-access",
                "refresh_token": "new-refresh",
                "token_type": "bearer",
                "expires_in": 1800,
                "expires_at": 1700000000,
                "id_token": "id-token",
            },
            "organization_id": "org_abc",
            "organization_slug": "sluggy",
            "claims": {"org_id": "org_abc"},
        }

    monkeypatch.setattr(router_module, "scope_workos_session_to_org", fake_scope)

    class DummyUser:
        id = "user-id"

    app = FastAPI()
    app.include_router(router_module.router)
    app.dependency_overrides[router_module.current_active_user] = lambda: DummyUser()

    client = TestClient(app)
    resp = client.post("/auth/workos/scope", json={"organization_id": "  org_abc  "})
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["access_token"] == "new-access"
    assert payload["refresh_token"] == "new-refresh"
    assert payload["organization_id"] == "org_abc"
    assert payload["organization_slug"] == "sluggy"
    assert payload["claims"]["org_id"] == "org_abc"
