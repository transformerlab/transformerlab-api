import importlib
import os
import sys
from typing import List

from fastapi.testclient import TestClient
from fastapi.routing import APIRoute

from transformerlab.services.user_service import current_active_user


def _reload_app(auth_enabled: bool):
    if auth_enabled:
        os.environ["_TFL_ENABLE_AUTH"] = "true"
    else:
        os.environ.pop("_TFL_ENABLE_AUTH", None)

    if "api" in sys.modules:
        del sys.modules["api"]

    import api
    importlib.reload(api)
    return api.app


def _protected_routes(app) -> List[APIRoute]:
    return [
        r
        for r in app.routes
        if hasattr(r, "dependant")
        and any(dep.call == current_active_user for dep in r.dependant.dependencies)
    ]


def _sample_public_path(app):
    for route in app.routes:
        if (
            "GET" in route.methods
            and "{" not in route.path
            and not route.path.startswith(("/openapi.json", "/docs", "/redoc"))
        ):
            return route.path
    raise RuntimeError("No suitable sample GET route found for smoke test.")


def test_routes_public_when_auth_disabled():
    app = _reload_app(auth_enabled=False)
    assert _protected_routes(app) == [], "Unexpected auth dependency detected"
    client = TestClient(app)
    path = _sample_public_path(app)
    r = client.get(path)
    assert (
        r.status_code < 400
    ), f"Expected route {path} to be public, got {r.status_code}"


def test_routes_protected_when_auth_enabled(monkeypatch):
    app = _reload_app(auth_enabled=True)
    prot = _protected_routes(app)
    assert prot, "No route was protected with auth enabled"
    client = TestClient(app)
    path = prot[0].path
    if "{" in path:
        path = path.format(**{p.strip("{}"): 1 for p in path.split("/") if p.startswith("{")})
    r = client.get(path)
    assert r.status_code in (
        401,
        403,
    ), f"Expected auth failure (401/403), got {r.status_code}"
    app.dependency_overrides[current_active_user] = lambda: {"sub": "fake-user"}
    r2 = client.get(path)
    assert (
        r2.status_code < 400
    ), f"Authenticated request should succeed, got {r2.status_code}"
