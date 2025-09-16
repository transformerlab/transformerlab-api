import asyncio

import pytest

from transformerlab.services.openid_client import OpenIDWithProfile
from transformerlab.services.redirect_transport import RedirectBearerTransport

from test.utils import import_module_with_temp_workspace


@pytest.mark.asyncio
async def test_redirect_transport_redirects_with_token():
    transport = RedirectBearerTransport(
        tokenUrl="auth/jwt/login", frontend_redirect_url="http://frontend.example/auth/callback"
    )

    response = await transport.get_login_response("abc123")

    assert response.status_code == 302
    assert response.headers["location"] == (
        "http://frontend.example/auth/callback#access_token=abc123&token_type=bearer"
    )


@pytest.mark.asyncio
async def test_redirect_transport_trims_trailing_slash():
    transport = RedirectBearerTransport(
        tokenUrl="auth/jwt/login", frontend_redirect_url="http://frontend.example/app/"
    )

    response = await transport.get_login_response("token")

    assert response.headers["location"] == (
        "http://frontend.example/app#access_token=token&token_type=bearer"
    )


def test_openid_with_profile_default_scopes():
    client = OpenIDWithProfile(
        "client_id",
        "client_secret",
        "https://workos.example/.well-known/openid-configuration",
    )

    assert client.base_scopes == ["openid", "email", "profile"]


@pytest.mark.asyncio
async def test_redirect_transport_encodes_token_characters():
    transport = RedirectBearerTransport(
        tokenUrl="auth/jwt/login", frontend_redirect_url="http://frontend.example/auth/callback"
    )

    response = await transport.get_login_response("a+/= ß")

    assert response.headers["location"].endswith("#access_token=a%2B%2F%3D%20%C3%9F&token_type=bearer")


@pytest.mark.asyncio
async def test_get_userinfo_endpoint_fetches_once(monkeypatch, tmp_path):
    user_service = import_module_with_temp_workspace(
        monkeypatch, tmp_path, "transformerlab.services.user_service"
    )

    monkeypatch.setenv(
        "WORKOS_OPENID_CONFIGURATION_URL",
        "https://workos.example/.well-known/openid-configuration",
    )

    calls: list[str] = []

    class DummyResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, str]:
            return {"userinfo_endpoint": "https://workos.example/userinfo"}

    class DummyClient:
        def __init__(self, *args, **kwargs):
            return

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def get(self, url: str):
            calls.append(url)
            return DummyResponse()

    monkeypatch.setattr(user_service.httpx, "AsyncClient", DummyClient)
    user_service._WORKOS_USERINFO_ENDPOINT = None

    endpoint = await user_service._get_userinfo_endpoint()

    assert endpoint == "https://workos.example/userinfo"
    assert calls == ["https://workos.example/.well-known/openid-configuration"]

    class ShouldNotRun:
        def __init__(self, *args, **kwargs):
            raise AssertionError("AsyncClient should not be constructed for cached endpoint")

    monkeypatch.setattr(user_service.httpx, "AsyncClient", ShouldNotRun)

    cached = await user_service._get_userinfo_endpoint()

    assert cached == endpoint


@pytest.mark.asyncio
async def test_get_userinfo_endpoint_from_issuer(monkeypatch, tmp_path):
    user_service = import_module_with_temp_workspace(
        monkeypatch, tmp_path, "transformerlab.services.user_service"
    )

    monkeypatch.delenv("WORKOS_OPENID_CONFIGURATION_URL", raising=False)
    monkeypatch.setenv("WORKOS_ISSUER", "https://workos.example")

    calls: list[str] = []

    class DummyResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, str]:
            return {"userinfo_endpoint": "https://workos.example/userinfo"}

    class DummyClient:
        def __init__(self, *args, **kwargs):
            return

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def get(self, url: str):
            calls.append(url)
            return DummyResponse()

    monkeypatch.setattr(user_service.httpx, "AsyncClient", DummyClient)
    user_service._WORKOS_USERINFO_ENDPOINT = None

    endpoint = await user_service._get_userinfo_endpoint()

    assert endpoint == "https://workos.example/userinfo"
    assert calls == ["https://workos.example/.well-known/openid-configuration"]


@pytest.mark.asyncio
async def test_get_userinfo_endpoint_concurrent_requests(monkeypatch, tmp_path):
    user_service = import_module_with_temp_workspace(
        monkeypatch, tmp_path, "transformerlab.services.user_service"
    )

    monkeypatch.setenv(
        "WORKOS_OPENID_CONFIGURATION_URL",
        "https://workos.example/.well-known/openid-configuration",
    )

    calls: list[str] = []

    class DummyResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, str]:
            return {"userinfo_endpoint": "https://workos.example/userinfo"}

    class DummyClient:
        def __init__(self, *args, **kwargs):
            return

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def get(self, url: str):
            calls.append(url)
            await asyncio.sleep(0.01)
            return DummyResponse()

    monkeypatch.setattr(user_service.httpx, "AsyncClient", DummyClient)
    user_service._WORKOS_USERINFO_ENDPOINT = None
    user_service._userinfo_endpoint_lock = asyncio.Lock()

    first, second = await asyncio.gather(
        user_service._get_userinfo_endpoint(), user_service._get_userinfo_endpoint()
    )

    assert first == "https://workos.example/userinfo"
    assert second == first
    assert calls == ["https://workos.example/.well-known/openid-configuration"]
