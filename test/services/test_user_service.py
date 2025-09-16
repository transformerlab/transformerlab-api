import importlib

import pytest
from fastapi_users.db import SQLAlchemyUserDatabase
from sqlalchemy import select

from test.utils import import_module_with_temp_workspace


@pytest.mark.asyncio
async def test_sync_profile_updates_user_name(monkeypatch, tmp_path):
    user_service = import_module_with_temp_workspace(
        monkeypatch, tmp_path, "transformerlab.services.user_service"
    )
    session_module = importlib.import_module("transformerlab.db.session")
    models = importlib.import_module("transformerlab.shared.models.models")

    await session_module.init()

    try:
        async with session_module.async_session() as session:
            user = models.User(
                email="user@example.com",
                hashed_password="hashed",
                is_active=True,
                is_superuser=False,
                is_verified=False,
                name="",
            )
            account = models.OAuthAccount(
                oauth_name="workos",
                access_token="token-123",
                account_id="account-id",
                account_email="user@example.com",
                refresh_token=None,
                expires_at=None,
            )
            user.oauth_accounts.append(account)
            session.add(user)
            await session.commit()
            user_id = user.id

        monkeypatch.setenv(
            "WORKOS_OPENID_CONFIGURATION_URL",
            "https://workos.example/.well-known/openid-configuration",
        )

        calls: list[str] = []

        class DummyResponseConfig:
            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict[str, str]:
                return {"userinfo_endpoint": "https://workos.example/userinfo"}

        class DummyResponseUser:
            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict[str, str]:
                return {"given_name": "Ada", "family_name": "Lovelace"}

        class DummyClient:
            def __init__(self, *args, **kwargs):
                self.last_headers: dict[str, str] | None = None

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return None

            async def get(self, url: str, *, headers=None):
                calls.append(url)
                if headers is not None:
                    self.last_headers = headers
                if url.endswith("openid-configuration"):
                    return DummyResponseConfig()
                assert url.endswith("userinfo")
                return DummyResponseUser()

        dummy_client = DummyClient()

        def client_factory(*args, **kwargs):
            return dummy_client

        monkeypatch.setattr(user_service.httpx, "AsyncClient", client_factory)
        user_service._WORKOS_USERINFO_ENDPOINT = None

        async with session_module.async_session() as session:
            user_db = SQLAlchemyUserDatabase(session, models.User, models.OAuthAccount)
            db_user = await user_db.get(user_id)
            manager = user_service.UserManager(user_db)
            await manager._sync_profile_from_oidc(db_user)

        async with session_module.async_session() as session:
            result = await session.execute(select(models.User).where(models.User.id == user_id))
            refreshed = result.scalar_one()
            assert refreshed.name == "Ada Lovelace"

        assert calls == [
            "https://workos.example/.well-known/openid-configuration",
            "https://workos.example/userinfo",
        ]
        assert dummy_client.last_headers == {"Authorization": "Bearer token-123"}
    finally:
        await session_module.close()


@pytest.mark.asyncio
async def test_sync_profile_skips_without_oauth_account(monkeypatch, tmp_path):
    user_service = import_module_with_temp_workspace(
        monkeypatch, tmp_path, "transformerlab.services.user_service"
    )
    session_module = importlib.import_module("transformerlab.db.session")
    models = importlib.import_module("transformerlab.shared.models.models")

    await session_module.init()

    try:
        async with session_module.async_session() as session:
            user = models.User(
                email="plain@example.com",
                hashed_password="hashed",
                is_active=True,
                is_superuser=False,
                is_verified=False,
                name="Original",
            )
            session.add(user)
            await session.commit()
            user_id = user.id

        class ShouldNotRun:
            def __init__(self, *args, **kwargs):
                raise AssertionError("OAuth discovery should not execute")

        monkeypatch.setattr(user_service.httpx, "AsyncClient", ShouldNotRun)
        user_service._WORKOS_USERINFO_ENDPOINT = None

        async with session_module.async_session() as session:
            user_db = SQLAlchemyUserDatabase(session, models.User, models.OAuthAccount)
            db_user = await user_db.get(user_id)
            manager = user_service.UserManager(user_db)
            await manager._sync_profile_from_oidc(db_user)

        async with session_module.async_session() as session:
            result = await session.execute(select(models.User).where(models.User.id == user_id))
            refreshed = result.scalar_one()
            assert refreshed.name == "Original"
    finally:
        await session_module.close()


@pytest.mark.asyncio
async def test_sync_profile_selects_workos_account(monkeypatch, tmp_path):
    user_service = import_module_with_temp_workspace(
        monkeypatch, tmp_path, "transformerlab.services.user_service"
    )
    session_module = importlib.import_module("transformerlab.db.session")
    models = importlib.import_module("transformerlab.shared.models.models")

    await session_module.init()

    try:
        async with session_module.async_session() as session:
            user = models.User(
                email="user@example.com",
                hashed_password="hashed",
                is_active=True,
                is_superuser=False,
                is_verified=False,
                name="",
            )
            other_account = models.OAuthAccount(
                oauth_name="google",
                access_token="other-token",
                account_id="google-id",
                account_email="user@example.com",
            )
            workos_account = models.OAuthAccount(
                oauth_name="workos",
                access_token="workos-token",
                account_id="workos-id",
                account_email="user@example.com",
            )
            user.oauth_accounts.extend([other_account, workos_account])
            session.add(user)
            await session.commit()
            user_id = user.id

        monkeypatch.setenv(
            "WORKOS_OPENID_CONFIGURATION_URL",
            "https://workos.example/.well-known/openid-configuration",
        )

        requests: list[tuple[str, dict[str, str] | None]] = []

        class DummyResponse:
            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict[str, str]:
                if len(requests) == 1:
                    return {"userinfo_endpoint": "https://workos.example/userinfo"}
                return {"name": "WorkOS User"}

        class DummyClient:
            def __init__(self, *args, **kwargs):
                return

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return None

            async def get(self, url: str, *, headers=None):
                requests.append((url, headers))
                return DummyResponse()

        monkeypatch.setattr(user_service.httpx, "AsyncClient", DummyClient)
        user_service._WORKOS_USERINFO_ENDPOINT = None

        async with session_module.async_session() as session:
            user_db = SQLAlchemyUserDatabase(session, models.User, models.OAuthAccount)
            db_user = await user_db.get(user_id)
            manager = user_service.UserManager(user_db)
            await manager._sync_profile_from_oidc(db_user)

        async with session_module.async_session() as session:
            result = await session.execute(select(models.User).where(models.User.id == user_id))
            refreshed = result.scalar_one()
            assert refreshed.name == "WorkOS User"

        assert requests[1][1] == {"Authorization": "Bearer workos-token"}
    finally:
        await session_module.close()


@pytest.mark.asyncio
async def test_sync_profile_requires_access_token(monkeypatch, tmp_path):
    user_service = import_module_with_temp_workspace(
        monkeypatch, tmp_path, "transformerlab.services.user_service"
    )
    session_module = importlib.import_module("transformerlab.db.session")
    models = importlib.import_module("transformerlab.shared.models.models")

    await session_module.init()

    try:
        async with session_module.async_session() as session:
            user = models.User(
                email="user@example.com",
                hashed_password="hashed",
                is_active=True,
                is_superuser=False,
                is_verified=False,
                name="Original",
            )
            account = models.OAuthAccount(
                oauth_name="workos",
                access_token=None,
                account_id="workos-id",
                account_email="user@example.com",
            )
            user.oauth_accounts.append(account)
            session.add(user)
            await session.commit()
            user_id = user.id

        class ShouldNotRun:
            def __init__(self, *args, **kwargs):
                raise AssertionError("Should not fetch discovery without access token")

        monkeypatch.setattr(user_service.httpx, "AsyncClient", ShouldNotRun)
        user_service._WORKOS_USERINFO_ENDPOINT = None

        async with session_module.async_session() as session:
            user_db = SQLAlchemyUserDatabase(session, models.User, models.OAuthAccount)
            db_user = await user_db.get(user_id)
            manager = user_service.UserManager(user_db)
            await manager._sync_profile_from_oidc(db_user)

        async with session_module.async_session() as session:
            result = await session.execute(select(models.User).where(models.User.id == user_id))
            refreshed = result.scalar_one()
            assert refreshed.name == "Original"
    finally:
        await session_module.close()
