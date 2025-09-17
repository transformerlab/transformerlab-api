import asyncio
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
import os
import uuid
from typing import Any, Optional

from fastapi import Depends, Request, Response
from fastapi_users import BaseUserManager, FastAPIUsers, UUIDIDMixin, models
from fastapi_users.authentication import (
    AuthenticationBackend,
    BearerTransport,
    JWTStrategy,
)

from fastapi_users.db import SQLAlchemyUserDatabase

import httpx
import jwt
from sqlalchemy import select
from sqlalchemy.orm import joinedload

from transformerlab.db.db import get_user_db
from transformerlab.shared.models.models import User
from transformerlab.db.session import async_session

SECRET = "TEMPSECRET"

_WORKOS_OPENID_CONFIGURATION: dict[str, Any] | None = None
_openid_configuration_lock: asyncio.Lock = asyncio.Lock()


class UserManager(UUIDIDMixin, BaseUserManager[User, uuid.UUID]):
    reset_password_token_secret = SECRET
    verification_token_secret = SECRET

    async def on_after_register(self, user: User, request: Optional[Request] = None):
        print(f"User {user.id} has registered.")
        # Try to backfill profile on first creation as well
        await self._sync_profile_from_oidc(user)

    async def on_after_forgot_password(self, user: User, token: str, request: Optional[Request] = None):
        print(f"User {user.id} has forgot their password. Reset token: {token}")

    async def on_after_request_verify(self, user: User, token: str, request: Optional[Request] = None):
        print(f"Verification requested for user {user.id}. Verification token: {token}")

    async def on_after_login(
        self,
        user: User,
        request: Optional[Request] = None,
        response: Optional[Response] = None,
    ):
        """
        After any successful login (JWT or OAuth), try to sync profile info from OIDC UserInfo.
        This is idempotent and safe to skip if not configured.
        """
        await self._sync_profile_from_oidc(user)

    async def _sync_profile_from_oidc(self, user: User) -> None:
        print("Syncing profile for user", user.id)
        try:
            async with async_session() as session:
                result = await session.execute(
                    select(User).options(joinedload(User.oauth_accounts)).where(User.id == user.id)
                )
                # Because of joined eager loading on a collection, ensure uniqueness
                db_user = result.unique().scalar_one_or_none()
                if not db_user:
                    return

                # Find a linked OAuth account for OpenID/WorkOS
                account = None
                for acc in getattr(db_user, "oauth_accounts", []) or []:
                    if getattr(acc, "oauth_name", None) in ("openid", "workos"):
                        account = acc
                        break
                if not account or not getattr(account, "access_token", None):
                    return

                userinfo_ep = await _get_userinfo_endpoint()
                info: Mapping[str, Any] | None = None
                if userinfo_ep:
                    async with httpx.AsyncClient(timeout=10) as client:
                        resp = await client.get(
                            userinfo_ep,
                            headers={"Authorization": f"Bearer {account.access_token}"},
                        )
                        resp.raise_for_status()
                        info_payload = resp.json()
                    print(info_payload)
                    if isinstance(info_payload, Mapping):
                        info = info_payload

                claims = _decode_claims_no_verify(getattr(account, "access_token", None))
                name = None
                for source in (info or {}, claims):
                    if not isinstance(source, Mapping):
                        continue
                    name = (
                        source.get("name")
                        or source.get("preferred_username")
                        or source.get("nickname")
                    )
                    if name:
                        break
                if not name:
                    given = None
                    family = None
                    if isinstance(info, Mapping):
                        given = info.get("given_name") or info.get("first_name")
                        family = info.get("family_name") or info.get("last_name")
                    if not given or not family:
                        if isinstance(claims, Mapping):
                            given = given or claims.get("given_name")
                            family = family or claims.get("family_name")
                    name = f"{given or ''} {family or ''}".strip()
                updated = False
                if name and name != (db_user.name or ""):
                    db_user.name = name
                    updated = True

                account_updated = False
                account_data: dict[str, object]
                if isinstance(getattr(account, "account_data", None), Mapping):
                    account_data = dict(account.account_data)  # type: ignore[arg-type]
                else:
                    account_data = {}

                # Persist the raw WorkOS user info for downstream consumers.
                metadata_updated, _, _ = _update_workos_metadata(
                    account_data, info=info, claims=claims
                )
                if metadata_updated:
                    account_updated = True

                # Align account identifier with WorkOS profile when available.
                profile_id = None
                for payload in (info or {}, claims):
                    if not isinstance(payload, Mapping):
                        continue
                    for key in ("id", "sub"):
                        value = payload.get(key) if isinstance(payload, Mapping) else None
                        if isinstance(value, str):
                            profile_id = value
                            break
                    if profile_id:
                        break
                if profile_id and getattr(account, "account_id", None) != profile_id:
                    account.account_id = profile_id
                    account_updated = True

                if account_updated:
                    account.account_data = account_data
                    session.add(account)
                    updated = True

                if updated:
                    session.add(db_user)
                    await session.commit()
        except Exception as e:
            print("Profile sync failed for user", user.id, "Error:", e)
            # Never block on profile sync errors
            return


def _decode_claims_no_verify(token: str | None) -> Mapping[str, Any]:
    if not token:
        return {}
    try:
        return jwt.decode(token, options={"verify_signature": False, "verify_aud": False})
    except jwt.InvalidTokenError:
        return {}
    except Exception:
        return {}


def _extract_workos_org(payload: Mapping[str, Any]) -> tuple[str | None, str | None]:
    """Best-effort extraction of WorkOS organization identifiers."""

    if not isinstance(payload, Mapping):
        return (None, None)

    org_id = payload.get("organization_id") or payload.get("org_id")
    org_slug = payload.get("organization_slug")

    org_payload = payload.get("organization")
    if isinstance(org_payload, Mapping):
        org_id = org_id or org_payload.get("id")
        org_slug = org_slug or org_payload.get("slug") or org_payload.get("name")

    profile_payload = payload.get("profile")
    if isinstance(profile_payload, Mapping):
        org_id = org_id or profile_payload.get("organization_id")
        org_slug = org_slug or profile_payload.get("organization_slug")

    for key, value in payload.items():
        if not isinstance(key, str) or not isinstance(value, str):
            continue
        lowered = key.lower()
        if "organization" in lowered or "org" in lowered:
            if "id" in lowered and not org_id:
                org_id = value
            elif "slug" in lowered and not org_slug:
                org_slug = value

    return (
        org_id if isinstance(org_id, str) else None,
        org_slug if isinstance(org_slug, str) else None,
    )


def _update_workos_metadata(
    account_data: dict[str, Any], *, info: Mapping[str, Any] | None, claims: Mapping[str, Any] | None
) -> tuple[bool, str | None, str | None]:
    """Merge WorkOS metadata from userinfo/claims into persisted account data."""

    updated = False

    if info is not None:
        info_dict = dict(info)
        if account_data.get("userinfo") != info_dict:
            account_data["userinfo"] = info_dict
            updated = True

    org_id_claims, org_slug_claims = _extract_workos_org(claims or {})
    org_id_info, org_slug_info = _extract_workos_org(info or {})
    org_id = org_id_claims or org_id_info
    org_slug = org_slug_claims or org_slug_info

    workos_raw = account_data.get("workos") if isinstance(account_data.get("workos"), Mapping) else {}
    workos_data = dict(workos_raw) if isinstance(workos_raw, Mapping) else {}

    if org_id and workos_data.get("organization_id") != org_id:
        workos_data["organization_id"] = org_id
        updated = True
    if org_slug and workos_data.get("organization_slug") != org_slug:
        workos_data["organization_slug"] = org_slug
        updated = True

    if workos_data:
        if account_data.get("workos") != workos_data:
            account_data["workos"] = workos_data
            updated = True
        else:
            account_data["workos"] = workos_data

    if org_id or org_slug:
        try:
            print(f"[workos] resolved organization org_id={org_id} org_slug={org_slug}")
        except Exception:
            pass
    else:
        print("[workos] no organization information found in userinfo or claims")

    return updated, org_id, org_slug


async def _get_openid_configuration() -> Mapping[str, Any] | None:
    global _WORKOS_OPENID_CONFIGURATION
    if _WORKOS_OPENID_CONFIGURATION is not None:
        return _WORKOS_OPENID_CONFIGURATION

    async with _openid_configuration_lock:
        if _WORKOS_OPENID_CONFIGURATION is not None:
            return _WORKOS_OPENID_CONFIGURATION

        workos_openid_config = os.environ.get("WORKOS_OPENID_CONFIGURATION_URL")
        workos_issuer = os.environ.get("WORKOS_ISSUER")
        if not workos_openid_config and workos_issuer:
            workos_openid_config = f"{workos_issuer.rstrip('/')}/.well-known/openid-configuration"
        if not workos_openid_config:
            return None

        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(workos_openid_config)
            resp.raise_for_status()
            disco = resp.json()
            if isinstance(disco, Mapping):
                _WORKOS_OPENID_CONFIGURATION = dict(disco)
            else:
                _WORKOS_OPENID_CONFIGURATION = {}
            return _WORKOS_OPENID_CONFIGURATION


async def _get_userinfo_endpoint() -> str | None:
    config = await _get_openid_configuration()
    if not isinstance(config, Mapping):
        return None
    endpoint = config.get("userinfo_endpoint")
    return endpoint if isinstance(endpoint, str) else None


async def _get_token_endpoint() -> str | None:
    config = await _get_openid_configuration()
    if not isinstance(config, Mapping):
        return None
    endpoint = config.get("token_endpoint")
    return endpoint if isinstance(endpoint, str) else None


def _get_workos_refresh_endpoint() -> str:
    override_keys = (
        "WORKOS_REFRESH_ENDPOINT",
        "WORKOS_AUTHENTICATE_ENDPOINT",
        "WORKOS_AUTHENTICATE_URL",
        "WORKOS_CONNECT_REFRESH_URL",
    )
    for key in override_keys:
        value = os.environ.get(key)
        if value:
            return value
    return "https://api.workos.com/user_management/authenticate"


def _get_workos_api_key() -> str:
    api_key = os.environ.get("WORKOS_API_KEY")
    if not api_key:
        raise RuntimeError(
            "WORKOS_API_KEY is not configured. Set it to scope WorkOS sessions to an organization."
        )
    return api_key


async def _refresh_workos_tokens(refresh_token: str, organization_id: str) -> dict[str, Any]:
    if not refresh_token:
        raise ValueError("WorkOS refresh token is required to scope the session.")
    if not organization_id:
        raise ValueError("organization_id is required when scoping a WorkOS session.")

    api_key = _get_workos_api_key()
    endpoint = _get_workos_refresh_endpoint()

    payload: dict[str, Any] = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "organization_id": organization_id,
    }
    client_id = os.environ.get("WORKOS_CLIENT_ID")
    if client_id:
        payload["client_id"] = client_id

    headers = {"Authorization": f"Bearer {api_key}"}

    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.post(endpoint, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()

    if not isinstance(data, Mapping):
        raise RuntimeError("Unexpected response from WorkOS refresh endpoint")
    return dict(data)


async def scope_workos_session_to_org(user: User, organization_id: str) -> dict[str, Any]:
    if not organization_id:
        raise ValueError("organization_id must be provided")

    async with async_session() as session:
        result = await session.execute(
            select(User).options(joinedload(User.oauth_accounts)).where(User.id == user.id)
        )
        db_user = result.unique().scalar_one_or_none()
        if not db_user:
            raise ValueError("User not found")

        account = None
        for acc in getattr(db_user, "oauth_accounts", []) or []:
            if getattr(acc, "oauth_name", None) in ("openid", "workos"):
                account = acc
                break

        if not account:
            raise ValueError("User does not have a linked WorkOS OAuth account")

        refresh_token = getattr(account, "refresh_token", None)
        if not refresh_token:
            raise ValueError(
                "WorkOS OAuth account is missing a refresh token. Have the user re-authenticate via WorkOS."
            )

        tokens = await _refresh_workos_tokens(refresh_token, organization_id)

        access_token = tokens.get("access_token") if isinstance(tokens, Mapping) else None
        if not isinstance(access_token, str) or not access_token:
            raise RuntimeError("WorkOS refresh response did not include an access_token")

        account.access_token = access_token

        new_refresh_token = tokens.get("refresh_token") if isinstance(tokens, Mapping) else None
        if isinstance(new_refresh_token, str) and new_refresh_token:
            account.refresh_token = new_refresh_token

        expires_in = tokens.get("expires_in") if isinstance(tokens, Mapping) else None
        if isinstance(expires_in, (int, float)):
            account.expires_at = datetime.now(tz=UTC) + timedelta(seconds=int(expires_in))
        else:
            expires_at = tokens.get("expires_at") if isinstance(tokens, Mapping) else None
            if isinstance(expires_at, (int, float)):
                account.expires_at = datetime.fromtimestamp(expires_at, tz=UTC)
            elif isinstance(expires_at, str):
                try:
                    as_int = int(expires_at)
                except ValueError:
                    as_int = None
                if as_int is not None:
                    account.expires_at = datetime.fromtimestamp(as_int, tz=UTC)

        account_data = dict(account.account_data) if isinstance(account.account_data, Mapping) else {}
        claims = _decode_claims_no_verify(access_token)
        metadata_updated, org_id, org_slug = _update_workos_metadata(account_data, info=None, claims=claims)

        if metadata_updated:
            account.account_data = account_data

        session.add(account)
        session.add(db_user)
        await session.commit()

    return {
        "tokens": dict(tokens),
        "claims": dict(claims) if isinstance(claims, Mapping) else {},
        "organization_id": org_id or organization_id,
        "organization_slug": org_slug,
    }


async def get_user_manager(user_db: SQLAlchemyUserDatabase = Depends(get_user_db)):
    yield UserManager(user_db)


### TEMP JWT AUTH TO GET THINGS WORKING


def get_jwt_strategy() -> JWTStrategy[models.UP, models.ID]:
    return JWTStrategy(secret=SECRET, lifetime_seconds=3600)


bearer_transport = BearerTransport(tokenUrl="auth/jwt/login")
auth_backend = AuthenticationBackend(
    name="jwt",
    transport=bearer_transport,
    get_strategy=get_jwt_strategy,
)

fastapi_users = FastAPIUsers[User, uuid.UUID](get_user_manager, [auth_backend])

current_active_user = fastapi_users.current_user(active=True)
