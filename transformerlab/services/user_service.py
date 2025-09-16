import asyncio
import os
import uuid
from typing import Optional

from fastapi import Depends, Request, Response
from fastapi_users import BaseUserManager, FastAPIUsers, UUIDIDMixin, models
from fastapi_users.authentication import (
    AuthenticationBackend,
    BearerTransport,
    JWTStrategy,
)

from fastapi_users.db import SQLAlchemyUserDatabase

import httpx
from sqlalchemy import select
from sqlalchemy.orm import joinedload

from transformerlab.db.db import get_user_db
from transformerlab.shared.models.models import User
from transformerlab.db.session import async_session

SECRET = "TEMPSECRET"

_WORKOS_USERINFO_ENDPOINT: str | None = None
_userinfo_endpoint_lock: asyncio.Lock = asyncio.Lock()


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
                if not userinfo_ep:
                    return

                async with httpx.AsyncClient(timeout=10) as client:
                    resp = await client.get(
                        userinfo_ep, headers={"Authorization": f"Bearer {account.access_token}"}
                    )
                    resp.raise_for_status()
                    info = resp.json()
                print(info)
                name = (
                    info.get("name")
                    or info.get("preferred_username")
                    or info.get("nickname")
                )
                if not name:
                    name = f"{info.get('given_name') or info.get('first_name') or ''} {info.get('family_name') or info.get('last_name') or ''}".strip()
                updated = False
                if name and name != (db_user.name or ""):
                    db_user.name = name
                    updated = True

                if updated:
                    session.add(db_user)
                    await session.commit()
        except Exception as e:
            print("Profile sync failed for user", user.id, "Error:", e)
            # Never block on profile sync errors
            return


async def _get_userinfo_endpoint() -> str | None:
    global _WORKOS_USERINFO_ENDPOINT
    if _WORKOS_USERINFO_ENDPOINT:
        return _WORKOS_USERINFO_ENDPOINT

    async with _userinfo_endpoint_lock:
        if _WORKOS_USERINFO_ENDPOINT:
            return _WORKOS_USERINFO_ENDPOINT

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
            endpoint = disco.get("userinfo_endpoint")
            if not endpoint:
                return None
            _WORKOS_USERINFO_ENDPOINT = endpoint
            return endpoint


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
