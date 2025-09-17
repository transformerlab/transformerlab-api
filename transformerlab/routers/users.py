import os
import logging
from collections.abc import Mapping
from typing import Any

import httpx
from fastapi import APIRouter, Depends, HTTPException, status

# from .user_service import auth_backend, current_active_user, fastapi_users

# FastAPI Users integration
from transformerlab.services.user_service import (
    auth_backend,
    current_active_user,
    fastapi_users,
    get_jwt_strategy,
    scope_workos_session_to_org,
    SECRET,
)
from transformerlab.services.redirect_transport import RedirectBearerTransport
from fastapi_users.authentication import AuthenticationBackend
from pydantic import BaseModel, Field
from transformerlab.schemas.user import UserCreate, UserRead, UserUpdate

logger = logging.getLogger(__name__)

router = APIRouter()


class WorkOSScopeSessionRequest(BaseModel):
    organization_id: str = Field(..., min_length=1)


class WorkOSScopeSessionResponse(BaseModel):
    access_token: str
    refresh_token: str | None = None
    token_type: str | None = None
    expires_in: int | None = None
    expires_at: int | None = None
    id_token: str | None = None
    organization_id: str | None = None
    organization_slug: str | None = None
    claims: dict[str, Any] | None = None

router.include_router(
    fastapi_users.get_auth_router(auth_backend),
    prefix="/auth/jwt",
    tags=["auth"],
)
router.include_router(
    fastapi_users.get_register_router(UserRead, UserCreate),
    prefix="/auth",
    tags=["auth"],
)
router.include_router(
    fastapi_users.get_reset_password_router(),
    prefix="/auth",
    tags=["auth"],
)
router.include_router(
    fastapi_users.get_verify_router(UserRead),
    prefix="/auth",
    tags=["auth"],
)
router.include_router(
    fastapi_users.get_users_router(UserRead, UserUpdate),
    prefix="/users",
    tags=["users"],
)

# Optional: WorkOS Connect (OpenID Connect) OAuth routes
# Configure via env vars: WORKOS_CLIENT_ID, WORKOS_CLIENT_SECRET,
# WORKOS_OPENID_CONFIGURATION_URL or WORKOS_ISSUER, WORKOS_REDIRECT_URL, OAUTH_STATE_SECRET

WORKOS_CLIENT_ID = os.environ.get("WORKOS_CLIENT_ID")
WORKOS_CLIENT_SECRET = os.environ.get("WORKOS_CLIENT_SECRET")
WORKOS_OPENID_CONFIGURATION_URL = os.environ.get("WORKOS_OPENID_CONFIGURATION_URL")
WORKOS_ISSUER = os.environ.get("WORKOS_ISSUER")
WORKOS_REDIRECT_URL = os.environ.get("WORKOS_REDIRECT_URL")  # e.g. https://your-b.example.com/auth/workos/callback
# After successful OAuth login, redirect the browser here (frontend)
FRONTEND_REDIRECT_URL = os.environ.get("FRONTEND_REDIRECT_URL", "http://localhost:1212/auth/callback")
STATE_SECRET = os.environ.get("OAUTH_STATE_SECRET", SECRET)

if not WORKOS_OPENID_CONFIGURATION_URL and WORKOS_ISSUER:
    WORKOS_OPENID_CONFIGURATION_URL = f"{WORKOS_ISSUER.rstrip('/')}/.well-known/openid-configuration"

have_all = all([
    WORKOS_CLIENT_ID,
    WORKOS_CLIENT_SECRET,
    WORKOS_REDIRECT_URL,
    WORKOS_OPENID_CONFIGURATION_URL,
])
if have_all:
    from transformerlab.services.openid_client import LazyOpenIDWithProfile
    # Use lazy client so import does not perform network discovery
    openid_client = LazyOpenIDWithProfile(
        client_id=WORKOS_CLIENT_ID,
        client_secret=WORKOS_CLIENT_SECRET,
        openid_configuration_endpoint=WORKOS_OPENID_CONFIGURATION_URL,
    )
    # Use a redirecting transport for OAuth so the browser does not display raw JSON
    oauth_redirect_transport = RedirectBearerTransport(
        tokenUrl="auth/jwt/login",
        frontend_redirect_url=FRONTEND_REDIRECT_URL,
    )
    oauth_backend = AuthenticationBackend(
        name="jwt_oauth",
        transport=oauth_redirect_transport,
        get_strategy=get_jwt_strategy,
    )

    oauth_router = fastapi_users.get_oauth_router(
        openid_client,
        oauth_backend,
        STATE_SECRET,
        WORKOS_REDIRECT_URL,
        associate_by_email=True,
        is_verified_by_default=True,
    )
    router.include_router(oauth_router, prefix="/auth/workos", tags=["auth"])
    logger.info("Mounted WorkOS OAuth routes at /auth/workos (authorize, callback)")

    @router.post(
        "/auth/workos/scope",
        response_model=WorkOSScopeSessionResponse,
        tags=["auth"],
    )
    async def scope_workos_session(
        payload: WorkOSScopeSessionRequest,
        user=Depends(current_active_user),
    ) -> WorkOSScopeSessionResponse:
        organization_id = payload.organization_id.strip()
        try:
            result = await scope_workos_session_to_org(user, organization_id)
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        except httpx.HTTPStatusError as exc:
            detail: Any
            try:
                detail = exc.response.json()
            except Exception:
                detail = exc.response.text
            raise HTTPException(status_code=exc.response.status_code, detail=detail) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc

        tokens = result.get("tokens", {})
        access_token = tokens.get("access_token")
        if not isinstance(access_token, str) or not access_token:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="WorkOS refresh succeeded without an access token.",
            )

        claims = result.get("claims")
        claims_dict = claims if isinstance(claims, dict) else dict(claims) if isinstance(claims, Mapping) else None

        return WorkOSScopeSessionResponse(
            access_token=access_token,
            refresh_token=tokens.get("refresh_token"),
            token_type=tokens.get("token_type"),
            expires_in=tokens.get("expires_in"),
            expires_at=tokens.get("expires_at"),
            id_token=tokens.get("id_token"),
            organization_id=result.get("organization_id"),
            organization_slug=result.get("organization_slug"),
            claims=claims_dict,
        )
else:
    logger.warning(
        "Skipping WorkOS OAuth routes. Missing vars: "
        f"WORKOS_CLIENT_ID={bool(WORKOS_CLIENT_ID)} "
        f"WORKOS_CLIENT_SECRET={bool(WORKOS_CLIENT_SECRET)} "
        f"WORKOS_REDIRECT_URL={bool(WORKOS_REDIRECT_URL)} "
        f"WORKOS_OPENID_CONFIGURATION_URL={bool(WORKOS_OPENID_CONFIGURATION_URL)}"
    )
