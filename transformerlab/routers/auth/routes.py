from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import RedirectResponse

from transformerlab.schemas.auth import UserResponse
from transformerlab.services.auth import AuthenticatedIdentity, auth_service
from transformerlab.shared.s3_mount import setup_user_s3_mount

from .api_key_auth import get_user_or_api_key

router = APIRouter(prefix="/auth", tags=["auth"])


@router.get("/login-url")
async def get_login_url(request: Request, provider: Optional[str] = None):
    try:
        login_url = auth_service.generate_login_url(request, provider=provider)
        return {"login_url": login_url}
    except Exception as exc:  # pragma: no cover - provider-specific errors
        raise HTTPException(status_code=500, detail=f"Failed to generate login URL: {exc}") from exc


@router.get("/callback", name="auth_callback")
async def auth_callback(request: Request, code: str):
    try:
        return await auth_service.handle_auth_callback(request, code)
    except HTTPException:
        raise
    except Exception:
        error_url = auth_service.get_frontend_error_url(request)
        return RedirectResponse(url=error_url)


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    identity: AuthenticatedIdentity = Depends(get_user_or_api_key),
):
    payload = auth_service.get_user_info(identity)

    # Setup S3 mount if user is authenticated and multitenant mode is enabled
    # This ensures the mount is restored even if the user is already logged in
    try:
        if payload.get("authenticated") and payload.get("source") == "session":
            user_id = payload.get("id")
            organization_id = payload.get("organization_id")

            if user_id:
                # success = setup_user_s3_mount(str(user_id), organization_id)
                success = False
                if not success:
                    print("S3 mount setup failed via auth/me")
    except Exception as exc:
        # Don't let S3 mount errors break the auth/me endpoint
        print(f"Error setting up S3 mount for user: {exc}")

    return UserResponse(**payload)


@router.get("/logout")
async def logout(request: Request):
    return await auth_service.logout_user(request)


@router.get("/allowed-scopes")
async def get_allowed_scopes():
    return {"scopes": auth_service.get_allowed_scopes()}


@router.get("/check")
async def check_auth(
    request: Request,
    response: Response,
    identity: AuthenticatedIdentity = Depends(get_user_or_api_key),
):
    return await auth_service.check_user_auth(request, response, identity)


@router.post("/refresh")
async def refresh_session(request: Request, response: Response):
    try:
        return await auth_service.refresh_user_session(request, response)
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - provider-specific errors
        raise HTTPException(status_code=500, detail=f"Failed to refresh session: {exc}") from exc
