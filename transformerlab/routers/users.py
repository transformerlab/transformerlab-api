import os
import logging
from fastapi import APIRouter

# from .user_service import auth_backend, current_active_user, fastapi_users

# FastAPI Users integration
from transformerlab.services.user_service import auth_backend, fastapi_users, SECRET, get_jwt_strategy
from transformerlab.services.redirect_transport import RedirectBearerTransport
from fastapi_users.authentication import AuthenticationBackend
from transformerlab.schemas.user import UserCreate, UserRead, UserUpdate

logger = logging.getLogger(__name__)

router = APIRouter()

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
else:
    logger.warning(
        "Skipping WorkOS OAuth routes. Missing vars: "
        f"WORKOS_CLIENT_ID={bool(WORKOS_CLIENT_ID)} "
        f"WORKOS_CLIENT_SECRET={bool(WORKOS_CLIENT_SECRET)} "
        f"WORKOS_REDIRECT_URL={bool(WORKOS_REDIRECT_URL)} "
        f"WORKOS_OPENID_CONFIGURATION_URL={bool(WORKOS_OPENID_CONFIGURATION_URL)}"
    )
