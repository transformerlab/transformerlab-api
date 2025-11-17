from fastapi import APIRouter, Depends, HTTPException
from transformerlab.shared.models.user_model import User
from transformerlab.models.users import (
    fastapi_users,
    auth_backend,
    current_active_user,
    UserRead,
    UserCreate,
    UserUpdate,
    get_user_manager,
    get_refresh_strategy,
    jwt_authentication,
)

from jose import jwt, JWTError

router = APIRouter(tags=["users"])


# Include Auth and Registration Routers
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
# Include User Management Router (allows authenticated users to view/update their profile)
router.include_router(
    fastapi_users.get_users_router(UserRead, UserUpdate),
    prefix="/users",
    tags=["users"],
)


@router.get("/test-users/authenticated-route")
async def authenticated_route(user: User = Depends(current_active_user)):
    return {"message": f"Hello, {user.email}! You are authenticated."}


# To test this, register a new user via /auth/register
# curl -X POST 'http://127.0.0.1:8338/auth/register' \
#  -H 'Content-Type: application/json' \
#  -d '{
#    "email": "test@example.com",
#    "password": "password123"
# }'

# Then
# curl -X POST 'http://127.0.0.1:8338/auth/jwt/login' \
#  -H 'Content-Type: application/x-www-form-urlencoded' \
#  -d 'username=test@example.com&password=password123'

# This will return a token, which you can use to access the authenticated route:
# curl -X GET 'http://127.0.0.1:8338/authenticated-route' \
#  -H 'Authorization: Bearer <YOUR_ACCESS_TOKEN>'


@router.post("/auth/refresh")
async def refresh_access_token(
    refresh_token: str,  # Sent by the client in the request body
    user_manager=Depends(get_user_manager),
):
    try:
        # 1. Decode and Validate the Refresh Token
        # Get a fresh refresh strategy instance and use its secret to decode
        refresh_strategy = get_refresh_strategy()
        payload = jwt.decode(refresh_token, str(refresh_strategy.secret), algorithms=["HS256"])
        user_id = payload.get("sub")

        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid refresh token payload")

        # 2. Get the user object from the database
        user = await user_manager.get(user_id)
        if user is None or not user.is_active:
            raise HTTPException(status_code=401, detail="User inactive or not found")

        # 3. Create a NEW Access Token (using the short-lived strategy from the main JWT)
        new_access_token = jwt_authentication.get_login_response(user)  # Needs custom helper

        return {"access_token": new_access_token["access_token"], "token_type": "bearer"}

    except JWTError:
        raise HTTPException(status_code=401, detail="Expired or invalid refresh token")


@router.get("/users/me/teams")
async def get_user_teams(user: User = Depends(current_active_user)):
    # Placeholder implementation
    # In a real application, fetch teams from the database
    return {
        "user_id": user.id,
        "teams": [
            {"id": "550e8400-e29b-41d4-a716-446655440000", "name": "Transformer Lab"},
            {"id": "6ba7b810-9dad-11d1-80b4-00c04fd430c8", "name": "Team 2"},
        ],
    }
