# users.py
import uuid
from typing import Optional, AsyncGenerator
from fastapi import Depends, Request
from fastapi_users import BaseUserManager, FastAPIUsers, UUIDIDMixin, schemas
from fastapi_users.authentication import AuthenticationBackend, BearerTransport, JWTStrategy
from fastapi_users.db import SQLAlchemyUserDatabase
from transformerlab.shared.models.user_model import User, get_async_session
from sqlalchemy.ext.asyncio import AsyncSession


# --- Pydantic Schemas for API interactions ---
class UserRead(schemas.BaseUser[uuid.UUID]):
    # Add your custom fields here if you added them to the User model
    pass


class UserCreate(schemas.BaseUserCreate):
    pass


class UserUpdate(schemas.BaseUserUpdate):
    pass


# --- User Manager (Handles registration, password reset, etc.) ---
SECRET = "YOUR_STRONG_SECRET"  # !! CHANGE THIS IN PRODUCTION !!


class UserManager(UUIDIDMixin, BaseUserManager[User, uuid.UUID]):
    reset_password_token_secret = SECRET
    verification_token_secret = SECRET

    # Optional: Define custom logic after registration
    async def on_after_register(self, user: User, request: Optional[Request] = None):
        print(f"User {user.id} has registered.")


async def get_user_db(session: AsyncSession = Depends(get_async_session)):
    yield SQLAlchemyUserDatabase(session, User)


async def get_user_manager(user_db: SQLAlchemyUserDatabase = Depends(get_user_db)):
    yield UserManager(user_db)


# --- Authentication Backend (JWT/Bearer Token) ---
bearer_transport = BearerTransport(tokenUrl="auth/jwt/login")


def get_jwt_strategy() -> JWTStrategy:
    # Token lasts for 3600 seconds (1 hour)
    return JWTStrategy(secret=SECRET, lifetime_seconds=3600)


auth_backend = AuthenticationBackend(
    name="jwt",
    transport=bearer_transport,
    get_strategy=get_jwt_strategy,
)

# --- FastAPIUsers Instance (The main utility) ---
fastapi_users = FastAPIUsers[User, uuid.UUID](
    get_user_manager,
    [auth_backend],  # Add more backends (like Google OAuth) here
)

# --- Dependency for Protected Routes ---
# This is what you'll use in your route decorators
current_active_user = fastapi_users.current_user(active=True)
