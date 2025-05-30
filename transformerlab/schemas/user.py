"""
User schemas for FastAPI Users integration.

Any fields we've added to the default FastAPI Users schema need to be added here.
"""

import uuid
from typing import Optional

from fastapi_users import schemas


class UserRead(schemas.BaseUser[uuid.UUID]):
    name: str


class UserCreate(schemas.BaseUserCreate):
    name: Optional[str] = ""


class UserUpdate(schemas.BaseUserUpdate):
    name: str
