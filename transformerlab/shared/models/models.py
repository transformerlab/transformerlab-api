# ruff: noqa: F401
from sqlmodel import SQLModel, Field
from typing import Optional


class Config(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    key: str = Field(unique=True, index=True)
    value: Optional[str] = None


class Plugin(SQLModel, table=True):
    id: int = Field(default=None, primary_key=True)
    name: str = Field(unique=True, index=True)
    type: str = Field(index=True)
