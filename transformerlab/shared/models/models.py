from typing import Optional
from sqlalchemy import String, JSON, DateTime, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Config(Base):
    """Configuration key-value store model."""

    __tablename__ = "config"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    key: Mapped[str] = mapped_column(String, unique=True, index=True, nullable=False)
    value: Mapped[Optional[str]] = mapped_column(String, nullable=True)


# I believe we are not using the following table anymore as the filesystem
# is being used to track plugins
class Plugin(Base):
    """Plugin definition model."""

    __tablename__ = "plugins"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String, unique=True, index=True, nullable=False)
    type: Mapped[str] = mapped_column(String, index=True, nullable=False)


class Experiment(Base):
    """Experiment model."""

    __tablename__ = "experiment"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String, unique=True, index=True, nullable=False)
    config: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[DateTime] = mapped_column(DateTime, server_default=func.now(), nullable=False)
    updated_at: Mapped[DateTime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now(), nullable=False
    )


class Model(Base):
    """Model definition."""

    __tablename__ = "model"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    model_id: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String, nullable=False)
    json_data: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
