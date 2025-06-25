from typing import Optional
from sqlalchemy import String, JSON, DateTime, func, Integer, Index
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from fastapi_users.db import SQLAlchemyBaseUserTableUUID


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


class User(SQLAlchemyBaseUserTableUUID, Base):
    """
    This builds a standard FastAPI-Users User model
    plus any additional fields we want.
    (By default you only get id, email and password and some statuses)
    """

    name: Mapped[str] = mapped_column(String(64), nullable=False, server_default="")


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
    model_id: Mapped[str] = mapped_column(String, unique=True, index=True, nullable=False)
    name: Mapped[str] = mapped_column(String, nullable=False)
    json_data: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)


class Dataset(Base):
    """Dataset model."""

    __tablename__ = "dataset"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    dataset_id: Mapped[str] = mapped_column(String, unique=True, index=True, nullable=False)
    location: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    size: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    json_data: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True, server_default="{}")


class TrainingTemplate(Base):
    """Training template model."""

    __tablename__ = "training_template"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String, unique=True, index=True, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    type: Mapped[Optional[str]] = mapped_column(String, index=True, nullable=True)
    datasets: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    config: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    created_at: Mapped[DateTime] = mapped_column(DateTime, index=True, server_default=func.now(), nullable=False)
    updated_at: Mapped[DateTime] = mapped_column(
        DateTime, index=True, server_default=func.now(), onupdate=func.now(), nullable=False
    )


class Job(Base):
    """Job model."""

    __tablename__ = "job"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    job_data: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    status: Mapped[Optional[str]] = mapped_column(String, index=True, nullable=True)
    type: Mapped[Optional[str]] = mapped_column(String, index=True, nullable=True)
    experiment_id: Mapped[Optional[int]] = mapped_column(Integer, index=True, nullable=True)
    progress: Mapped[int] = mapped_column(Integer, nullable=False, server_default="-1")
    created_at: Mapped[DateTime] = mapped_column(DateTime, index=True, server_default=func.now(), nullable=False)
    updated_at: Mapped[DateTime] = mapped_column(
        DateTime, index=True, server_default=func.now(), onupdate=func.now(), nullable=False
    )

    __table_args__ = (Index("idx_experiment_type", "experiment_id", "type"),)


class Workflow(Base):
    """Workflow model."""

    __tablename__ = "workflows"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    config: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    status: Mapped[Optional[str]] = mapped_column(String, index=True, nullable=True)
    experiment_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    created_at: Mapped[DateTime] = mapped_column(DateTime, server_default=func.now(), nullable=False)
    updated_at: Mapped[DateTime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now(), nullable=False
    )

    __table_args__ = (Index("idx_workflow_id_experiment", "id", "experiment_id"),)


class WorkflowRun(Base):
    """Run of a workflow"""

    __tablename__ = "workflow_runs"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    workflow_id: Mapped[int] = mapped_column(Integer, nullable=True)
    workflow_name: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    job_ids: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    node_ids: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    status: Mapped[Optional[str]] = mapped_column(String, index=True, nullable=True)
    current_tasks: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    current_job_ids: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    experiment_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    created_at: Mapped[DateTime] = mapped_column(DateTime, server_default=func.now(), nullable=False)
    updated_at: Mapped[DateTime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now(), nullable=False
    )


class Task(Base):
    """Task model."""

    __tablename__ = "tasks"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    type: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    inputs: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    config: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    plugin: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    outputs: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    experiment_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    created_at: Mapped[DateTime] = mapped_column(DateTime, server_default=func.now(), nullable=False)
    updated_at: Mapped[DateTime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now(), nullable=False
    )


class NetworkMachine(Base):
    """Network machine model for distributed computing."""

    __tablename__ = "network_machines"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String, unique=True, index=True, nullable=False)
    host: Mapped[str] = mapped_column(String, nullable=False)
    port: Mapped[int] = mapped_column(Integer, nullable=False, server_default="8338")
    api_token: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    status: Mapped[Optional[str]] = mapped_column(String, index=True, nullable=True, server_default="offline")
    last_seen: Mapped[Optional[DateTime]] = mapped_column(DateTime, nullable=True)
    machine_metadata: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True, server_default="{}")
    # Reservation fields
    is_reserved: Mapped[bool] = mapped_column(nullable=False, server_default="0")
    reserved_by_host: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    reserved_at: Mapped[Optional[DateTime]] = mapped_column(DateTime, nullable=True)
    reservation_duration_minutes: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    reservation_metadata: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True, server_default="{}")
    created_at: Mapped[DateTime] = mapped_column(DateTime, server_default=func.now(), nullable=False)
    updated_at: Mapped[DateTime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now(), nullable=False
    )

    __table_args__ = (Index("idx_host_port", "host", "port"),)


class NetworkQuotaConfig(Base):
    """Network quota configuration model."""

    __tablename__ = "network_quota_config"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    host_identifier: Mapped[str] = mapped_column(String, nullable=False, index=True)
    time_period: Mapped[str] = mapped_column(String, nullable=False)  # 'daily', 'weekly', 'monthly', 'yearly'
    minutes_limit: Mapped[int] = mapped_column(Integer, nullable=False)
    warning_threshold_percent: Mapped[int] = mapped_column(Integer, nullable=False, server_default="80")
    is_active: Mapped[bool] = mapped_column(nullable=False, server_default="1")
    created_at: Mapped[DateTime] = mapped_column(DateTime, server_default=func.now(), nullable=False)
    updated_at: Mapped[DateTime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now(), nullable=False
    )

    __table_args__ = (Index("idx_host_period", "host_identifier", "time_period", unique=True),)


class NetworkQuotaUsage(Base):
    """Network quota usage tracking model."""

    __tablename__ = "network_quota_usage"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    host_identifier: Mapped[str] = mapped_column(String, nullable=False, index=True)
    time_period: Mapped[str] = mapped_column(String, nullable=False)
    period_start_date: Mapped[str] = mapped_column(String, nullable=False)  # 'YYYY-MM-DD' format
    minutes_used: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    last_updated: Mapped[DateTime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now(), nullable=False
    )
    created_at: Mapped[DateTime] = mapped_column(DateTime, server_default=func.now(), nullable=False)

    __table_args__ = (
        Index("idx_host_period_date", "host_identifier", "time_period", "period_start_date", unique=True),
    )


class NetworkQuotaHistory(Base):
    """Network quota usage history model for audit trail."""

    __tablename__ = "network_quota_history"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    host_identifier: Mapped[str] = mapped_column(String, nullable=False, index=True)
    machine_id: Mapped[int] = mapped_column(Integer, nullable=False)
    reservation_start: Mapped[DateTime] = mapped_column(DateTime, nullable=False)
    reservation_end: Mapped[Optional[DateTime]] = mapped_column(DateTime, nullable=True)
    minutes_used: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[DateTime] = mapped_column(DateTime, server_default=func.now(), nullable=False)

    __table_args__ = (Index("idx_host_machine_start", "host_identifier", "machine_id", "reservation_start"),)
