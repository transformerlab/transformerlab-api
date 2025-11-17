# database.py
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.ext.declarative import DeclarativeMeta, declarative_base
from sqlalchemy.orm import sessionmaker
from fastapi_users.db import SQLAlchemyBaseUserTableUUID
from sqlalchemy import Column, String, ForeignKey
import uuid

# Replace with your actual database URL (e.g., PostgreSQL, SQLite)
from transformerlab.db.constants import DATABASE_FILE_NAME, DATABASE_URL

Base: DeclarativeMeta = declarative_base()


# 1. Define your User Model (inherits from a FastAPI Users base class)
class User(SQLAlchemyBaseUserTableUUID, Base):
    pass  # You can add custom fields here later, like 'first_name: str'


# 2. Define Team Model
class Team(Base):
    __tablename__ = "teams"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False)


# 3. Define User-Team Association Model
class UserTeam(Base):
    __tablename__ = "users_teams"

    user_id = Column(String, ForeignKey("users.id"), primary_key=True)
    team_id = Column(String, ForeignKey("teams.id"), primary_key=True)


# 2. Setup the Async Engine and Session
engine = create_async_engine(DATABASE_URL)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


# 3. Utility to create tables (run this on app startup)
async def create_db_and_tables():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


# 4. Database session dependency
async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        yield session


# 5. Create default team if not exists
async def create_default_team(session: AsyncSession) -> Team:
    stmt = select(Team).where(Team.name == "Default Team")
    result = await session.execute(stmt)
    team = result.scalar_one_or_none()
    if not team:
        team = Team(name="Default Team")
        session.add(team)
        await session.commit()
        await session.refresh(team)
    return team
