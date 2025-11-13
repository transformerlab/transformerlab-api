# database.py
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.ext.declarative import DeclarativeMeta, declarative_base
from sqlalchemy.orm import sessionmaker
from fastapi_users.db import SQLAlchemyBaseUserTableUUID

# Replace with your actual database URL (e.g., PostgreSQL, SQLite)
from transformerlab.db.constants import DATABASE_FILE_NAME, DATABASE_URL

Base: DeclarativeMeta = declarative_base()


# 1. Define your User Model (inherits from a FastAPI Users base class)
class User(SQLAlchemyBaseUserTableUUID, Base):
    pass  # You can add custom fields here later, like 'first_name: str'


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
