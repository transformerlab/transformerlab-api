from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from transformerlab.shared.models.user_model import User, Team, UserTeam, get_async_session
from transformerlab.models.users import current_active_user
from pydantic import BaseModel
from sqlalchemy import select, delete, update


class TeamCreate(BaseModel):
    name: str


class TeamUpdate(BaseModel):
    name: str


class TeamResponse(BaseModel):
    id: str
    name: str


router = APIRouter(tags=["teams"])


@router.post("/teams", response_model=TeamResponse)
async def create_team(
    team_data: TeamCreate,
    session: AsyncSession = Depends(get_async_session),
    user: User = Depends(current_active_user),
):
    # Create team
    team = Team(name=team_data.name)
    session.add(team)
    await session.commit()
    await session.refresh(team)

    # Add user to the team
    user_team = UserTeam(user_id=str(user.id), team_id=team.id)
    session.add(user_team)
    await session.commit()

    return TeamResponse(id=team.id, name=team.name)


@router.put("/teams/{team_id}", response_model=TeamResponse)
async def update_team(
    team_id: str,
    team_data: TeamUpdate,
    session: AsyncSession = Depends(get_async_session),
    user: User = Depends(current_active_user),
):
    # Check if user is in the team
    stmt = select(UserTeam).where(UserTeam.user_id == str(user.id), UserTeam.team_id == team_id)
    result = await session.execute(stmt)
    if not result.scalar_one_or_none():
        raise HTTPException(status_code=403, detail="Not authorized to update this team")

    # Update
    stmt = update(Team).where(Team.id == team_id).values(name=team_data.name)
    await session.execute(stmt)
    await session.commit()

    # Fetch updated
    stmt = select(Team).where(Team.id == team_id)
    result = await session.execute(stmt)
    team = result.scalar_one()

    return TeamResponse(id=team.id, name=team.name)


@router.delete("/teams/{team_id}")
async def delete_team(
    team_id: str,
    session: AsyncSession = Depends(get_async_session),
    user: User = Depends(current_active_user),
):
    # Check if user is in the team
    stmt = select(UserTeam).where(UserTeam.user_id == str(user.id), UserTeam.team_id == team_id)
    result = await session.execute(stmt)
    if not result.scalar_one_or_none():
        raise HTTPException(status_code=403, detail="Not authorized to delete this team")

    # Check if user has other teams
    stmt = select(UserTeam).where(UserTeam.user_id == str(user.id))
    result = await session.execute(stmt)
    user_teams = result.scalars().all()
    if len(user_teams) <= 1:
        raise HTTPException(status_code=400, detail="Cannot delete the last team")

    # Check if team has only this user
    stmt = select(UserTeam).where(UserTeam.team_id == team_id)
    result = await session.execute(stmt)
    team_users = result.scalars().all()
    if len(team_users) > 1:
        raise HTTPException(status_code=400, detail="Cannot delete team with multiple users")

    # Delete associations and team
    stmt = delete(UserTeam).where(UserTeam.team_id == team_id)
    await session.execute(stmt)
    stmt = delete(Team).where(Team.id == team_id)
    await session.execute(stmt)
    await session.commit()

    return {"message": "Team deleted"}