from fastapi import APIRouter, Depends
from transformerlab.shared.models.user_model import User
from transformerlab.models.users import (
    current_active_user,
)


router = APIRouter(prefix="/test_users", tags=["users"])


@router.get("/authenticated-route")
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
