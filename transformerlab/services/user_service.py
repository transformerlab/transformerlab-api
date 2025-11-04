from __future__ import annotations

from typing import Optional

from fastapi import Request

from transformerlab.routers.auth.provider.auth_provider import AuthUser
from transformerlab.shared.s3_mount import setup_user_s3_mount


class UserService:
    async def on_after_login(
        self,
        user: AuthUser,
        request: Optional[Request] = None,
        response: Optional[object] = None,
    ) -> None:
        """Called after a user successfully logs in."""
        try:
            print(f"User {user.id} has logged in. Setting up S3 mount if needed.")
            # Prefer organization id from user, fallback to cookie
            organization_id = getattr(user, "organization_id", None)
            print(f"Organization ID: {organization_id}")
            # success = setup_user_s3_mount(str(user.id), organization_id)
            success = False
            if success:
                print(f"S3 mount setup completed for user {user.id}")
            else:
                print(f"S3 mount setup failed for user {user.id}")
        except Exception as exc:
            print(f"Error setting up S3 mount for user {user.id}: {exc}")


user_service = UserService()
