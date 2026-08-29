from __future__ import annotations

import os
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, Query, status

from auth import get_current_user
from models import UserInfo
import repositories as repo

router = APIRouter(prefix="/api/admin", tags=["admin"])


async def get_admin_user(user: UserInfo = Depends(get_current_user)) -> UserInfo:
    """FastAPI dependency that ensures the caller is the configured admin.

    Raises 403 if the authenticated user's email does not match
    the ``ADMIN_EMAIL`` environment variable.
    """
    if not _ADMIN_EMAIL:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ADMIN_EMAIL is not configured on the server.",
        )

    if user.email != _ADMIN_EMAIL:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied. Admin privileges required.",
        )

    return user

    
@router.get("/stats")
async def admin_stats(
    _admin: UserInfo = Depends(get_admin_user),
) -> Dict[str, int]:
    """Return aggregate counts: total users, sessions, and messages."""
    return {
        "users": await repo.get_user_count(),
        "sessions": await repo.get_session_count(),
        "messages": await repo.get_message_count(),
    }


@router.get("/users")
async def admin_list_users(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    sort: str = Query("last_seen"),
    _admin: UserInfo = Depends(get_admin_user),
) -> List[Dict[str, Any]]:
    """Return a paginated list of all registered users."""
    return await repo.get_all_users(skip=skip, limit=limit, sort_field=sort)


@router.get("/users/{uid}/sessions")
async def admin_user_sessions(
    uid: str,
    _admin: UserInfo = Depends(get_admin_user),
) -> List[Dict[str, Any]]:
    """Return all sessions belonging to a specific user."""
    return await repo.get_user_sessions(uid)


@router.get("/sessions/{session_id}/messages")
async def admin_session_messages(
    session_id: str,
    _admin: UserInfo = Depends(get_admin_user),
) -> List[Dict[str, Any]]:
    """Return all messages in a specific session."""
    return await repo.get_session_messages(session_id)
