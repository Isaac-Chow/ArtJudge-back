"""Repository functions for ArtJudge MongoDB collections.

Provides async CRUD helpers for users, sessions, and messages.
All functions are safe to call even if MongoDB is unavailable —
errors are logged and swallowed so the application stays responsive.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from db import get_db


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------


async def upsert_user(
    uid: str,
    email: Optional[str] = None,
    display_name: Optional[str] = None,
) -> None:
    """Create or update a user document.

    On first login, sets ``created_at``.  On every login, updates
    ``last_seen`` and any changed profile fields.
    """
    db = get_db()
    now = datetime.utcnow()

    update: Dict[str, Any] = {
        "$set": {
            "last_seen": now,
            "email": email,
            "display_name": display_name,
        },
        "$setOnInsert": {
            "created_at": now,
            "session_count": 0,
        },
    }

    try:
        await db.users.update_one({"uid": uid}, update, upsert=True)
    except Exception as exc:
        print(f"[repos] upsert_user error: {exc}")


async def get_all_users(
    skip: int = 0,
    limit: int = 50,
    sort_field: str = "last_seen",
    sort_dir: int = -1,
) -> List[Dict[str, Any]]:
    """Return a paginated list of users for the admin console."""
    db = get_db()
    try:
        cursor = (
            db.users.find({}, {"_id": 0})
            .sort(sort_field, sort_dir)
            .skip(skip)
            .limit(limit)
        )
        return await cursor.to_list(length=limit)
    except Exception as exc:
        print(f"[repos] get_all_users error: {exc}")
        return []


async def get_user_count() -> int:
    """Total number of registered users."""
    db = get_db()
    try:
        return await db.users.count_documents({})
    except Exception as exc:
        print(f"[repos] get_user_count error: {exc}")
        return 0


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------


async def ensure_session(session_id: str, uid: str, title: Optional[str] = None) -> None:
    """Create a session document if it does not already exist."""
    db = get_db()
    now = datetime.utcnow()

    set_on_insert: Dict[str, Any] = {
        "session_id": session_id,
        "uid": uid,
        "created_at": now,
        "message_count": 0,
    }
    if title:
        set_on_insert["title"] = title

    try:
        result = await db.sessions.update_one(
            {"session_id": session_id},
            {
                "$setOnInsert": set_on_insert,
                "$set": {"updated_at": now},
            },
            upsert=True,
        )
        # If this is a new session, increment the user's session_count
        if result.upserted_id is not None:
            await db.users.update_one(
                {"uid": uid}, {"$inc": {"session_count": 1}}
            )
    except Exception as exc:
        print(f"[repos] ensure_session error: {exc}")


async def set_session_title(session_id: str, title: str) -> None:
    """Update the display title for a session."""
    db = get_db()
    try:
        await db.sessions.update_one(
            {"session_id": session_id},
            {"$set": {"title": title, "updated_at": datetime.utcnow()}},
        )
    except Exception as exc:
        print(f"[repos] set_session_title error: {exc}")


async def get_user_sessions(uid: str) -> List[Dict[str, Any]]:
    """Return all sessions belonging to a user, newest first."""
    db = get_db()
    try:
        cursor = (
            db.sessions.find({"uid": uid}, {"_id": 0})
            .sort("created_at", -1)
        )
        return await cursor.to_list(length=200)
    except Exception as exc:
        print(f"[repos] get_user_sessions error: {exc}")
        return []


async def get_session_count() -> int:
    """Total number of sessions."""
    db = get_db()
    try:
        return await db.sessions.count_documents({})
    except Exception as exc:
        print(f"[repos] get_session_count error: {exc}")
        return 0


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------


async def save_message(
    session_id: str,
    uid: str,
    role: str,
    text: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Persist a single chat message."""
    db = get_db()
    doc: Dict[str, Any] = {
        "session_id": session_id,
        "uid": uid,
        "role": role,
        "text": text,
        "created_at": datetime.utcnow(),
    }
    if metadata:
        doc["metadata"] = metadata

    try:
        await db.messages.insert_one(doc)
        # Bump the session's message_count
        await db.sessions.update_one(
            {"session_id": session_id},
            {"$inc": {"message_count": 1}},
        )
    except Exception as exc:
        print(f"[repos] save_message error: {exc}")


async def get_session_messages(session_id: str) -> List[Dict[str, Any]]:
    """Return all messages in a session, oldest first."""
    db = get_db()
    try:
        cursor = (
            db.messages.find({"session_id": session_id}, {"_id": 0})
            .sort("created_at", 1)
        )
        return await cursor.to_list(length=500)
    except Exception as exc:
        print(f"[repos] get_session_messages error: {exc}")
        return []


async def get_message_count() -> int:
    """Total number of messages."""
    db = get_db()
    try:
        return await db.messages.count_documents({})
    except Exception as exc:
        print(f"[repos] get_message_count error: {exc}")
        return 0
