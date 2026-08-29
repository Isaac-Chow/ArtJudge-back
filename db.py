"""MongoDB async client and database initialisation for ArtJudge.

Uses motor (async MongoDB driver) for non-blocking database operations.
The MONGO_URI environment variable controls the connection string;
defaults to a local MongoDB instance for development.
"""

from __future__ import annotations

import os
from typing import Optional

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

# ---------------------------------------------------------------------------
# Client singleton
# ---------------------------------------------------------------------------

_client: Optional[AsyncIOMotorClient] = None
_db: Optional[AsyncIOMotorDatabase] = None

_MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
_DB_NAME = os.getenv("MONGO_DB_NAME", "artjudge")


def get_client() -> AsyncIOMotorClient:
    """Return the singleton MongoDB client (creates on first call)."""
    global _client
    if _client is None:
        _client = AsyncIOMotorClient(_MONGO_URI, serverSelectionTimeoutMS=5000)
    return _client


def get_db() -> AsyncIOMotorDatabase:
    """Return the artjudge database handle."""
    global _db
    if _db is None:
        _db = get_client()[_DB_NAME]
    return _db


# ---------------------------------------------------------------------------
# Startup: create indexes
# ---------------------------------------------------------------------------


async def init_db() -> None:
    """Create indexes on application startup.

    Called from the FastAPI lifespan handler.  Safe to call multiple times —
    create_index is idempotent.
    """
    db = get_db()

    # Users collection
    await db.users.create_index("uid", unique=True)
    await db.users.create_index("email")

    # Sessions collection
    await db.sessions.create_index("session_id", unique=True)
    await db.sessions.create_index("uid")
    await db.sessions.create_index("created_at")

    # Messages collection
    await db.messages.create_index("session_id")
    await db.messages.create_index([("session_id", 1), ("created_at", 1)])

    print(f"[db] MongoDB initialised — database '{_DB_NAME}' ready")


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------


def close_client() -> None:
    """Close the MongoDB client connection (called on shutdown)."""
    global _client, _db
    if _client is not None:
        _client.close()
        _client = None
        _db = None
