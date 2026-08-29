"""User-facing session API and title generation for ArtJudge.

Provides endpoints for authenticated users to:
  - List their chat sessions (with auto-generated titles)
  - Retrieve messages for a specific session

Also provides a background title-generation helper that runs a
lightweight Mistral call to produce a short, descriptive title
from the user's first message.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException
from pydantic_ai.models.mistral import MistralModel
from dotenv import load_dotenv

from auth import get_current_user
from models import UserInfo
import repositories as repo

load_dotenv()

router = APIRouter(prefix="/api/sessions", tags=["sessions"])

# Lightweight model for quick title generation (fast + cheap)
_title_model = MistralModel("mistral-small-latest")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("")
async def list_sessions(
    user: UserInfo = Depends(get_current_user),
) -> List[Dict[str, Any]]:
    """Return all chat sessions for the authenticated user, newest first.

    Each session includes: session_id, title (auto-generated),
    created_at, updated_at, and message_count.
    """
    return await repo.get_user_sessions(user.uid)


@router.get("/{session_id}/messages")
async def get_messages(
    session_id: str,
    user: UserInfo = Depends(get_current_user),
) -> List[Dict[str, Any]]:
    """Return all messages in a session, oldest first.

    Only the session owner can access their messages.
    """
    # Verify ownership — get all sessions and check
    sessions = await repo.get_user_sessions(user.uid)
    owned_ids = {s["session_id"] for s in sessions}

    if session_id not in owned_ids:
        raise HTTPException(status_code=404, detail="Session not found.")

    return await repo.get_session_messages(session_id)


# ---------------------------------------------------------------------------
# Background title generation
# ---------------------------------------------------------------------------


async def generate_session_title(user_message: str) -> str:
    """Generate a short, descriptive title for a chat session.

    Uses a lightweight Mistral call to produce a title (max ~8 words)
    from the user's first message.  Returns a fallback if generation fails.
    """
    prompt = (
        "Generate a very short title (2-8 words) for a chat conversation "
        "that starts with this user message. Return ONLY the title text, "
        "no quotes or punctuation:\n\n"
        f'"{user_message}"'
    )

    try:
        from pydantic_ai import Agent

        title_agent = Agent(
            _title_model,
            output_type=str,
            system_prompt="You generate short, descriptive chat titles. Reply with 2-8 words only.",
        )
        result = await asyncio.wait_for(
            title_agent.run(prompt), timeout=15
        )

        title = result.output.strip().strip('"').strip("'")
        # Truncate to reasonable length
        if len(title) > 60:
            title = title[:57] + "..."
        return title or "New Chat"
    except Exception as exc:
        print(f"[sessions] Title generation failed: {exc}")
        # Fallback: use the first few words of the user message
        words = user_message.split()[:6]
        return " ".join(words) + ("..." if len(words) >= 6 else "") or "New Chat"


async def generate_and_save_title(session_id: str, user_message: str) -> None:
    """Generate a title and save it to the session.  Fire-and-forget."""
    title = await generate_session_title(user_message)
    await repo.set_session_title(session_id, title)
    print(f"[sessions] Title for {session_id}: '{title}'")
