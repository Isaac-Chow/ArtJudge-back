"""Server-Sent Events (SSE) streaming for ArtJudge chat.

Runs the PydanticAI agent to completion, then streams the response
message word-by-word as SSE ``text`` events.  Structured metadata
(art classification, sources) is sent as a final ``metadata`` event
so the frontend can render rich cards after the typing animation.

SSE protocol
------------
``event: thinking``  -- ``{"status": "..."}`` (processing feedback)
``event: start``     -- ``{}`` (stream begins)
``event: text``      -- ``{"delta": "word "}``  (incremental text)
``event: metadata``  -- full JSON with art_classification & sources
``event: title``     -- ``{"title": "..."}`` (auto-generated session title)
``event: done``      -- ``{}`` (stream complete)
``event: error``     -- ``{"detail": "..."}`` (something went wrong)
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, AsyncGenerator, Dict, Optional

from models import ChatResponse, UserMessage, UserInfo
from agent import run_agent
import repositories as repo

# Maximum seconds to wait for the agent before returning an error.
_AGENT_TIMEOUT = 120


# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------


def _sse_event(event: str, data: Dict[str, Any]) -> str:
    """Format a single SSE frame."""
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


# ---------------------------------------------------------------------------
# Streaming generator
# ---------------------------------------------------------------------------


async def stream_chat_response(
    body: UserMessage,
    user: UserInfo,
    *,
    words_per_chunk: int = 2,
    delay_seconds: float = 0.04,
) -> AsyncGenerator[str, None]:
    """Yield SSE frames that stream the agent's reply.

    1. Send an immediate ``thinking`` event so the UI shows feedback instantly.
    2. Run the agent with a timeout.
    3. Send a ``start`` event.
    4. Stream ``response.message`` word-by-word.
    5. Send a ``metadata`` event with classification / sources.
    6. Send a ``done`` event.
    7. Persist the exchange to MongoDB.
    """

    # --- 1. Immediate feedback -----------------------------------------------
    yield _sse_event("thinking", {"status": "Analysing your request..."})

    # --- 2. Run the agent (with timeout) ------------------------------------
    t0 = time.perf_counter()
    try:
        response: ChatResponse = await asyncio.wait_for(
            run_agent(body), timeout=_AGENT_TIMEOUT
        )
    except asyncio.TimeoutError:
        elapsed = time.perf_counter() - t0
        print(f"[streaming] Agent timed out after {elapsed:.1f}s")
        yield _sse_event("error", {"detail": "The agent took too long to respond. Please try again."})
        yield _sse_event("done", {})
        return
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        print(f"[streaming] Agent error after {elapsed:.1f}s: {exc}")
        yield _sse_event("error", {"detail": f"An error occurred: {exc}"})
        yield _sse_event("done", {})
        return

    agent_elapsed = time.perf_counter() - t0
    print(f"[streaming] Agent completed in {agent_elapsed:.1f}s")

    # --- 2. Start event -----------------------------------------------------
    yield _sse_event("start", {})

    # --- 3. Stream text word-by-word ---------------------------------------
    words = response.message.split(" ")
    for i in range(0, len(words), words_per_chunk):
        chunk = " ".join(words[i: i + words_per_chunk])
        # Add a trailing space unless it's the last chunk
        if i + words_per_chunk < len(words):
            chunk += " "
        yield _sse_event("text", {"delta": chunk})
        await asyncio.sleep(delay_seconds)

    # --- 4. Metadata --------------------------------------------------------
    meta: Dict[str, Any] = {}
    if response.art_classification:
        meta["art_classification"] = response.art_classification.model_dump()
    if response.sources:
        meta["sources"] = response.sources
    if meta:
        yield _sse_event("metadata", meta)

    # --- 5. Done ------------------------------------------------------------
    yield _sse_event("done", {})

    # --- 6. Persist to MongoDB & generate title ---------------------------------
    try:
        await repo.ensure_session(body.session_id, user.uid)

        # Save user message WITH image if present (for history retrieval)
        user_meta: Optional[Dict[str, Any]] = None
        if body.image_base64:
            user_meta = {"image": body.image_base64}
        await repo.save_message(
            body.session_id, user.uid, "user", body.text, metadata=user_meta
        )
        agent_meta: Optional[Dict[str, Any]] = None
        if response.art_classification:
            agent_meta = {
                "art_classification": response.art_classification.model_dump()
            }
        if response.sources:
            agent_meta = agent_meta or {}
            agent_meta["sources"] = response.sources
        await repo.save_message(
            body.session_id,
            user.uid,
            "agent",
            response.message,
            metadata=agent_meta,
        )

        # Generate title on first exchange (message_count <= 2 means just this pair)
        from db import get_db
        session_doc = await get_db().sessions.find_one(
            {"session_id": body.session_id}
        )
        if session_doc and session_doc.get("message_count", 0) <= 2:
            # Generate title in the background and send via SSE
            try:
                title = await asyncio.wait_for(
                    _gen_title(body.session_id, body.text), timeout=15
                )
                yield _sse_event("title", {"title": title})
            except Exception as exc:
                print(f"[streaming] Title generation error: {exc}")

    except Exception as exc:
        print(f"[streaming] MongoDB persist error: {exc}")


async def _gen_title(session_id: str, user_text: str) -> str:
    """Generate and persist a session title, returning it."""
    from sessions import generate_session_title
    title = await generate_session_title(user_text)
    await repo.set_session_title(session_id, title)
    return title
