"""ArtJudge — FastAPI backend.

Provides REST endpoints for:
  - Chatting with the PydanticAI agent (Mistral + tools)
  - Streaming chat via SSE (typing animation)
  - Uploading an image for art-style classification
  - Firebase Auth token verification
  - Admin console API
  - Health check
"""

from __future__ import annotations

import base64
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from agent import run_agent
from auth import get_current_user, optional_user
from models import ChatResponse, UserMessage, UserInfo, ArtStyleClassification
from streaming import stream_chat_response
from tools.art_classifier import classify_art_style
import repositories as repo
from db import init_db, close_client
from admin import router as admin_router
from sessions import router as sessions_router

# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown)
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run database init on startup and cleanup on shutdown."""
    await init_db()
    yield
    close_client()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="ArtJudge API",
    description="AI art-style classifier and conversational agent powered by Mistral.",
    version="1.1.0",
    lifespan=lifespan,
)

# CORS — allow the Vite dev server and the Docker frontend container
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",   # Vite dev
        "http://localhost:3000",   # Docker frontend
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
        "https://artjudge-front.onrender.com"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(admin_router)
app.include_router(sessions_router)

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/api/health", tags=["meta"])
async def health():
    """Simple health check — no auth required."""
    return {"status": "ok", "service": "ArtJudge"}


@app.post("/api/auth/verify", response_model=UserInfo, tags=["auth"])
async def verify_token(user: UserInfo = Depends(get_current_user)):
    """Verify a Firebase ID token and return the user's basic info.

    Also upserts the user into MongoDB so the admin console can
    track who has signed in.
    """
    # Persist / update user record
    await repo.upsert_user(
        uid=user.uid, email=user.email, display_name=user.display_name
    )
    return user


@app.post("/api/chat", response_model=ChatResponse, tags=["agent"])
async def chat(
    body: UserMessage,
    user: UserInfo = Depends(get_current_user),
):
    """Send a text message (and optionally a base64 image) to the agent.

    Returns a structured ChatResponse that may include an art-style
    classification and/or web-search sources.
    """
    response = await run_agent(body)

    # Persist to MongoDB
    try:
        await repo.ensure_session(body.session_id, user.uid)
        await repo.save_message(body.session_id, user.uid, "user", body.text)
        agent_meta = {}
        if response.art_classification:
            agent_meta["art_classification"] = response.art_classification.model_dump()
        if response.sources:
            agent_meta["sources"] = response.sources
        await repo.save_message(
            body.session_id, user.uid, "agent", response.message,
            metadata=agent_meta or None,
        )
    except Exception as exc:
        print(f"[chat] MongoDB persist error: {exc}")

    return response


@app.post("/api/chat/stream", tags=["agent"])
async def chat_stream(
    body: UserMessage,
    user: UserInfo = Depends(get_current_user),
):
    """Stream a chat response via Server-Sent Events (SSE).

    The agent runs to completion, then streams the message text
    word-by-word for a typing animation effect.  Structured metadata
    (classification cards, sources) arrives as a final event.
    """
    generator = stream_chat_response(body, user)
    return StreamingResponse(
        generator,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # disable nginx buffering
        },
    )


@app.post("/api/classify", response_model=ArtStyleClassification, tags=["agent"])
async def classify_image(
    image: UploadFile = File(...),
    session_id: str = Form("default"),
    user: UserInfo = Depends(get_current_user),
):
    """Upload an image and classify its art style.

    Accepts multipart/form-data with an `image` file field.
    Returns a structured ArtStyleClassification.
    """
    # Validate file type
    allowed = {"image/png", "image/jpeg", "image/jpg", "image/webp"}
    if image.content_type not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image type '{image.content_type}'. Use PNG, JPEG, or WebP.",
        )

    # Read and encode
    contents = await image.read()
    image_b64 = base64.b64encode(contents).decode("utf-8")

    result = classify_art_style(image_b64)
    return ArtStyleClassification(**result)
