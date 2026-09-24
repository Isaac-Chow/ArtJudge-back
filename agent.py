from __future__ import annotations

import base64
import os
import time
from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field
from pydantic_ai import Agent, BinaryContent, RunContext
from pydantic_ai.models.mistral import MistralModel
from dotenv import load_dotenv

from models import ArtStyleClassification, ChatResponse, UserMessage
from tools.web_search import search_web

load_dotenv()

class AgentDependencies(BaseModel):
    """Dependencies for the agent."""

    session_id: str
    conversion_history: List[dict] = Field(default_factory=list)
    session_start: datetime = Field(default_factory=datetime.now)
    search_urls: List[str] = Field(default_factory=list, description="List of URLs to search for information.")

model = MistralModel('ministral-8b-2512')
agent = Agent(
    model,
    deps_type=AgentDependencies,
    output_type=ChatResponse,
    system_prompt=(
        "You are ArtJudge, an expert art educator AI with deep knowledge of art history "
        "and styles. You help students understand art styles by analysing images they "
        "submit and answering questions about art history, techniques, and famous artists.\n\n"
        "Guidelines:\n"
        "- When a user submits an image, analyse it directly to identify the art style. "
        "Provide the classification in the art_classification field of your response, including "
        "the style name, confidence, historical period, a description, key visual features, "
        "and notable artists of that movement. Then explain the result conversationally.\n"
        "- If the user asks about art history or techniques, use the search_web tool "
        "to find reliable information and cite your sources.\n"
        "- Always be encouraging and help students appreciate different art movements.\n"
        "- Keep responses concise but informative (3-6 sentences for chat, more detail "
        "when explaining a classification).\n"
        "- If you are unsure about something, say so honestly."
    ),
)

@agent.tool
async def search_web_tool(
    ctx: RunContext[AgentDependencies],
    query: str,
) -> str:
    """Search the web for information about art, artists, or art movements.

    Args:
        query: The search query (e.g. 'Impressionism origins', 'Who painted Guernica?').
    """
    result = search_web(query)

    lines: List[str] = []
    urls: List[str] = []
    if result.get("summary"):
        lines.append(f"Summary: {result['summary']}")

    for i, r in enumerate(result.get("results", []), 1):
        lines.append(
            f"{i}. {r['title']}\n   URL: {r['url']}\n   {r['snippet']}")
        urls.append(r['url'])

    # Track URLs for automatic source population
    if urls:
        ctx.deps.search_urls.extend(urls)

    if not lines:
        return f"No results found for '{query}'."

    return f"Search results for '{query}':\n" + "\n".join(lines)


_sessions: Dict[str, AgentDependencies] = {}


def get_or_create_session(session_id: str) -> AgentDependencies:
    """Return existing session deps or create a new one."""
    if session_id not in _sessions:
        _sessions[session_id] = AgentDependencies(session_id=session_id)
    return _sessions[session_id]

def expire_old_sessions(max_age_minutes: int = 60) -> None:
    """Remove sessions older than max_age_minutes to free memory."""
    now = datetime.now()
    expired = [
        sid
        for sid, deps in _sessions.items()
        if (now - deps.session_start).total_seconds() > max_age_minutes * 60
    ]
    for sid in expired:
        del _sessions[sid]

async def run_agent(
    user_msg_or_session_id: UserMessage | str,
    maybe_user_msg: Optional[UserMessage] = None,
    *,
    session_id: Optional[str] = None,
) -> ChatResponse:
    """Run the agent for a single user message and return the structured response.

    Supports both the current call pattern `run_agent(user_msg)` and the older
    positional style `run_agent(session_id, user_msg)` for compatibility.
    """
    if isinstance(user_msg_or_session_id, str):
        if maybe_user_msg is None or not isinstance(maybe_user_msg, UserMessage):
            raise TypeError("run_agent requires a UserMessage when called with a session id.")
        effective_session_id = user_msg_or_session_id
        user_msg = maybe_user_msg
    else:
        if not isinstance(user_msg_or_session_id, UserMessage):
            raise TypeError("run_agent expects a UserMessage object.")
        user_msg = user_msg_or_session_id
        effective_session_id = session_id or user_msg.session_id

    deps = get_or_create_session(effective_session_id)

    content: list = [user_msg.text]
    if user_msg.image_base64:
        b64_data = user_msg.image_base64
        media_type = "image/jpeg"
        if "," in b64_data[:60]:
            header, b64_data = b64_data.split(",", 1)
            if "image/png" in header:
                media_type = "image/png"
            elif "image/webp" in header:
                media_type = "image/webp"
        image_bytes = base64.b64decode(b64_data)
        content.append(BinaryContent(media_type=media_type, data=image_bytes))
        print(f"[agent] Received image of {len(image_bytes)} bytes for session {effective_session_id}")

    # Append user message test to history
    deps.conversion_history.append({"role": "user", "content": user_msg.text})
    
    # Reset search URLs for this turn
    deps.search_urls=[]

    try:
        t0 = time.perf_counter()
        result = await agent.run(content, deps=deps)
        elapsed = time.perf_counter() - t0
        print(f"[agent] Completed in {elapsed:.1f}s for session {effective_session_id}")

        if hasattr(result, "output") and isinstance(result.output, ChatResponse):
            response = result.output
        elif isinstance(result, ChatResponse):
            response = result
        else:
            response = ChatResponse(
                message=str(getattr(result, "output", result)))

        if not response.sources and deps.search_urls:
            response.sources = deps.search_urls[:5]
            print(f"[agent] Populated sources from search URLs: {response.sources}")
        
    except Exception as exc:
        response = ChatResponse(
            message=f"An error occurred while processing your request: {exc}",
            confidence=0.0,
        )

    deps.conversion_history.append({"role": "assistant", "content": response.message})
    expire_old_sessions()
    return response