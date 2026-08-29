"""Pydantic models for structured responses in the ArtJudge application."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class ArtStyleClassification(BaseModel):
    """Structured result from the art style classifier tool.

    Provides the detected art style along with educational context
    so students can learn about the artistic movement.
    """

    style_name: str = Field(
        description="Name of the detected art style (e.g. 'Impressionism', 'Cubism')"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Classifier confidence score between 0 and 1",
    )
    period: str = Field(
        description="Historical period or century when the style was prominent"
    )
    description: str = Field(
        description="Brief educational description of the art style (2-4 sentences)"
    )
    key_features: List[str] = Field(
        default_factory=list,
        description="List of 3-5 visual characteristics that define this style",
    )
    notable_artists: List[str] = Field(
        default_factory=list,
        description="List of 2-4 famous artists associated with this style",
    )


class ChatResponse(BaseModel):
    """The agent's structured reply to the user."""

    message: str = Field(description="The main conversational response text")
    art_classification: Optional[ArtStyleClassification] = Field(
        default=None,
        description="Art style classification if an image was analyzed",
    )
    sources: Optional[List[str]] = Field(
        default=None,
        description="URLs consulted during web search (if any)",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        default=0.85,
        description="Overall confidence in the response",
    )


class UserMessage(BaseModel):
    """Incoming message from the frontend."""

    text: str = Field(description="Text content of the user's message")
    image_base64: Optional[str] = Field(
        default=None,
        description="Base64-encoded image data (optional)",
    )
    session_id: str = Field(description="Unique session identifier for conversation memory")


class UserInfo(BaseModel):
    """Authenticated user information extracted from a Firebase token."""

    uid: str
    email: Optional[str] = None
    display_name: Optional[str] = None
