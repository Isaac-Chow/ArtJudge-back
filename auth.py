"""Firebase Auth middleware for FastAPI.

Verifies incoming Bearer tokens using the Firebase Admin SDK and exposes
a `get_current_user` dependency that FastAPI endpoints can inject.

The service account JSON file is read from:
    backend/cloud/art-judge-firebase-adminsdk-fbsvc-*.json
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import firebase_admin
from firebase_admin import auth as fb_auth, credentials
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from models import UserInfo

# ---------------------------------------------------------------------------
# Firebase Admin initialisation (runs once at import time)
# ---------------------------------------------------------------------------

_firebase_initialized = False

# Resolve the service account JSON path relative to this file
_CLOUD_DIR = Path(__file__).resolve().parent / "cloud"


def _find_service_account_file() -> Optional[Path]:
    """Locate the Firebase service account JSON in the cloud/ directory."""
    if not _CLOUD_DIR.is_dir():
        return None
    # Match the naming pattern used by Firebase Console
    matches = list(_CLOUD_DIR.glob("*firebase-adminsdk*.json"))
    return matches[0] if matches else None


def _init_firebase() -> None:
    """Initialise the Firebase Admin SDK from the service account JSON file."""
    global _firebase_initialized
    if _firebase_initialized:
        return

    sa_file = _find_service_account_file()

    if sa_file is None:
        print(
            "[auth] WARNING: No Firebase service account JSON found in "
            f"{_CLOUD_DIR}. Auth-protected endpoints will reject all requests."
        )
        return

    try:
        # credentials.Certificate accepts a file path string directly
        cred = credentials.Certificate(str(sa_file))
        print(f"[auth] Loaded Firebase service account from {sa_file.name}")
    except Exception as exc:
        print(f"[auth] ERROR loading Firebase credentials: {exc}")
        return

    # Read project_id from the JSON for app initialisation
    with open(sa_file) as f:
        sa_data = json.load(f)
    project_id = sa_data.get("project_id")

    firebase_admin.initialize_app(cred, {"projectId": project_id})
    _firebase_initialized = True


_init_firebase()

# ---------------------------------------------------------------------------
# Security scheme
# ---------------------------------------------------------------------------

_bearer_scheme = HTTPBearer(auto_error=False)


async def get_current_user(
    creds: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme),
) -> UserInfo:
    """FastAPI dependency that verifies a Firebase ID token and returns user info.

    Usage in an endpoint::

        @app.post("/api/chat")
        async def chat(user: UserInfo = Depends(get_current_user)):
            ...
    """
    if creds is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication token. Please log in with Google.",
        )

    token = creds.credentials

    if not _firebase_initialized:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication service is not configured.",
        )

    try:
        decoded = fb_auth.verify_id_token(token)
    except fb_auth.ExpiredIdTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Your session has expired. Please log in again.",
        )
    except fb_auth.InvalidIdTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token.",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Authentication failed: {exc}",
        )

    return UserInfo(
        uid=decoded.get("uid", ""),
        email=decoded.get("email"),
        display_name=decoded.get("name"),
    )


async def optional_user(
    creds: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme),
) -> Optional[UserInfo]:
    """Optional auth dependency — returns None instead of 401 when no token."""
    if creds is None:
        return None
    try:
        return await get_current_user(creds)
    except HTTPException:
        return None
