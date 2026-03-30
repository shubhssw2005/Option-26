"""
auth_manager.py — Handles Nubra authentication for deployment.

On Render (no terminal): uses TOTP if configured, else reads token from env.
Locally: interactive OTP prompt as usual.
"""

import os
import json
import time
import logging
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

TOKEN_FILE = os.getenv("TOKEN_FILE", ".nubra_token.json")


def save_token(session_token: str, user_id: int = None):
    """Persist token to file for reuse across restarts."""
    data = {
        "session_token": session_token,
        "user_id": user_id,
        "saved_at": time.time(),
    }
    with open(TOKEN_FILE, "w") as f:
        json.dump(data, f)
    logger.info(f"Token saved to {TOKEN_FILE}")


def load_token() -> str | None:
    """Load cached token if it exists and is recent (< 8 hours)."""
    if not os.path.exists(TOKEN_FILE):
        return None
    try:
        with open(TOKEN_FILE) as f:
            data = json.load(f)
        age_hours = (time.time() - data.get("saved_at", 0)) / 3600
        if age_hours < 8:
            logger.info(f"Using cached token (age: {age_hours:.1f}h)")
            return data["session_token"]
        logger.info("Cached token expired (>8h)")
        return None
    except Exception:
        return None


def get_nubra_client():
    """
    Returns authenticated InitNubraSdk instance.
    Strategy:
      1. Try cached token file
      2. Try TOTP login (if NUBRA_TOTP_SECRET set)
      3. Fall back to interactive OTP (local dev only)
    """
    from nubra_python_sdk.start_sdk import InitNubraSdk, NubraEnv

    env_str = os.getenv("NUBRA_ENV", "uat").lower()
    env = NubraEnv.PROD if env_str == "production" else NubraEnv.UAT

    # Check if running on Render (non-interactive)
    is_render = os.getenv("RENDER", "") == "true"

    if is_render:
        # On Render: must use TOTP or pre-set token
        totp_secret = os.getenv("NUBRA_TOTP_SECRET", "")
        if totp_secret:
            logger.info("Using TOTP login for Render deployment")
            nubra = InitNubraSdk(env, totp_login=True, env_creds=True)
        else:
            # Try cached token — if expired, server starts without live data
            cached = load_token()
            if cached:
                logger.info("Using cached session token")
                nubra = InitNubraSdk.__new__(InitNubraSdk)
                # Inject token directly
                nubra._InitNubraSdk__set_token(cached)
            else:
                logger.warning(
                    "No TOTP secret and no cached token — starting without auth"
                )
                return None
    else:
        # Local: normal interactive flow
        nubra = InitNubraSdk(env, env_creds=True)

    return nubra
