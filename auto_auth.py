"""
auto_auth.py — Nubra authentication for Render deployment.

On Render: uses NUBRA_SESSION_TOKEN (pre-generated locally, set as env var).
Locally: interactive OTP as usual.

To generate a fresh token locally:
    python auto_auth.py token
"""

import os
import time
import logging
import shelve

logger = logging.getLogger(__name__)


def get_current_token() -> str:
    """Get session token from local shelve cache."""
    try:
        with shelve.open("auth_data.db", flag="r") as db:
            return db.get("session_token", "")
    except Exception:
        return ""


def get_device_id() -> str:
    try:
        with shelve.open("auth_data.db", flag="r") as db:
            return db.get("x-device-id", "TS123")
    except Exception:
        return "TS123"


def get_authenticated_client():
    from dotenv import load_dotenv

    load_dotenv()

    from nubra_python_sdk.start_sdk import InitNubraSdk, NubraEnv

    env_str = os.getenv("NUBRA_ENV", "uat").lower()
    env = NubraEnv.PROD if env_str == "production" else NubraEnv.UAT

    # On Render: use pre-set session token
    pretoken = os.getenv("NUBRA_SESSION_TOKEN", "")
    if pretoken:
        logger.info("[auth] Using NUBRA_SESSION_TOKEN from env")
        return _client_from_token(env, pretoken)

    # Local: normal interactive login
    logger.info("[auth] Interactive OTP login")
    nubra = InitNubraSdk(env, env_creds=True)
    return nubra


def _client_from_token(env, token: str):
    """
    Create an InitNubraSdk-like object with a pre-existing session token.
    Bypasses all login flows.
    """
    from nubra_python_sdk.start_sdk import InitNubraSdk, NubraEnv

    # Create instance without calling __init__
    nubra = object.__new__(InitNubraSdk)

    # Set required attributes that MarketData reads
    nubra.env = env
    nubra.totp_login = False
    nubra.db_path = "auth_data.db"
    nubra.env_path_login = False
    nubra.token_data = {
        "session_token": token,
        "auth_token": token,
        "x-device-id": get_device_id(),
    }
    nubra.API_BASE_URL = "https://api.nubra.io"

    # Minimal methods needed by MarketData
    def _get_token():
        return token

    def _auth_flow():
        logger.info("[auth] auth_flow called — token already set")

    nubra.get_token = _get_token
    nubra.auth_flow = _auth_flow
    nubra.BEARER_TOKEN = token

    logger.info("[auth] Client created from pre-set token")
    return nubra


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "token":
        # Print current token for use as Render env var
        token = get_current_token()
        if token:
            print(f"\nAdd this to Render environment variables:")
            print(f"NUBRA_SESSION_TOKEN = {token}")
            print(f"\nToken expires in ~8 hours. Re-run to refresh.")
        else:
            print("No token found. Run the server locally first to authenticate.")
    else:
        print("Usage: python auto_auth.py token")
