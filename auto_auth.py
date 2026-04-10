"""
auto_auth.py — Nubra authentication for Render.
Uses NUBRA_SESSION_TOKEN env var. Writes it to shelve so SDK reads it normally.
"""

import os, time, logging, shelve

logger = logging.getLogger(__name__)


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
    token = os.getenv("NUBRA_SESSION_TOKEN", "")

    if token:
        logger.info("[auth] Using NUBRA_SESSION_TOKEN")

        # Write token to shelve so SDK reads it on init
        device = get_device_id()
        try:
            with shelve.open("auth_data.db", flag="c", writeback=True) as db:
                db["session_token"] = token
                db["auth_token"] = token
                if device and device != "TS123":
                    db["x-device-id"] = device
        except Exception as e:
            logger.warning(f"[auth] shelve write: {e}")

        # Update class-level HEADERS so all HTTP sessions use new token
        InitNubraSdk.HEADERS["Authorization"] = f"Bearer {token}"
        InitNubraSdk.HEADERS["x-device-id"] = device
        InitNubraSdk.HEADERS["x-app-version"] = getattr(
            InitNubraSdk, "VERSION", "1.0.0"
        )
        InitNubraSdk.HEADERS["x-device-os"] = "sdk"
        InitNubraSdk.HEADERS["Cookie"] = f"deviceId={device}"

        # Patch input() so SDK never prompts interactively
        import builtins

        orig_input = builtins.input
        builtins.input = lambda p="": ""
        try:
            # __new__ + manual attribute setup — avoids triggering login flow
            nubra = object.__new__(InitNubraSdk)
            nubra.env = env
            nubra.totp_login = False
            nubra.db_path = "auth_data.db"
            nubra.env_path_login = False
            nubra.token_data = {
                "session_token": token,
                "auth_token": token,
                "x-device-id": device,
            }
            nubra.API_BASE_URL = "https://api.nubra.io"
            nubra.BEARER_TOKEN = token
            nubra.VERSION = getattr(InitNubraSdk, "VERSION", "1.0.0")
            nubra.VERSION_URL = getattr(InitNubraSdk, "VERSION_URL", "")

            # auth_flow re-injects token instead of prompting
            def _auth_flow():
                InitNubraSdk.HEADERS["Authorization"] = f"Bearer {token}"
                logger.info("[auth] auth_flow: re-injected token")

            nubra.auth_flow = _auth_flow

            logger.info("[auth] Client ready")
            return nubra
        finally:
            builtins.input = orig_input

    # Local dev: interactive OTP
    logger.info("[auth] Interactive OTP login")
    return InitNubraSdk(env, env_creds=True)
