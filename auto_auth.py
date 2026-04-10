"""
auto_auth.py — Nubra authentication with automatic token refresh.

Strategy:
  1. Try NUBRA_SESSION_TOKEN env var (fast path)
  2. If expired (403), re-authenticate via OTP using Nubra REST API directly
     (no interactive prompt — uses phone/mpin from env vars)
  3. Cache new token in memory + update Render env via Render API (if key set)

Token refresh flow (no TOTP needed):
  POST /sendphoneotp  → get temp_token
  POST /sendphoneotp  → trigger OTP send
  POST /verifyphoneotp → verify OTP  ← PROBLEM: needs OTP from SMS
  POST /verifypin     → get session_token

The only way to avoid SMS OTP on Render is to use the Nubra REST API
with a long-lived token approach. Since Nubra tokens last ~8h, we refresh
by storing the token in Render's env vars via the Render API.
"""

import os
import time
import logging
import shelve
import threading

logger = logging.getLogger(__name__)

# In-memory token cache
_token_cache = {"session_token": "", "expires_at": 0, "device_id": "TS123"}
_refresh_lock = threading.Lock()


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


def _is_token_valid(token: str) -> bool:
    """Quick check if token is still valid."""
    if not token:
        return False
    import requests

    try:
        r = requests.get(
            "https://api.nubra.io/userinfo",
            headers={
                "Authorization": f"Bearer {token}",
                "x-device-id": get_device_id(),
            },
            timeout=8,
        )
        return r.status_code == 200
    except Exception:
        return False


def _push_token_to_render(token: str):
    """Update NUBRA_SESSION_TOKEN on Render via Render API."""
    render_api_key = os.getenv("RENDER_API_KEY", "")
    service_id = os.getenv("RENDER_SERVICE_ID", "")
    if not render_api_key or not service_id:
        logger.info("[auth] No RENDER_API_KEY/SERVICE_ID — skipping Render env update")
        return
    import requests

    try:
        # Get current env vars
        r = requests.get(
            f"https://api.render.com/v1/services/{service_id}/env-vars",
            headers={
                "Authorization": f"Bearer {render_api_key}",
                "Accept": "application/json",
            },
            timeout=10,
        )
        if r.status_code != 200:
            logger.warning(f"[auth] Render API get env failed: {r.status_code}")
            return

        env_vars = r.json()
        # Update or add NUBRA_SESSION_TOKEN
        updated = False
        for ev in env_vars:
            if ev.get("envVar", {}).get("key") == "NUBRA_SESSION_TOKEN":
                ev["envVar"]["value"] = token
                updated = True
                break
        if not updated:
            env_vars.append({"envVar": {"key": "NUBRA_SESSION_TOKEN", "value": token}})

        # PUT updated env vars
        r2 = requests.put(
            f"https://api.render.com/v1/services/{service_id}/env-vars",
            headers={
                "Authorization": f"Bearer {render_api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            json=env_vars,
            timeout=10,
        )
        if r2.status_code in (200, 201):
            logger.info("[auth] ✓ Token pushed to Render env vars")
        else:
            logger.warning(
                f"[auth] Render env update failed: {r2.status_code} {r2.text[:100]}"
            )
    except Exception as e:
        logger.warning(f"[auth] Render API error: {e}")


def _client_from_token(env, token: str):
    """
    Inject token into SDK by writing to shelve + updating class HEADERS.
    Then do a real InitNubraSdk init which reads from shelve.
    """
    from nubra_python_sdk.start_sdk import InitNubraSdk

    device = get_device_id()

    # 1. Write to shelve so SDK __init__ reads it
    try:
        with shelve.open("auth_data.db", flag="c", writeback=True) as db:
            db["session_token"] = token
            db["auth_token"]    = token
            if device and device != "TS123":
                db["x-device-id"] = device
    except Exception as e:
        logger.warning(f"[auth] Shelve write: {e}")

    # 2. Update class-level HEADERS immediately
    InitNubraSdk.HEADERS["Authorization"] = f"Bearer {token}"
    InitNubraSdk.HEADERS["x-device-id"]   = device
    InitNubraSdk.HEADERS["x-app-version"] = "1.0.0"
    InitNubraSdk.HEADERS["x-device-os"]   = "sdk"
    InitNubraSdk.HEADERS["Cookie"]        = f"deviceId={device}"

    # 3. Patch input() to prevent any interactive prompts
    import builtins
    orig = builtins.input
    builtins.input = lambda p="": ""
    try:
        # Use __new__ + minimal __init__ state to avoid triggering login
        nubra = object.__new__(InitNubraSdk)
        nubra.env            = env
        nubra.totp_login     = False
        nubra.db_path        = "auth_data.db"
        nubra.env_path_login = False
        nubra.token_data     = {"session_token": token, "auth_token": token, "x-device-id": device}
        nubra.API_BASE_URL   = "https://api.nubra.io"
        nubra.BEARER_TOKEN   = token
        nubra.VERSION        = getattr(InitNubraSdk, "VERSION", "1.0.0")
        nubra.VERSION_URL    = getattr(InitNubraSdk, "VERSION_URL", "")

        # Patch auth_flow to re-inject token instead of prompting
        def _auth_flow():
            InitNubraSdk.HEADERS["Authorization"] = f"Bearer {token}"
            logger.info("[auth] auth_flow re-injected token")
        nubra.auth_flow = _auth_flow

        logger.info(f"[auth] Client ready, HEADERS auth set: {InitNubraSdk.HEADERS.get('Authorization','')[:30]}...")
        return nubra
    finally:
        builtins.input = orig


def refresh_token_background(state: dict):
    """
    Background thread: checks token validity every 30 min.
    If expired, triggers re-auth via interactive OTP (only works locally).
    On Render: pushes new token via Render API after local refresh.
    """
    while True:
        time.sleep(1800)  # check every 30 min
        try:
            token = _token_cache.get("session_token", "")
            if token and not _is_token_valid(token):
                logger.warning("[auth] Token expired — attempting refresh")
                # On Render we can't do interactive OTP
                # Best we can do: log the warning and keep serving with stale token
                # until operator refreshes manually
                logger.warning("[auth] ⚠ Token expired on Render. Run locally:")
                logger.warning("[auth]   python auto_auth.py token")
                logger.warning(
                    "[auth]   Then update NUBRA_SESSION_TOKEN in Render dashboard"
                )
                state["authenticated"] = False
        except Exception as e:
            logger.error(f"[auth] Token check error: {e}")


def get_authenticated_client(state: dict = None):
    from dotenv import load_dotenv

    load_dotenv()

    from nubra_python_sdk.start_sdk import InitNubraSdk, NubraEnv

    env_str = os.getenv("NUBRA_ENV", "uat").lower()
    env = NubraEnv.PROD if env_str == "production" else NubraEnv.UAT
    pretoken = os.getenv("NUBRA_SESSION_TOKEN", "")

    # ── Pre-set token (Render) ────────────────────────────────────────────────
    if pretoken:
        logger.info("[auth] Using NUBRA_SESSION_TOKEN")
        if _is_token_valid(pretoken):
            _token_cache["session_token"] = pretoken
            nubra = _client_from_token(env, pretoken)
            logger.info("[auth] ✓ Token valid")
            return nubra
        else:
            logger.warning("[auth] NUBRA_SESSION_TOKEN is expired")

    # ── Local shelve token ────────────────────────────────────────────────────
    local_token = get_current_token()
    if local_token and _is_token_valid(local_token):
        logger.info("[auth] Using local shelve token")
        _token_cache["session_token"] = local_token
        return _client_from_token(env, local_token)

    # ── Interactive OTP (local dev only) ──────────────────────────────────────
    logger.info("[auth] Interactive OTP login")
    nubra = InitNubraSdk(env, env_creds=True)
    # Cache the new token
    try:
        new_token = get_current_token()
        if new_token:
            _token_cache["session_token"] = new_token
            _push_token_to_render(new_token)
    except Exception:

        pass
    return nubra


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "token":
        token = get_current_token()
        if not token:
            print("No token. Logging in...")
            from nubra_python_sdk.start_sdk import InitNubraSdk, NubraEnv
            from dotenv import load_dotenv

            load_dotenv()
            env_str = os.getenv("NUBRA_ENV", "uat").lower()
            env = NubraEnv.PROD if env_str == "production" else NubraEnv.UAT
            InitNubraSdk(env, env_creds=True)
            token = get_current_token()

        if token:
            print(f"\nNUBRA_SESSION_TOKEN={token}")
            print(f"\nToken length: {len(token)}")
            valid = _is_token_valid(token)
            print(f"Token valid: {valid}")
            if valid:
                _push_token_to_render(token)
        else:
            print("Failed to get token")
    else:
        print("Usage: python auto_auth.py token")
        print(
            "  Gets current session token and pushes to Render if RENDER_API_KEY is set"
        )
