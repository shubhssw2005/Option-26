"""
token_refresh.py — Automatically refreshes Nubra session token using TOTP.
No SMS needed. Runs as a background thread inside render_server.py.

Flow: TOTP login -> auth_token -> verifypin (MPIN) -> session_token
Then pushes new token to Render env vars so next restart uses it.
"""

import os
import time
import logging
import requests

logger = logging.getLogger(__name__)

BASE = "https://api.nubra.io"
RENDER_API = "https://api.render.com/v1"


def _get_new_session_token() -> str:
    """Login via TOTP + MPIN, return fresh session_token."""
    import pyotp

    totp_secret = os.getenv("NUBRA_TOTP_SECRET", "QYXKT4VJ65BSWPK6RG6ICEBMXEHJL5HP")
    phone = os.getenv("PHONE_NO", "9372056598")
    mpin = os.getenv("MPIN", "1211")
    device_id = os.getenv("NUBRA_DEVICE_ID", "e89633c4-388f-11f1-a43f-6e19e8448b32-sdk-0-4-0")

    totp_code = pyotp.TOTP(totp_secret).now()

    # Step 1: TOTP login -> auth_token
    r = requests.post(
        f"{BASE}/totp/login",
        json={"phone": phone, "totp": int(totp_code), "otp": ""},
        headers={"x-device-id": device_id, "Content-Type": "application/json"},
        timeout=15,
    )
    if r.status_code != 201:
        raise ValueError(f"totp/login failed: {r.status_code} {r.text[:200]}")
    auth_token = r.json().get("auth_token", "")
    if not auth_token:
        raise ValueError("No auth_token in totp/login response")

    # Step 2: MPIN -> session_token
    r2 = requests.post(
        f"{BASE}/verifypin",
        json={"pin": mpin},
        headers={
            "Authorization": f"Bearer {auth_token}",
            "x-device-id": device_id,
            "x-app-version": "0.4.0",
            "x-device-os": "sdk",
            "Cookie": f"deviceId={device_id}",
            "Content-Type": "application/json",
        },
        timeout=15,
    )
    if r2.status_code != 200:
        raise ValueError(f"verifypin failed: {r2.status_code} {r2.text[:200]}")
    session_token = r2.json().get("session_token", "")
    if not session_token:
        raise ValueError("No session_token in verifypin response")

    return session_token


def _push_token_to_render(session_token: str):
    """Update NUBRA_SESSION_TOKEN on Render so next restart uses new token."""
    render_key = os.getenv("RENDER_API_KEY", "")
    service_id = os.getenv("RENDER_SERVICE_ID", "")
    if not render_key or not service_id:
        return
    device_id = os.getenv("NUBRA_DEVICE_ID", "e89633c4-388f-11f1-a43f-6e19e8448b32-sdk-0-4-0")
    try:
        requests.put(
            f"{RENDER_API}/services/{service_id}/env-vars",
            headers={"Authorization": f"Bearer {render_key}", "Content-Type": "application/json"},
            json=[
                {"key": "NUBRA_SESSION_TOKEN", "value": session_token},
                {"key": "NUBRA_DEVICE_ID", "value": device_id},
            ],
            timeout=10,
        )
        logger.info("[token_refresh] Pushed new token to Render env vars")
    except Exception as e:
        logger.warning(f"[token_refresh] Failed to push to Render: {e}")


def refresh_token_and_update_server(state: dict):
    """Get new token, update in-memory state and Render env vars."""
    try:
        logger.info("[token_refresh] Refreshing session token via TOTP...")
        new_token = _get_new_session_token()
        device_id = os.getenv("NUBRA_DEVICE_ID", "e89633c4-388f-11f1-a43f-6e19e8448b32-sdk-0-4-0")

        # Update the live client in-memory
        client = state.get("client")
        if client:
            client.token = new_token
            client.session.headers.update({"Authorization": f"Bearer {new_token}"})

        # Update env var for next restart
        os.environ["NUBRA_SESSION_TOKEN"] = new_token
        _push_token_to_render(new_token)

        logger.info("[token_refresh] Token refreshed successfully ✓")
        return new_token
    except Exception as e:
        logger.error(f"[token_refresh] Failed: {e}")
        return None


def start_token_refresh_loop(state: dict, interval_hours: float = 10.0):
    """Background thread: refresh token every interval_hours."""
    import threading

    def _loop():
        # First refresh after 10h
        time.sleep(interval_hours * 3600)
        while True:
            refresh_token_and_update_server(state)
            time.sleep(interval_hours * 3600)

    t = threading.Thread(target=_loop, daemon=True)
    t.start()
    logger.info(f"[token_refresh] Auto-refresh loop started (every {interval_hours}h)")
