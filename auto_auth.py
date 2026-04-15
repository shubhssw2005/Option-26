"""
auto_auth.py — Nubra authentication using TOTP (no interactive prompts).
NUBRA_TOTP_SECRET must be set as env var on Render.
"""

import os
import time
import logging
import shelve

logger = logging.getLogger(__name__)


def generate_totp(secret: str) -> str:
    import pyotp

    return pyotp.TOTP(secret).now()


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
    secret = os.getenv("NUBRA_TOTP_SECRET", "")
    mpin = os.getenv("MPIN") or os.getenv("NUBRA_MPIN", "")

    if secret:
        logger.info("[auth] TOTP login (automatic)")

        def _auto_input(prompt=""):
            p = str(prompt).lower()
            if "totp" in p or "otp" in p:
                code = generate_totp(secret)
                logger.info(f"[auth] TOTP: {code}")
                return code
            elif "mpin" in p or "pin" in p:
                return mpin
            elif "phone" in p:
                return os.getenv("PHONE_NO") or os.getenv("NUBRA_PHONE", "")
            return ""

        import builtins

        orig = builtins.input
        builtins.input = _auto_input
        try:
            for attempt in range(3):
                try:
                    nubra = InitNubraSdk(env, totp_login=True, env_creds=True)
                    logger.info("[auth] TOTP login successful")

                    # Update class-level HEADERS so all HTTP sessions use new token
                    try:
                        with shelve.open("auth_data.db", flag="r") as db:
                            token = db.get("session_token", "")
                            device = db.get("x-device-id", "TS123")
                        InitNubraSdk.HEADERS["Authorization"] = f"Bearer {token}"
                        InitNubraSdk.HEADERS["x-device-id"] = device
                        InitNubraSdk.HEADERS["x-app-version"] = getattr(
                            InitNubraSdk, "VERSION", "1.0.0"
                        )
                        InitNubraSdk.HEADERS["x-device-os"] = "sdk"
                        InitNubraSdk.HEADERS["Cookie"] = f"deviceId={device}"
                        logger.info(f"[auth] HEADERS updated, device={device[:20]}")
                    except Exception as e:
                        logger.warning(f"[auth] HEADERS update: {e}")

                    return nubra
                except Exception as e:
                    logger.warning(f"[auth] TOTP attempt {attempt+1} failed: {e}")
                    if attempt < 2:
                        time.sleep(31)  # wait for next TOTP window
        finally:
            builtins.input = orig

        logger.error("[auth] All TOTP attempts failed")
        return None

    # Local dev: interactive OTP
    logger.info("[auth] Interactive OTP login")
    return InitNubraSdk(env, env_creds=True)
