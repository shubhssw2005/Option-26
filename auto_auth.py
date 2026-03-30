"""
auto_auth.py — Fully automatic Nubra authentication.

If NUBRA_TOTP_SECRET is set → TOTP login (no human needed, works on Render).
Otherwise → interactive OTP (local dev).
"""

import os
import time
import logging

logger = logging.getLogger(__name__)


def generate_totp(secret: str) -> str:
    import pyotp

    return pyotp.TOTP(secret).now()


def get_authenticated_client():
    from dotenv import load_dotenv

    load_dotenv()

    from nubra_python_sdk.start_sdk import InitNubraSdk, NubraEnv

    env_str = os.getenv("NUBRA_ENV", "uat").lower()
    env = NubraEnv.PROD if env_str == "production" else NubraEnv.UAT
    secret = os.getenv("NUBRA_TOTP_SECRET", "")

    if secret:
        logger.info("[auth] TOTP login — fully automatic")
        for attempt in range(3):
            try:
                nubra = InitNubraSdk(env, totp_login=True, env_creds=True)
                logger.info("[auth] TOTP login successful")
                return nubra
            except Exception as e:
                logger.warning(f"[auth] TOTP attempt {attempt+1} failed: {e}")
                time.sleep(31)  # wait for next 30s TOTP window
        raise RuntimeError("TOTP login failed after 3 attempts")
    else:
        logger.info("[auth] Interactive OTP login")
        return InitNubraSdk(env, env_creds=True)
