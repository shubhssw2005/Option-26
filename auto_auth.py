"""
auto_auth.py — Fully automatic Nubra authentication.

If NUBRA_TOTP_SECRET is set → generates TOTP and monkey-patches input()
so the SDK never blocks waiting for user input.
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
    mpin = os.getenv("MPIN") or os.getenv("NUBRA_MPIN", "")

    if secret:
        logger.info("[auth] TOTP auto-login — patching input()")

        # Pre-generate answers for all interactive prompts the SDK may ask
        # Order: TOTP code, then MPIN (if asked again)
        answers = []

        def _auto_input(prompt=""):
            """Replace input() with automatic responses."""
            prompt_lower = str(prompt).lower()
            logger.info(f"[auth] SDK prompt: {prompt!r}")

            if "totp" in prompt_lower or "otp" in prompt_lower:
                code = generate_totp(secret)
                logger.info(f"[auth] Auto-answering TOTP: {code}")
                return code
            elif "mpin" in prompt_lower or "pin" in prompt_lower:
                logger.info("[auth] Auto-answering MPIN")
                return mpin
            elif "phone" in prompt_lower:
                phone = os.getenv("PHONE_NO") or os.getenv("NUBRA_PHONE", "")
                logger.info(f"[auth] Auto-answering phone: {phone}")
                return phone
            else:
                logger.warning(f"[auth] Unknown prompt: {prompt!r} — returning empty")
                return ""

        # Patch builtins.input globally
        import builtins

        original_input = builtins.input
        builtins.input = _auto_input

        try:
            for attempt in range(3):
                try:
                    # Regenerate TOTP fresh for each attempt
                    nubra = InitNubraSdk(env, totp_login=True, env_creds=True)
                    logger.info("[auth] TOTP login successful")
                    return nubra
                except Exception as e:
                    logger.warning(f"[auth] Attempt {attempt+1} failed: {e}")
                    if attempt < 2:
                        logger.info("[auth] Waiting 31s for next TOTP window...")
                        time.sleep(31)
            raise RuntimeError("TOTP login failed after 3 attempts")
        finally:
            builtins.input = original_input  # always restore

    else:
        logger.info("[auth] No TOTP secret — interactive OTP login")
        return InitNubraSdk(env, env_creds=True)
