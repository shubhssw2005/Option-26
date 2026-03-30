"""
auto_otp.py — Nubra 4-step auth flow.
Returns a session_token ready for REST + WebSocket use.
"""

import os, requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.getenv("NUBRA_BASE_URL", "https://uatapi.nubra.io")
PHONE = os.getenv("NUBRA_PHONE", "9372056598")
MPIN = os.getenv("NUBRA_MPIN", "1211")
DEVICE_ID = "TS123"


def _post(endpoint: str, payload: dict, headers: dict = None) -> dict:
    h = {"Content-Type": "application/json", **(headers or {})}
    r = requests.post(f"{BASE_URL}{endpoint}", json=payload, headers=h, timeout=15)
    r.raise_for_status()
    return r.json()


def login(otp: str = None) -> str:
    """Full login flow. If otp is None, prompts on stdin."""
    # Step 1
    r1 = _post("/sendphoneotp", {"phone": PHONE, "skip_totp": False})
    temp1 = r1["temp_token"]

    # Step 2
    r2 = _post(
        "/sendphoneotp",
        {"phone": PHONE, "skip_totp": True},
        headers={"x-temp-token": temp1},
    )
    temp2 = r2["temp_token"]

    # Step 3
    if otp is None:
        otp = input("Enter OTP: ").strip()
    r3 = _post(
        "/verifyphoneotp",
        {"phone": PHONE, "otp": otp},
        headers={"x-temp-token": temp2, "x-device-id": DEVICE_ID},
    )
    auth_token = r3["auth_token"]

    # Step 4
    r4 = _post(
        "/verifypin",
        {"pin": MPIN},
        headers={"Authorization": f"Bearer {auth_token}", "x-device-id": DEVICE_ID},
    )
    session_token = r4["session_token"]
    print(f"[auth] Login successful. userId={r4.get('userId')}")
    return session_token


if __name__ == "__main__":
    token = login()
    print(f"SESSION_TOKEN={token}")
