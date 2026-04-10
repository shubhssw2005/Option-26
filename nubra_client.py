"""
nubra_client.py — Direct HTTP client for Nubra API.
No SDK dependency — uses requests directly with token + device ID.
This avoids all SDK auth/device-ID issues on Render.
"""

import os
import logging
import requests
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)

BASE_URL = "https://api.nubra.io"


class NubraDirectClient:
    """Direct HTTP client using session token + device ID."""

    def __init__(self, token: str, device_id: str):
        self.token = token
        self.device_id = device_id
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {token}",
                "x-device-id": device_id,
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
        )

    def is_valid(self) -> bool:
        try:
            r = self.session.get(f"{BASE_URL}/userinfo", timeout=8)
            return r.status_code == 200
        except Exception:
            return False

    def option_chain(
        self, instrument: str, exchange: str = "NSE", expiry: str = ""
    ) -> dict:
        params = {"exchange": exchange}
        if expiry:
            params["expiry"] = expiry
        r = self.session.get(
            f"{BASE_URL}/optionchains/{instrument}", params=params, timeout=15
        )
        r.raise_for_status()
        return r.json()

    def historical_data(self, payload: dict) -> dict:
        r = self.session.post(
            f"{BASE_URL}/charts/timeseries",
            json={"query": [payload] if isinstance(payload, dict) else payload},
            timeout=30,
        )
        r.raise_for_status()
        return r.json()

    def get_historical_closes(
        self, symbol: str, exchange: str = "NSE", interval: str = "1d", days: int = 90
    ) -> list:
        """Returns list of {timestamp, value} dicts."""
        end_dt = datetime.now(timezone.utc)
        start_dt = end_dt - timedelta(days=days)
        data = self.historical_data(
            {
                "exchange": exchange,
                "type": "INDEX",
                "values": [symbol],
                "fields": ["close"],
                "startDate": start_dt.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                "endDate": end_dt.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                "interval": interval,
                "intraDay": False,
                "realTime": False,
            }
        )
        try:
            sym_data = data["result"][0]["values"][0][symbol]
            return sym_data.get("close", [])
        except (KeyError, IndexError):
            return []


def get_client() -> NubraDirectClient | None:
    """Get authenticated direct client from env vars."""
    token = os.getenv("NUBRA_SESSION_TOKEN", "")
    device = os.getenv("NUBRA_DEVICE_ID", "")
    if not token or not device:
        logger.warning("[client] NUBRA_SESSION_TOKEN or NUBRA_DEVICE_ID not set")
        return None
    client = NubraDirectClient(token, device)
    if client.is_valid():
        logger.info("[client] Direct client authenticated ✓")
        return client
    else:
        logger.error(
            "[client] Token invalid — check NUBRA_SESSION_TOKEN and NUBRA_DEVICE_ID"
        )
        return None
