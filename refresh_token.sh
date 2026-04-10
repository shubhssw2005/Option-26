#!/bin/zsh
# refresh_token.sh — Auto-refresh Nubra session token on Render
# Cron: 0 8,20 * * 1-5 /Users/shubhamsw2005/stock-2026/refresh_token.sh

DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="/opt/anaconda3/bin/python3"
LOG="$DIR/logs/token_refresh.log"
mkdir -p "$DIR/logs"

echo "[$(TZ=Asia/Kolkata date '+%Y-%m-%d %H:%M IST')] Refreshing Nubra token..." | tee -a "$LOG"

"$PYTHON" - << 'PYEOF' 2>&1 | tee -a "$DIR/logs/token_refresh.log"
import os, sys, shelve, requests, json, base64, datetime
sys.path.insert(0, '/Users/shubhamsw2005/stock-2026')
os.chdir('/Users/shubhamsw2005/stock-2026')

RENDER_API_KEY    = 'rnd_KfFjKqMcZgBgLCpjpHj8yQuaKuAk'
RENDER_SERVICE_ID = 'srv-d755o3haae7s73brnik0'

def get_token():
    try:
        with shelve.open('auth_data.db', flag='r') as db:
            return db.get('session_token',''), db.get('x-device-id','TS123')
    except Exception:
        return '', 'TS123'

def is_valid(token, device):
    if not token: return False
    try:
        r = requests.get('https://api.nubra.io/userinfo',
            headers={'Authorization': f'Bearer {token}', 'x-device-id': device}, timeout=8)
        return r.status_code == 200
    except Exception:
        return False

def hours_left(token):
    try:
        payload = json.loads(base64.b64decode(token.split('.')[1]+'=='))
        exp = datetime.datetime.fromtimestamp(payload['exp'])
        return (exp - datetime.datetime.now()).total_seconds() / 3600
    except Exception:
        return 0

token, device = get_token()

if not is_valid(token, device):
    print('Token invalid — logging in fresh...')
    from nubra_python_sdk.start_sdk import InitNubraSdk, NubraEnv
    InitNubraSdk(NubraEnv.PROD, env_creds=True)
    token, device = get_token()

if not token:
    print('ERROR: Could not get token')
    sys.exit(1)

h = hours_left(token)
print(f'Token valid: {len(token)} chars, {h:.1f}h remaining')

headers = {
    'Authorization': f'Bearer {RENDER_API_KEY}',
    'Accept': 'application/json',
    'Content-Type': 'application/json',
}

# Push token to Render (correct format: list of key/value dicts)
r = requests.put(
    f'https://api.render.com/v1/services/{RENDER_SERVICE_ID}/env-vars',
    headers=headers,
    json=[
        {'key': 'NUBRA_SESSION_TOKEN', 'value': token},
        {'key': 'NUBRA_DEVICE_ID',     'value': device},
        {'key': 'NUBRA_DEVICE_ID',     'value': device},
    ],
    timeout=15
)
if r.status_code in (200, 201):
    print('✓ Token pushed to Render')
else:
    print(f'ERROR: {r.status_code} {r.text[:200]}')
    sys.exit(1)

# Trigger redeploy
r2 = requests.post(
    f'https://api.render.com/v1/services/{RENDER_SERVICE_ID}/deploys',
    headers=headers, json={}, timeout=15
)
d = r2.json()
print(f'✓ Redeploy: {r2.status_code} id={d.get("id","?")} status={d.get("status","?")}')
print('Done.')
PYEOF
