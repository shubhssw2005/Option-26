#!/bin/zsh
# refresh_token.sh — Auto-refresh Nubra session token on Render
#
# Run this daily (or set up as a cron job):
#   crontab -e
#   0 8 * * 1-5 /Users/shubhamsw2005/stock-2026/refresh_token.sh
#   (runs at 8am IST Mon-Fri, before market opens)
#
# Requires:
#   RENDER_API_KEY  — from Render dashboard → Account → API Keys
#   RENDER_SERVICE_ID — from Render dashboard → your service → Settings → Service ID

set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="/opt/anaconda3/bin/python3"
LOG="$DIR/logs/token_refresh.log"
mkdir -p "$DIR/logs"

echo "[$(TZ=Asia/Kolkata date '+%Y-%m-%d %H:%M IST')] Refreshing Nubra token..." | tee -a "$LOG"

# Load env
source "$DIR/.env" 2>/dev/null || true

# Get fresh token (will prompt for OTP if needed)
TOKEN=$("$PYTHON" -c "
import sys, os
sys.path.insert(0, '$DIR')
os.chdir('$DIR')
from auto_auth import get_current_token, _is_token_valid
token = get_current_token()
if token and _is_token_valid(token):
    print(token)
else:
    print('')
" 2>/dev/null)

if [ -z "$TOKEN" ]; then
    echo "[$(date)] Token invalid or missing — need manual OTP login" | tee -a "$LOG"
    echo "Run: cd $DIR && /opt/anaconda3/bin/python3 auto_auth.py token"
    exit 1
fi

echo "[$(date)] Token valid (${#TOKEN} chars)" | tee -a "$LOG"

# Push to Render via API
if [ -n "$RENDER_API_KEY" ] && [ -n "$RENDER_SERVICE_ID" ]; then
    echo "[$(date)] Pushing to Render..." | tee -a "$LOG"
    
    # Get current env vars
    ENVVARS=$(curl -s \
        -H "Authorization: Bearer $RENDER_API_KEY" \
        -H "Accept: application/json" \
        "https://api.render.com/v1/services/$RENDER_SERVICE_ID/env-vars")
    
    # Update NUBRA_SESSION_TOKEN using Python (easier JSON manipulation)
    "$PYTHON" -c "
import json, sys, os, requests

api_key    = '$RENDER_API_KEY'
service_id = '$RENDER_SERVICE_ID'
new_token  = '''$TOKEN'''

headers = {
    'Authorization': f'Bearer {api_key}',
    'Accept': 'application/json',
    'Content-Type': 'application/json',
}

# Get current env vars
r = requests.get(f'https://api.render.com/v1/services/{service_id}/env-vars', headers=headers)
env_vars = r.json()

# Update token
updated = False
for ev in env_vars:
    if ev.get('envVar', {}).get('key') == 'NUBRA_SESSION_TOKEN':
        ev['envVar']['value'] = new_token
        updated = True
        break
if not updated:
    env_vars.append({'envVar': {'key': 'NUBRA_SESSION_TOKEN', 'value': new_token}})

# Push
r2 = requests.put(
    f'https://api.render.com/v1/services/{service_id}/env-vars',
    headers=headers, json=env_vars
)
if r2.status_code in (200, 201):
    print('✓ Token pushed to Render')
else:
    print(f'✗ Failed: {r2.status_code} {r2.text[:100]}')
    sys.exit(1)
" 2>&1 | tee -a "$LOG"

    # Trigger redeploy
    echo "[$(date)] Triggering Render redeploy..." | tee -a "$LOG"
    curl -s -X POST \
        -H "Authorization: Bearer $RENDER_API_KEY" \
        -H "Accept: application/json" \
        "https://api.render.com/v1/services/$RENDER_SERVICE_ID/deploys" \
        -d '{"clearCache": false}' | "$PYTHON" -c "
import json,sys
d=json.load(sys.stdin)
print('Deploy ID:', d.get('id','?'), '| Status:', d.get('status','?'))
" 2>&1 | tee -a "$LOG"

else
    echo "[$(date)] No RENDER_API_KEY/SERVICE_ID — skipping Render push" | tee -a "$LOG"
    echo "Set these in .env:"
    echo "  RENDER_API_KEY=rnd_xxxx"
    echo "  RENDER_SERVICE_ID=srv-xxxx"
fi

echo "[$(date)] Done." | tee -a "$LOG"
