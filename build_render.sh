#!/bin/bash
# build_render.sh — Render FREE tier build (no persistent disk)
set -e

echo "=== Installing dependencies ==="
pip install -r requirements_deploy.txt

echo "=== Installing Nubra SDK ==="
pip install --index-url https://test.pypi.org/simple/ \
            --extra-index-url https://pypi.org/simple \
            nubra-sdk

echo "=== Setting up directories ==="
mkdir -p /tmp/trained_models
mkdir -p /tmp/logs

echo "=== Downloading pre-trained models from GitHub ==="
REPO="${GITHUB_REPO:-shubhamsw2005/options-intelligence}"
BASE="https://github.com/${REPO}/releases/latest/download"

for model in nifty banknifty finnifty midcpnifty; do
    FILE="${model}_ensemble.pkl"
    echo "  Downloading ${FILE}..."
    if curl -fsSL "${BASE}/${FILE}" -o "/tmp/trained_models/${FILE}" 2>/dev/null; then
        SIZE=$(du -h "/tmp/trained_models/${FILE}" | cut -f1)
        echo "  ✓ ${FILE} (${SIZE})"
    else
        echo "  ⚠ ${FILE} not found in releases — will train on startup if data available"
        # Copy from repo if present (first deploy)
        if [ -f "trained_models/${FILE}" ]; then
            cp "trained_models/${FILE}" "/tmp/trained_models/${FILE}"
            echo "  ✓ Copied from repo"
        fi
    fi
done

echo "=== Initialising database ==="
python3 -c "
import os
os.environ['DB_PATH'] = '/tmp/data.db'
os.environ['MODELS_DIR'] = '/tmp/trained_models'
from collect_data import init_db
conn = init_db('/tmp/data.db')
conn.close()
print('DB initialised at /tmp/data.db')
"

echo "=== Build complete ==="
ls -lh /tmp/trained_models/ 2>/dev/null || echo "No models yet"
