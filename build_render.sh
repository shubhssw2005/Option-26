#!/bin/bash
# build_render.sh — Render FREE tier build (no persistent disk)
set -e

echo "=== Installing dependencies ==="
pip install -r requirements_deploy.txt

echo "=== Installing Nubra SDK ==="
pip install --only-binary=:all: grpcio grpcio-tools 2>/dev/null || \
pip install grpcio grpcio-tools
pip install --index-url https://test.pypi.org/simple/ \
            --extra-index-url https://pypi.org/simple \
            nubra-sdk

echo "=== Setting up directories ==="
mkdir -p /tmp/trained_models
mkdir -p /tmp/logs

echo "=== Copying pre-trained models ==="
# Models are committed to the repo — just copy them
if [ -d "trained_models" ] && ls trained_models/*.pkl 1>/dev/null 2>&1; then
    cp trained_models/*.pkl /tmp/trained_models/
    echo "  ✓ Copied $(ls /tmp/trained_models/*.pkl | wc -l | tr -d ' ') models from repo"
else
    echo "  ⚠ No models in repo — will need to train after data collection"
fi

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
echo "Models: $(ls /tmp/trained_models/*.pkl 2>/dev/null | wc -l | tr -d ' ')"
ls -lh /tmp/trained_models/ 2>/dev/null || true
