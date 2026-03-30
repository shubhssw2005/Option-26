#!/bin/bash
# build_render.sh — Render build script
set -e

echo "Installing dependencies..."
pip install -r requirements_deploy.txt

echo "Installing Nubra SDK..."
pip install --index-url https://test.pypi.org/simple/ \
            --extra-index-url https://pypi.org/simple \
            nubra-sdk

echo "Copying pre-trained models to persistent disk..."
mkdir -p /data/trained_models
if [ -d "trained_models" ]; then
    cp trained_models/*.pkl /data/trained_models/ 2>/dev/null || true
    echo "Models copied: $(ls /data/trained_models/*.pkl 2>/dev/null | wc -l) files"
fi

echo "Initialising database..."
python3 -c "
from collect_data import init_db
conn = init_db('/data/data.db')
conn.close()
print('DB initialised at /data/data.db')
"

echo "Build complete."
