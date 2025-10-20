#!/usr/bin/env bash
set -o errexit

echo "--- Installing system dependencies (fontconfig) ---"
apt-get update && apt-get install -y fontconfig fonts-wqy-zenhei

echo "--- Installing Python dependencies ---"
pip install -r requirements.txt

echo "--- Force-clearing Matplotlib font cache ---"
rm -rf $(python -c "import matplotlib; print(matplotlib.get_cachedir())")

echo "--- Build process finished ---"