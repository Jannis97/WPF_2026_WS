#!/usr/bin/env bash
# run.sh – Führt die komplette NIR-Hesperidin-Pipeline aus
set -e

cd "$(dirname "$0")"

echo "=== NIR Hesperidin Pipeline ==="
echo ""

# Pipeline ausführen
.venv/bin/python main.py

echo ""
echo "=== Fertig! Ergebnisse in runs/ ==="
