#!/bin/bash
set -uo pipefail

DATA_ROOT="data/iMED_NVS"
CONFIG="arguments/imed.py"
PORT=6017

failed=()

for seq_path in "$DATA_ROOT"/*/; do
    seq=$(basename "$seq_path")
    echo "=== Training $seq ==="
    PYTHONPATH='.' python train.py -s "$DATA_ROOT/$seq" --port "$PORT" --expname "imed/$seq" --configs "$CONFIG"
    if [ $? -ne 0 ]; then
        echo "!!! Training failed for $seq"
        failed+=("$seq")
    fi
done

echo ""
if [ ${#failed[@]} -eq 0 ]; then
    echo "All sequences trained successfully."
else
    echo "Failed sequences: ${failed[*]}"
    exit 1
fi
