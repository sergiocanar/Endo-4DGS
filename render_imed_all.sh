#!/bin/bash
set -uo pipefail

DATA_ROOT="data/iMED_NVS"
OUTPUT_ROOT="output/imed"
CONFIG="arguments/imed.py"

failed=()
skipped=()

for seq_path in "$DATA_ROOT"/*/; do
    seq=$(basename "$seq_path")
    model_path="$OUTPUT_ROOT/$seq"

    if [ ! -f "$model_path/cfg_args" ]; then
        echo "!!! Skipping $seq (no trained model found at $model_path)"
        skipped+=("$seq")
        continue
    fi

    echo "=== Rendering $seq ==="
    python render.py --model_path "$model_path" --pc --skip_video --skip_train \
        --configs "$CONFIG" --white_background
    if [ $? -ne 0 ]; then
        echo "!!! Rendering failed for $seq"
        failed+=("$seq")
        continue
    fi

    echo "=== Evaluating $seq ==="
    python metrics.py --model_paths "$model_path"
    if [ $? -ne 0 ]; then
        echo "!!! Metrics failed for $seq"
        failed+=("$seq")
    fi
done

echo ""
if [ ${#skipped[@]} -ne 0 ]; then
    echo "Skipped (no trained model): ${skipped[*]}"
fi
if [ ${#failed[@]} -eq 0 ]; then
    echo "All available sequences rendered/evaluated successfully."
else
    echo "Failed sequences: ${failed[*]}"
    exit 1
fi
