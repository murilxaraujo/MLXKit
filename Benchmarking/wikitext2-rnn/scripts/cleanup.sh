#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"

remove_data=false
while (($#)); do
    case "$1" in
        --data) remove_data=true; shift ;;
        --help|-h)
            cat <<'EOF'
Usage: cleanup.sh [--data]

Removes this workload's results and Swift build directory.
--data also removes the downloaded WikiText-2 corpus and prepared token files.
The Conda environment and downloaded NLTK tokenizer resources are not removed.
EOF
            exit 0
            ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done

rm -rf -- "$BENCHMARK_DIR/results" "$BENCHMARK_DIR/.build"
if [[ "$remove_data" == true ]]; then rm -rf -- "$BENCHMARK_DIR/data"; fi
echo "Removed WikiText-2 RNN results and Swift build files."
if [[ "$remove_data" == true ]]; then echo "Removed the downloaded corpus and prepared token files."; fi
echo "Conda environment and NLTK resources were left untouched."
