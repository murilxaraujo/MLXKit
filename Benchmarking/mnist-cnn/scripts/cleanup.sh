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

Removes only this workload's results and Swift build directory.
--data also removes the shared MNIST cache used by the MLP workload.
The Conda environment is never removed.
EOF
            exit 0
            ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done

rm -rf -- "$BENCHMARK_DIR/results" "$BENCHMARK_DIR/.build"
if [[ "$remove_data" == true ]]; then
    rm -rf -- "$BENCHMARK_DIR/../mnist-mlp/data"
fi
echo "Removed MNIST CNN results and Swift build files."
if [[ "$remove_data" == true ]]; then echo "Removed the shared MNIST cache."; fi
echo "Conda environment was left untouched."
