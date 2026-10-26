#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd -- "$BENCHMARK_DIR/../.." && pwd)"
COMMON_SCRIPTS="$REPO_ROOT/Benchmarking/mnist-mlp/scripts"

RUN_ID="${RUN_ID:-mnist-cnn-$(date -u +%Y%m%dT%H%M%SZ)}"
EPOCHS="${EPOCHS:-5}"
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-32}"
WARMUP_STEPS="${WARMUP_STEPS:-10}"
DATA_DIR="${MNIST_DATA_DIR:-$REPO_ROOT/Benchmarking/mnist-mlp/data}"
RESULTS_ROOT="${MNIST_CNN_RESULTS_DIR:-$BENCHMARK_DIR/results}"

usage() {
    cat <<'EOF'
Usage: run-mlxkit.sh [--run-id ID] [--epochs N] [--seed N] [--batch-size N] [--warmup-steps N]

Environment overrides: RUN_ID, EPOCHS, SEED, BATCH_SIZE, WARMUP_STEPS,
MNIST_DATA_DIR, MNIST_CNN_RESULTS_DIR, PYTHON
EOF
}

while (($#)); do
    case "$1" in
        --run-id) RUN_ID="${2:?Missing value for --run-id}"; shift 2 ;;
        --epochs) EPOCHS="${2:?Missing value for --epochs}"; shift 2 ;;
        --seed) SEED="${2:?Missing value for --seed}"; shift 2 ;;
        --batch-size) BATCH_SIZE="${2:?Missing value for --batch-size}"; shift 2 ;;
        --warmup-steps) WARMUP_STEPS="${2:?Missing value for --warmup-steps}"; shift 2 ;;
        --help|-h) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

case "$RUN_ID" in
    ''|.|..|*[!A-Za-z0-9._-]*) echo "RUN_ID may contain only letters, numbers, dot, underscore, and hyphen." >&2; exit 2 ;;
esac

if ! command -v swift >/dev/null 2>&1; then
    echo "Swift was not found. Install Xcode 27 / Swift 6.4 or later." >&2
    exit 127
fi

RESULTS_DIR="$RESULTS_ROOT/$RUN_ID"
mkdir -p "$RESULTS_DIR"
python_bin="${PYTHON:-python3}"
"$python_bin" "$COMMON_SCRIPTS/prepare-data.py" --data-dir "$DATA_DIR"
INITIALIZATION_DIR="$RESULTS_DIR/initialization"
"$python_bin" "$SCRIPT_DIR/prepare-initialization.py" --output-dir "$INITIALIZATION_DIR" --seed "$SEED"

revision="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
if [[ -n "$(git -C "$REPO_ROOT" status --porcelain --untracked-files=normal 2>/dev/null || true)" ]]; then
    export MLXKIT_GIT_DIRTY=true
else
    export MLXKIT_GIT_DIRTY=false
fi
export MLXKIT_GIT_REVISION="$revision"
export MNIST_SWIFT_VERSION="$(swift --version | head -n 1)"
echo "Run ID: $RUN_ID"

set -o pipefail
swift run --package-path "$REPO_ROOT" --scratch-path "$BENCHMARK_DIR/.build" -c release MNISTCNNBenchmark \
    --data-dir "$DATA_DIR" \
    --initialization-dir "$INITIALIZATION_DIR" \
    --output "$RESULTS_DIR/mlxkit.json" \
    --run-id "$RUN_ID" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --seed "$SEED" \
    --warmup-steps "$WARMUP_STEPS" \
    2>&1 | tee "$RESULTS_DIR/mlxkit.log"
