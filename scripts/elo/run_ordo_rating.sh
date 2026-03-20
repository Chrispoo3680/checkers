#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG_PATH="${1:-$SCRIPT_DIR/elo.env}"

if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "Config not found: $CONFIG_PATH" >&2
    echo "Copy $SCRIPT_DIR/elo.env.example to $SCRIPT_DIR/elo.env and edit it." >&2
    exit 1
fi

# shellcheck disable=SC1090
source "$CONFIG_PATH"

: "${ORDO_BIN:=ordo}"
: "${RESULTS_DIR:=results/elo}"
: "${RUN_TAG:=}"
: "${PGN_NAME:=matches.pgn}"
: "${ORDO_RATINGS_NAME:=ordo_ratings.txt}"
: "${ORDO_EXTRA_ARGS:=}"

if ! command -v "$ORDO_BIN" >/dev/null 2>&1; then
    echo "Ordo binary not found: $ORDO_BIN" >&2
    exit 1
fi

resolve_latest_run_tag() {
    local base_dir="$REPO_ROOT/$RESULTS_DIR"
    if [[ ! -d "$base_dir" ]]; then
        echo ""; return
    fi
    ls -1 "$base_dir" 2>/dev/null | sort | tail -n 1
}

if [[ -z "$RUN_TAG" ]]; then
    RUN_TAG="$(resolve_latest_run_tag)"
fi

if [[ -z "$RUN_TAG" ]]; then
    echo "No RUN_TAG set and no prior runs found under $REPO_ROOT/$RESULTS_DIR" >&2
    exit 1
fi

RUN_DIR="$REPO_ROOT/$RESULTS_DIR/$RUN_TAG"
PGN_INPUT="$RUN_DIR/$PGN_NAME"
RATINGS_OUT="$RUN_DIR/$ORDO_RATINGS_NAME"

if [[ ! -f "$PGN_INPUT" ]]; then
    echo "PGN not found: $PGN_INPUT" >&2
    exit 1
fi

CMD=("$ORDO_BIN" -a "$PGN_INPUT" -o "$RATINGS_OUT")

if [[ -n "$ORDO_EXTRA_ARGS" ]]; then
    read -r -a extra <<<"$ORDO_EXTRA_ARGS"
    CMD+=("${extra[@]}")
fi

echo "Run tag: $RUN_TAG"
echo "PGN input: $PGN_INPUT"
echo "Ratings out: $RATINGS_OUT"
echo "Command:"
printf '  %q' "${CMD[@]}"
echo

"${CMD[@]}"

echo ""
echo "Ordo rating complete: $RATINGS_OUT"
