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

: "${CUTECHESS_BIN:=cutechess-cli}"
: "${ENGINE_A_NAME:?ENGINE_A_NAME is required}"
: "${ENGINE_A_CMD:?ENGINE_A_CMD is required}"
: "${ENGINE_B_NAME:?ENGINE_B_NAME is required}"
: "${ENGINE_B_CMD:?ENGINE_B_CMD is required}"
: "${OPENINGS_FILE:?OPENINGS_FILE is required}"

: "${ENGINE_A_PROTO:=uci}"
: "${ENGINE_B_PROTO:=uci}"
: "${OPENINGS_FORMAT:=pgn}"
: "${OPENINGS_ORDER:=random}"
: "${OPENINGS_PLIES:=16}"
: "${TC:=10+0.1}"
: "${GAMES:=200}"
: "${CONCURRENCY:=1}"
: "${TIMEMARGIN_MS:=100}"
: "${ENABLE_DRAW_ADJ:=1}"
: "${DRAW_MOVENUMBER:=40}"
: "${DRAW_MOVECOUNT:=8}"
: "${DRAW_SCORE_CP:=10}"
: "${ENABLE_RESIGN_ADJ:=1}"
: "${RESIGN_MOVECOUNT:=3}"
: "${RESIGN_SCORE_CP:=600}"
: "${RESULTS_DIR:=results/elo}"
: "${RUN_TAG:=$(date +%Y%m%d_%H%M%S)}"
: "${PGN_NAME:=matches.pgn}"
: "${CUTECHESS_LOG_NAME:=cutechess.log}"
: "${CUTECHESS_EXTRA_ARGS:=}"

if [[ ! -f "$OPENINGS_FILE" ]]; then
    echo "Openings file not found: $OPENINGS_FILE" >&2
    exit 1
fi

if ! command -v "$CUTECHESS_BIN" >/dev/null 2>&1; then
    echo "cutechess binary not found: $CUTECHESS_BIN" >&2
    exit 1
fi

RUN_DIR="$REPO_ROOT/$RESULTS_DIR/$RUN_TAG"
mkdir -p "$RUN_DIR"

PGN_OUT="$RUN_DIR/$PGN_NAME"
LOG_OUT="$RUN_DIR/$CUTECHESS_LOG_NAME"

append_engine_options() {
    local options="$1"
    local -n arr_ref=$2

    if [[ -z "$options" ]]; then
        return
    fi

    IFS=';' read -r -a pairs <<<"$options"
    for pair in "${pairs[@]}"; do
        pair="${pair#${pair%%[![:space:]]*}}"
        pair="${pair%${pair##*[![:space:]]}}"
        [[ -z "$pair" ]] && continue
        arr_ref+=("option.$pair")
    done
}

ENGINE_A_ARGS=(
    -engine
    "name=$ENGINE_A_NAME"
    "cmd=$ENGINE_A_CMD"
    "proto=$ENGINE_A_PROTO"
)
append_engine_options "${ENGINE_A_OPTIONS:-}" ENGINE_A_ARGS

ENGINE_B_ARGS=(
    -engine
    "name=$ENGINE_B_NAME"
    "cmd=$ENGINE_B_CMD"
    "proto=$ENGINE_B_PROTO"
)
append_engine_options "${ENGINE_B_OPTIONS:-}" ENGINE_B_ARGS

CMD=(
    "$CUTECHESS_BIN"
    "${ENGINE_A_ARGS[@]}"
    "${ENGINE_B_ARGS[@]}"
    -each "tc=$TC" "timemargin=$TIMEMARGIN_MS"
    -openings "file=$OPENINGS_FILE" "format=$OPENINGS_FORMAT" "order=$OPENINGS_ORDER" "plies=$OPENINGS_PLIES"
    -games "$GAMES"
    -repeat
    -recover
    -concurrency "$CONCURRENCY"
    -pgnout "$PGN_OUT"
)

if [[ "$ENABLE_DRAW_ADJ" == "1" ]]; then
    CMD+=(
        -draw
        "movenumber=$DRAW_MOVENUMBER"
        "movecount=$DRAW_MOVECOUNT"
        "score=$DRAW_SCORE_CP"
    )
fi

if [[ "$ENABLE_RESIGN_ADJ" == "1" ]]; then
    CMD+=(
        -resign
        "movecount=$RESIGN_MOVECOUNT"
        "score=$RESIGN_SCORE_CP"
    )
fi

if [[ -n "$CUTECHESS_EXTRA_ARGS" ]]; then
    read -r -a extra <<<"$CUTECHESS_EXTRA_ARGS"
    CMD+=("${extra[@]}")
fi

{
    echo "Run tag: $RUN_TAG"
    echo "Run dir: $RUN_DIR"
    echo "PGN out: $PGN_OUT"
    echo "Command:"
    printf '  %q' "${CMD[@]}"
    echo
} | tee "$LOG_OUT"

"${CMD[@]}" | tee -a "$LOG_OUT"

echo ""
echo "Match run finished."
echo "PGN: $PGN_OUT"
echo "Log: $LOG_OUT"
