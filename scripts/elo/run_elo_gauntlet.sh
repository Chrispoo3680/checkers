#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG_PATH="${1:-$SCRIPT_DIR/elo.env}"

if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "Config not found: $CONFIG_PATH" >&2
    echo "Copy $SCRIPT_DIR/elo.env.preset-rating-pool to $SCRIPT_DIR/elo.env and edit it." >&2
    exit 1
fi

# shellcheck disable=SC1090
source "$CONFIG_PATH"

: "${CUTECHESS_BIN:=cutechess-cli}"
: "${ORDO_BIN:=ordo}"
: "${ENGINE_A_NAME:?ENGINE_A_NAME is required}"
: "${ENGINE_A_CMD:?ENGINE_A_CMD is required}"
: "${OPENINGS_FILE:?OPENINGS_FILE is required}"

: "${ENGINE_A_PROTO:=uci}"
: "${OPENINGS_FORMAT:=pgn}"
: "${OPENINGS_ORDER:=random}"
: "${OPENINGS_PLIES:=16}"
: "${TC:=10+0.1}"
: "${CONCURRENCY:=1}"
: "${TIMEMARGIN_MS:=100}"
: "${RESULTS_DIR:=results/elo}"
: "${RUN_TAG:=$(date +%Y%m%d_%H%M%S)}"
: "${PGN_NAME:=matches.pgn}"
: "${CUTECHESS_LOG_NAME:=cutechess.log}"
: "${ORDO_RATINGS_NAME:=ordo_ratings.txt}"
: "${GAUNTLET_SUMMARY_NAME:=gauntlet_summary.txt}"

if [[ ! -f "$OPENINGS_FILE" ]]; then
    echo "Openings file not found: $OPENINGS_FILE" >&2
    exit 1
fi

if ! command -v "$CUTECHESS_BIN" >/dev/null 2>&1; then
    echo "cutechess binary not found: $CUTECHESS_BIN" >&2
    exit 1
fi

if ! command -v "$ORDO_BIN" >/dev/null 2>&1; then
    echo "ordo binary not found: $ORDO_BIN" >&2
    exit 1
fi

RUN_DIR="$REPO_ROOT/$RESULTS_DIR/$RUN_TAG"
mkdir -p "$RUN_DIR"

PGN_OUT="$RUN_DIR/$PGN_NAME"
LOG_OUT="$RUN_DIR/$CUTECHESS_LOG_NAME"
RATINGS_OUT="$RUN_DIR/$ORDO_RATINGS_NAME"
SUMMARY_OUT="$RUN_DIR/$GAUNTLET_SUMMARY_NAME"

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

collect_pool_engines() {
    local -n engines_ref=$1
    local i=1
    
    while true; do
        local var_name="POOL_ENGINE_${i}_NAME"
        if [[ -z "${!var_name:-}" ]]; then
            break
        fi
        engines_ref+=("$i")
        ((i++))
    done
}

pool_engines=()
collect_pool_engines pool_engines

if [[ ${#pool_engines[@]} -eq 0 ]]; then
    echo "Error: No pool engines defined (POOL_ENGINE_1_NAME, etc.)" >&2
    exit 1
fi

{
    echo "=== ELO GAUNTLET RUN ==="
    echo "Run tag: $RUN_TAG"
    echo "Run dir: $RUN_DIR"
    echo "Your engine: $ENGINE_A_NAME"
    echo "Pool opponents: ${#pool_engines[@]}"
    echo "Openings: $OPENINGS_FILE"
    echo "Time control: $TC"
    echo ""
} | tee "$LOG_OUT"

# Accumulate all PGNs into one file for final Ordo run
temp_pgn_files=()

for engine_idx in "${pool_engines[@]}"; do
    local engine_name_var="POOL_ENGINE_${engine_idx}_NAME"
    local engine_cmd_var="POOL_ENGINE_${engine_idx}_CMD"
    local engine_proto_var="POOL_ENGINE_${engine_idx}_PROTO"
    local engine_options_var="POOL_ENGINE_${engine_idx}_OPTIONS"
    local engine_games_var="POOL_ENGINE_${engine_idx}_games"

    local engine_name="${!engine_name_var:-}"
    local engine_cmd="${!engine_cmd_var:-}"
    local engine_proto="${!engine_proto_var:=uci}"
    local engine_options="${!engine_options_var:-}"
    local engine_games="${!engine_games_var:=100}"

    if [[ -z "$engine_name" ]] || [[ -z "$engine_cmd" ]]; then
        continue
    fi

    echo "" | tee -a "$LOG_OUT"
    echo "=== Match $engine_idx: $ENGINE_A_NAME vs $engine_name ===" | tee -a "$LOG_OUT"
    echo "Games: $engine_games | TC: $TC | Engine cmd: $engine_cmd" | tee -a "$LOG_OUT"
    echo "" | tee -a "$LOG_OUT"

    ENGINE_A_ARGS=(
        -engine
        "name=$ENGINE_A_NAME"
        "cmd=$ENGINE_A_CMD"
        "proto=$ENGINE_A_PROTO"
    )
    append_engine_options "${ENGINE_A_OPTIONS:-}" ENGINE_A_ARGS

    ENGINE_B_ARGS=(
        -engine
        "name=$engine_name"
        "cmd=$engine_cmd"
        "proto=$engine_proto"
    )
    append_engine_options "$engine_options" ENGINE_B_ARGS

    temp_pgn_file="$RUN_DIR/match_${engine_idx}_${engine_name}.pgn"
    temp_pgn_files+=("$temp_pgn_file")

    CMD=(
        "$CUTECHESS_BIN"
        "${ENGINE_A_ARGS[@]}"
        "${ENGINE_B_ARGS[@]}"
        -each "tc=$TC" "timemargin=$TIMEMARGIN_MS"
        -openings "file=$OPENINGS_FILE" "format=$OPENINGS_FORMAT" "order=$OPENINGS_ORDER" "plies=$OPENINGS_PLIES"
        -games "$engine_games"
        -repeat
        -recover
        -concurrency "$CONCURRENCY"
        -pgnout "$temp_pgn_file"
    )

    if [[ "${ENABLE_DRAW_ADJ:-1}" == "1" ]]; then
        CMD+=(
            -draw
            "movenumber=${DRAW_MOVENUMBER:=40}"
            "movecount=${DRAW_MOVECOUNT:=8}"
            "score=${DRAW_SCORE_CP:=10}"
        )
    fi

    if [[ "${ENABLE_RESIGN_ADJ:-1}" == "1" ]]; then
        CMD+=(
            -resign
            "movecount=${RESIGN_MOVECOUNT:=3}"
            "score=${RESIGN_SCORE_CP:=600}"
        )
    fi

    if [[ -n "${CUTECHESS_EXTRA_ARGS:-}" ]]; then
        read -r -a extra <<<"$CUTECHESS_EXTRA_ARGS"
        CMD+=("${extra[@]}")
    fi

    printf '  %q' "${CMD[@]}" | tee -a "$LOG_OUT"
    echo "" | tee -a "$LOG_OUT"

    "${CMD[@]}" | tee -a "$LOG_OUT"
done

# Combine all PGNs into final output
{
    echo ""
    echo "=== COMBINING MATCH PGNs ==="
} | tee -a "$LOG_OUT"

cat "${temp_pgn_files[@]}" > "$PGN_OUT"

{
    echo "Combined PGN: $PGN_OUT"
    echo ""
    echo "=== COMPUTING RATINGS WITH ORDO ==="
} | tee -a "$LOG_OUT"

CMD_ORDO=("$ORDO_BIN" -a "$PGN_OUT" -o "$RATINGS_OUT")

if [[ -n "${ORDO_EXTRA_ARGS:-}" ]]; then
    read -r -a extra <<<"$ORDO_EXTRA_ARGS"
    CMD_ORDO+=("${extra[@]}")
fi

printf '  %q' "${CMD_ORDO[@]}" | tee -a "$LOG_OUT"
echo "" | tee -a "$LOG_OUT"

"${CMD_ORDO[@]}" | tee -a "$LOG_OUT"

# Generate summary
{
    echo ""
    echo "=== GAUNTLET SUMMARY ==="
    echo "Run: $RUN_TAG"
    echo "Engine: $ENGINE_A_NAME"
    echo "Pool size: ${#pool_engines[@]} opponents"
    echo "Total games: $(grep -c "1-0\|0-1\|1/2-1/2" "$PGN_OUT" || echo '?')"
    echo ""
    echo "Ratings output:"
} | tee -a "$LOG_OUT"

cat "$RATINGS_OUT" | tee -a "$LOG_OUT"

cp "$LOG_OUT" "$SUMMARY_OUT"

echo "" | tee -a "$LOG_OUT"
echo "=== COMPLETE ===" | tee -a "$LOG_OUT"
echo "Results saved to: $RUN_DIR" | tee -a "$LOG_OUT"
echo "  PGN:       $PGN_OUT" | tee -a "$LOG_OUT"
echo "  Ratings:   $RATINGS_OUT" | tee -a "$LOG_OUT"
echo "  Full log:  $SUMMARY_OUT" | tee -a "$LOG_OUT"
