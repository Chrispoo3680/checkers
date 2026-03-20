#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${1:-$SCRIPT_DIR/elo.env}"

"$SCRIPT_DIR/run_cutechess_match.sh" "$CONFIG_PATH"
"$SCRIPT_DIR/run_ordo_rating.sh" "$CONFIG_PATH"
