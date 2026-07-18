#!/bin/bash
# Launchd-safe GEX stack starter.
#
# This script is intended to live outside protected folders such as Downloads.
# It starts GEX/Macro plus the official QQQ 1m collector from a small runtime
# copy. KDJ is disabled by default; set GEX_ENABLE_KDJ=1 to opt in.
# rooted at ~/.scripts/gex_runtime by default.

set -euo pipefail

RUNTIME_DIR="${GEX_RUNTIME_DIR:-$(cd "$(dirname "$0")" && pwd)}"
PYTHON_BIN="${GEX_PYTHON_BIN:-/opt/homebrew/anaconda3/envs/gex/bin/python}"
CONFIG_FILE="${GEX_CONFIG_FILE:-$RUNTIME_DIR/config/config.yaml}"
ENV_FILE="${GEX_ENV_FILE:-$RUNTIME_DIR/.env}"
SRC_DIR="$RUNTIME_DIR/src"
LOG_DIR="$RUNTIME_DIR/logs"
PID_FILE="$LOG_DIR/.pids"
MODE="${1:-dry}"
QQQ_BAR_CLIENT_ID="${GEX_QQQ_BAR_CLIENT_ID:-210}"
SPY_BAR_CLIENT_ID="${GEX_SPY_BAR_CLIENT_ID:-211}"

mkdir -p "$LOG_DIR" "$SRC_DIR/data"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$*"
}

if [ -r "$ENV_FILE" ]; then
  set +e
  set -a
  # shellcheck disable=SC1090
  . "$ENV_FILE"
  env_load_status=$?
  set +a
  set -e
  if [ "$env_load_status" -eq 0 ]; then
    log "Loaded environment file: $ENV_FILE"
  else
    log "WARN: failed to load environment file: $ENV_FILE"
  fi
else
  log "No environment file loaded: $ENV_FILE"
fi

# GEX and the official-bar collector must share one canonical directory.  The
# runtime .env may override this (recommended: the project's data directory).
DATA_DIR="${GEX_DATA_DIR:-$RUNTIME_DIR/data}"
ENABLE_KDJ="${GEX_ENABLE_KDJ:-0}"
mkdir -p "$DATA_DIR"

find_matching_pids() {
  local my_pid=$$
  while read -r pid cmd; do
    [ -z "$pid" ] && continue
    [ "$pid" = "$my_pid" ] && continue
    for pattern in "$@"; do
      case "$cmd" in
        *find_matching_pids*) continue 2 ;;
        *"$pattern"*)
          echo "$pid"
          break
          ;;
      esac
    done
  done < <(ps -axo pid=,command=)
}

stop_stack() {
  local stopped_any=0
  if [ -f "$PID_FILE" ]; then
    while read -r pid; do
      [ -z "$pid" ] && continue
      kill "$pid" 2>/dev/null && log "Stopped PID $pid" && stopped_any=1
    done < "$PID_FILE"
    rm -f "$PID_FILE"
  fi

  local stale_pids
  stale_pids="$(find_matching_pids \
    "python -m gex_monitor.main" \
    "python -m gex_monitor.kdj_live_trader" \
    "python -m gex_monitor.macro_app" \
    "python -m gex_monitor.qqq_bars")"
  if [ -n "$stale_pids" ]; then
    echo "$stale_pids" | while read -r pid; do
      kill "$pid" 2>/dev/null && log "Stopped stale PID $pid"
    done
    stopped_any=1
  fi

  if [ "$stopped_any" = "0" ]; then
    log "No running GEX processes found"
  fi
}

case "$MODE" in
  stop)
    stop_stack
    exit 0
    ;;
  live|trade)
    DRY_FLAG=""
    GEX_HEDGE_FLAG=""
    log "LIVE MODE - real orders may be enabled"
    ;;
  *)
    DRY_FLAG="--dry-run"
    GEX_HEDGE_FLAG="--no-hedge"
    log "DRY RUN - signals only"
    ;;
esac

if [ ! -x "$PYTHON_BIN" ]; then
  log "ERROR: Python not executable: $PYTHON_BIN"
  exit 1
fi
if [ ! -r "$CONFIG_FILE" ]; then
  log "ERROR: config not readable: $CONFIG_FILE"
  exit 1
fi
if [ ! -d "$SRC_DIR/gex_monitor" ]; then
  log "ERROR: runtime source missing: $SRC_DIR/gex_monitor"
  exit 1
fi
if [ ! -r "$SRC_DIR/gex_monitor/qqq_bars.py" ]; then
  log "ERROR: official QQQ Bar module missing: $SRC_DIR/gex_monitor/qqq_bars.py"
  exit 1
fi

existing="$(find_matching_pids \
  "python -m gex_monitor.main" \
  "python -m gex_monitor.kdj_live_trader" \
  "python -m gex_monitor.macro_app" \
  "python -m gex_monitor.qqq_bars")"
if [ -n "$existing" ]; then
  log "ERROR: existing GEX/KDJ/Macro process found: $existing"
  log "Run $0 stop first, then start again."
  exit 1
fi

GEX_LOG="$LOG_DIR/gex_$(date +%Y%m%d).log"
KDJ_LOG="$LOG_DIR/kdj_trader_$(date +%Y%m%d).log"
MACRO_LOG="$LOG_DIR/macro_$(date +%Y%m%d).log"
QQQ_BAR_LOG="$LOG_DIR/qqq_bars_$(date +%Y%m%d).log"
SPY_BAR_LOG="$LOG_DIR/spy_bars_$(date +%Y%m%d).log"

log "Starting GEX stack from runtime: $RUNTIME_DIR"
log "GEX log: $GEX_LOG"

cd "$SRC_DIR"

nohup "$PYTHON_BIN" -m gex_monitor.main -c "$CONFIG_FILE" $GEX_HEDGE_FLAG > "$GEX_LOG" 2>&1 &
GEX_PID=$!
log "GEX PID=$GEX_PID"
sleep 5
if ! kill -0 "$GEX_PID" 2>/dev/null; then
  log "ERROR: GEX Monitor exited during startup"
  tail -80 "$GEX_LOG" 2>/dev/null || true
  exit 1
fi

log "Starting official QQQ 1m bars (clientId=$QQQ_BAR_CLIENT_ID)"
nohup "$PYTHON_BIN" -m gex_monitor.qqq_bars \
  -c "$CONFIG_FILE" --live --client-id "$QQQ_BAR_CLIENT_ID" \
  --symbol QQQ --data-dir "$DATA_DIR" \
  > "$QQQ_BAR_LOG" 2>&1 &
QQQ_BAR_PID=$!
log "QQQ Bar PID=$QQQ_BAR_PID"
sleep 2
if ! kill -0 "$QQQ_BAR_PID" 2>/dev/null; then
  log "ERROR: QQQ Bar collector exited during startup"
  tail -80 "$QQQ_BAR_LOG" 2>/dev/null || true
  kill "$GEX_PID" 2>/dev/null || true
  exit 1
fi

log "Starting official SPY 1m bars (clientId=$SPY_BAR_CLIENT_ID)"
nohup "$PYTHON_BIN" -m gex_monitor.qqq_bars \
  -c "$CONFIG_FILE" --live --client-id "$SPY_BAR_CLIENT_ID" \
  --symbol SPY --data-dir "$DATA_DIR" \
  > "$SPY_BAR_LOG" 2>&1 &
SPY_BAR_PID=$!
log "SPY Bar PID=$SPY_BAR_PID"
sleep 2
if ! kill -0 "$SPY_BAR_PID" 2>/dev/null; then
  log "ERROR: SPY Bar collector exited during startup"
  tail -80 "$SPY_BAR_LOG" 2>/dev/null || true
  kill "$GEX_PID" "$QQQ_BAR_PID" 2>/dev/null || true
  exit 1
fi

KDJ_PID=""
if [ "$ENABLE_KDJ" = "1" ]; then
  log "Starting KDJ Trader (explicitly enabled)"
  GEX_LOG_DIR="$LOG_DIR" nohup "$PYTHON_BIN" -m gex_monitor.kdj_live_trader $DRY_FLAG --qty 500 --port 4002 >> "$KDJ_LOG" 2>&1 &
  KDJ_PID=$!
  log "KDJ PID=$KDJ_PID"
  sleep 1
  if ! kill -0 "$KDJ_PID" 2>/dev/null; then
    log "ERROR: KDJ Trader exited during startup"
    tail -80 "$KDJ_LOG" 2>/dev/null || true
    kill "$GEX_PID" 2>/dev/null || true
    kill "$QQQ_BAR_PID" 2>/dev/null || true
    kill "$SPY_BAR_PID" 2>/dev/null || true
    exit 1
  fi
else
  log "KDJ disabled (set GEX_ENABLE_KDJ=1 to enable)"
fi

nohup "$PYTHON_BIN" -m gex_monitor.macro_app > "$MACRO_LOG" 2>&1 &
MACRO_PID=$!
log "Macro PID=$MACRO_PID"
sleep 1
if ! kill -0 "$MACRO_PID" 2>/dev/null; then
  log "ERROR: Macro Dashboard exited during startup"
  tail -80 "$MACRO_LOG" 2>/dev/null || true
  kill "$GEX_PID" "$QQQ_BAR_PID" "$SPY_BAR_PID" 2>/dev/null || true
  if [ -n "$KDJ_PID" ]; then
    kill "$KDJ_PID" 2>/dev/null || true
  fi
  exit 1
fi

{
  printf '%s\n%s\n%s\n' "$GEX_PID" "$QQQ_BAR_PID" "$SPY_BAR_PID"
  if [ -n "$KDJ_PID" ]; then
    printf '%s\n' "$KDJ_PID"
  fi
  printf '%s\n' "$MACRO_PID"
} > "$PID_FILE"

log "All running"
log "GEX UI: http://localhost:8050"
log "Macro UI: http://localhost:8051"
