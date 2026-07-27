#!/bin/bash
# Run the full IB Gateway + gex restart once inside the NY premarket window.
# In the NY post-close window, stop the gex stack once for the trading date.

set -euo pipefail

# launchd provides a minimal PATH, so Homebrew tools such as tmux may not be
# visible unless we add the usual user shell locations explicitly.
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:${PATH:-}"

ROOT_DIR="${GEX_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
ROOT_DIR="$(cd "$ROOT_DIR" && pwd)"
AUTOMATION_DIR="${IB_RESTART_HOME:-$HOME/.scripts/ib_gateway_restart}"
LOG_DIR="$AUTOMATION_DIR/logs"
STATE_DIR="$AUTOMATION_DIR/state"
STATE_FILE="$STATE_DIR/last_premarket_full_restart_ny_date"
CLOSE_STATE_FILE="$STATE_DIR/last_postclose_gex_stop_ny_date"
LOCK_DIR="$STATE_DIR/premarket_full_restart.lock"
CLOSE_LOCK_DIR="$STATE_DIR/postclose_gex_stop.lock"
LOG_FILE="$LOG_DIR/premarket_full_restart_$(date +%Y%m%d).log"

FULL_RESTART_SCRIPT="${FULL_RESTART_SCRIPT:-$ROOT_DIR/restart_ib_gateway_then_trade.sh}"
GEX_MODE="${GEX_MODE:-trade}"
WINDOW_START_MIN="${WINDOW_START_MIN:-540}"  # 09:00 ET
WINDOW_END_MIN="${WINDOW_END_MIN:-550}"      # 09:10 ET
GEX_CLOSE_WINDOW_START_MIN="${GEX_CLOSE_WINDOW_START_MIN:-965}"  # 16:05 ET
GEX_CLOSE_WINDOW_END_MIN="${GEX_CLOSE_WINDOW_END_MIN:-980}"      # 16:20 ET
GEX_TMUX_SESSION="${GEX_TMUX_SESSION:-gex-trade-stack}"
GEX_SAFE_START_SCRIPT="${GEX_SAFE_START_SCRIPT:-$HOME/.scripts/gex_runtime/start_gex_stack.sh}"
PORT_WAIT_SEC="${PORT_WAIT_SEC:-300}"
GEX_WAIT_SEC="${GEX_WAIT_SEC:-120}"
IB_HOST="${IB_HOST:-${IB_GATEWAY_HOST:-127.0.0.1}}"
IB_PORT="${IB_PORT:-${IB_GATEWAY_PORT:-4002}}"

mkdir -p "$LOG_DIR" "$STATE_DIR"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$*" | tee -a "$LOG_FILE"
}

cleanup() {
  rmdir "$LOCK_DIR" >/dev/null 2>&1 || true
  rmdir "$CLOSE_LOCK_DIR" >/dev/null 2>&1 || true
}

stop_gex_stack() {
  if [ -r "$GEX_SAFE_START_SCRIPT" ]; then
    /bin/bash "$GEX_SAFE_START_SCRIPT" stop
  elif [ -r "$ROOT_DIR/start.sh" ]; then
    /bin/bash "$ROOT_DIR/start.sh" stop
  else
    log "warn: no readable gex stop script: $GEX_SAFE_START_SCRIPT or $ROOT_DIR/start.sh"
  fi
  if command -v tmux >/dev/null 2>&1; then
    tmux kill-session -t "$GEX_TMUX_SESSION" >/dev/null 2>&1 || true
  fi
}

ny_date="$(TZ=America/New_York date '+%Y-%m-%d')"
ny_hour="$(TZ=America/New_York date '+%H')"
ny_minute="$(TZ=America/New_York date '+%M')"
ny_weekday="$(TZ=America/New_York date '+%u')"  # 1=Mon ... 7=Sun
ny_total_min=$((10#$ny_hour * 60 + 10#$ny_minute))

force="${FORCE:-0}"
check_only="${CHECK_ONLY:-0}"

if [ "$force" != "1" ] && [ "$force" != "true" ]; then
  if [ "$ny_weekday" -gt 5 ]; then
    log "skip: NY date=$ny_date weekend weekday=$ny_weekday"
    exit 0
  fi
  if [ "$ny_total_min" -ge "$GEX_CLOSE_WINDOW_START_MIN" ] && [ "$ny_total_min" -le "$GEX_CLOSE_WINDOW_END_MIN" ]; then
    if [ -f "$CLOSE_STATE_FILE" ] && [ "$(cat "$CLOSE_STATE_FILE")" = "$ny_date" ]; then
      log "skip: post-close gex stop already completed for NY date $ny_date"
      exit 0
    fi
    if ! mkdir "$CLOSE_LOCK_DIR" 2>/dev/null; then
      log "skip: another post-close gex stop is already running"
      exit 0
    fi
    trap cleanup EXIT
    if [ "$check_only" = "1" ] || [ "$check_only" = "true" ]; then
      log "check_only: would stop gex stack for NY date $ny_date window=$GEX_CLOSE_WINDOW_START_MIN-$GEX_CLOSE_WINDOW_END_MIN"
      exit 0
    fi
    log "post-close gex stop begin: NY=$ny_date $(TZ=America/New_York date '+%H:%M %Z') root=$ROOT_DIR tmux=$GEX_TMUX_SESSION"
    stop_gex_stack 2>&1 | tee -a "$LOG_FILE"
    echo "$ny_date" > "$CLOSE_STATE_FILE"
    log "post-close gex stop completed for NY date $ny_date"
    exit 0
  fi

  if [ "$ny_total_min" -lt "$WINDOW_START_MIN" ] || [ "$ny_total_min" -gt "$WINDOW_END_MIN" ]; then
    log "skip: NY time $(TZ=America/New_York date '+%H:%M') outside full restart window"
    exit 0
  fi
  if [ -f "$STATE_FILE" ] && [ "$(cat "$STATE_FILE")" = "$ny_date" ]; then
    log "skip: full restart already completed for NY date $ny_date"
    exit 0
  fi
fi

if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  log "skip: another premarket full restart is already running"
  exit 0
fi
trap cleanup EXIT

if [ ! -r "$FULL_RESTART_SCRIPT" ]; then
  log "error: full restart script is not readable: $FULL_RESTART_SCRIPT"
  exit 1
fi

if [ "$check_only" = "1" ] || [ "$check_only" = "true" ]; then
  log "check_only: would run /bin/bash $FULL_RESTART_SCRIPT $GEX_MODE for NY date $ny_date window=$WINDOW_START_MIN-$WINDOW_END_MIN"
  exit 0
fi

log "full restart begin: script=/bin/bash $FULL_RESTART_SCRIPT mode=$GEX_MODE host=$IB_HOST port=$IB_PORT NY=$ny_date $(TZ=America/New_York date '+%H:%M %Z')"

GEX_ROOT="$ROOT_DIR" \
PORT_WAIT_SEC="$PORT_WAIT_SEC" \
GEX_WAIT_SEC="$GEX_WAIT_SEC" \
IB_HOST="$IB_HOST" \
IB_PORT="$IB_PORT" \
/bin/bash "$FULL_RESTART_SCRIPT" "$GEX_MODE" 2>&1 | tee -a "$LOG_FILE"

echo "$ny_date" > "$STATE_FILE"
log "full restart completed for NY date $ny_date"
