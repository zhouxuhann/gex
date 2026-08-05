#!/bin/bash
# Restart IB Gateway first, then restart the gex trade stack.

set -euo pipefail

# launchd provides a minimal PATH, so Homebrew tools such as tmux may not be
# visible unless we add the usual user shell locations explicitly.
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:${PATH:-}"

ROOT_DIR="${GEX_ROOT:-$(cd "$(dirname "$0")" && pwd)}"
ROOT_DIR="$(cd "$ROOT_DIR" && pwd)"
AUTOMATION_HOME="${IB_RESTART_HOME:-$HOME/.scripts/ib_gateway_restart}"
AUTOMATION_LOG_DIR="$AUTOMATION_HOME/logs"
SAFE_START_SCRIPT="${GEX_SAFE_START_SCRIPT:-$HOME/.scripts/gex_runtime/start_gex_stack.sh}"
START_SCRIPT="${START_SCRIPT:-$SAFE_START_SCRIPT}"
IB_RESTART_SCRIPT="${IB_RESTART_SCRIPT:-$HOME/.scripts/ib_gateway_premarket_restart.sh}"
IB_HOST="${IB_HOST:-127.0.0.1}"
IB_PORT="${IB_PORT:-4002}"
PORT_WAIT_SEC="${PORT_WAIT_SEC:-300}"
GEX_WAIT_SEC="${GEX_WAIT_SEC:-90}"
GEX_MODE="${1:-${GEX_MODE:-trade}}"
DRY_RUN="${DRY_RUN:-0}"
GEX_START_LOG="${GEX_START_LOG:-$AUTOMATION_LOG_DIR/restart_ib_gateway_then_${GEX_MODE}_$(date +%Y%m%d_%H%M%S).log}"
GEX_TMUX_SESSION="${GEX_TMUX_SESSION:-gex-trade-stack}"
START_MODE="$GEX_MODE"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$*"
}

die() {
  log "ERROR: $*"
  exit 1
}

run_cmd() {
  if [ "$DRY_RUN" = "1" ] || [ "$DRY_RUN" = "true" ]; then
    printf 'DRY_RUN:'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

run_ib_restart() {
  if [ "$DRY_RUN" = "1" ] || [ "$DRY_RUN" = "true" ]; then
    printf 'DRY_RUN: FORCE=1 PORT_WAIT_SEC=%q IB_GATEWAY_HOST=%q IB_GATEWAY_PORT=%q %q\n' \
      "$PORT_WAIT_SEC" "$IB_HOST" "$IB_PORT" "$IB_RESTART_SCRIPT"
    return 0
  fi

  FORCE=1 \
  PORT_WAIT_SEC="$PORT_WAIT_SEC" \
  IB_GATEWAY_HOST="$IB_HOST" \
  IB_GATEWAY_PORT="$IB_PORT" \
  "$IB_RESTART_SCRIPT"
}

wait_for_ib_port() {
  if [ "$DRY_RUN" = "1" ] || [ "$DRY_RUN" = "true" ]; then
    log "DRY_RUN: would wait for $IB_HOST:$IB_PORT"
    return 0
  fi

  local deadline
  deadline=$((SECONDS + PORT_WAIT_SEC))
  while true; do
    if nc -z "$IB_HOST" "$IB_PORT" >/dev/null 2>&1; then
      log "IB Gateway is reachable at $IB_HOST:$IB_PORT"
      return 0
    fi
    if [ "$SECONDS" -ge "$deadline" ]; then
      die "IB Gateway port $IB_HOST:$IB_PORT did not become reachable within ${PORT_WAIT_SEC}s"
    fi
    sleep 5
  done
}

wait_for_port() {
  local host="$1"
  local port="$2"
  local label="$3"
  local deadline
  deadline=$((SECONDS + GEX_WAIT_SEC))
  while true; do
    if nc -z "$host" "$port" >/dev/null 2>&1; then
      log "$label is reachable at $host:$port"
      return 0
    fi
    if [ "$SECONDS" -ge "$deadline" ]; then
      die "$label port $host:$port did not become reachable within ${GEX_WAIT_SEC}s"
    fi
    sleep 2
  done
}

run_gex_start() {
  if [ "$DRY_RUN" = "1" ] || [ "$DRY_RUN" = "true" ]; then
    printf 'DRY_RUN: tmux new-session -d -s %q ... /bin/bash %q %q\n' "$GEX_TMUX_SESSION" "$START_SCRIPT" "$START_MODE"
    return 0
  fi

  mkdir -p "$AUTOMATION_LOG_DIR"
  log "start output: $GEX_START_LOG"
  if ! command -v tmux >/dev/null 2>&1; then
    die "tmux is required to keep the gex stack alive from this wrapper"
  fi

  tmux kill-session -t "$GEX_TMUX_SESSION" >/dev/null 2>&1 || true
  local tmux_cmd
  local start_cwd
  start_cwd="$(cd "$(dirname "$START_SCRIPT")" && pwd)"
  printf -v tmux_cmd 'cd %q && IB_HOST=%q IB_PORT=%q /bin/bash %q %q > %q 2>&1; exec bash -l' \
    "$start_cwd" "$IB_HOST" "$IB_PORT" "$START_SCRIPT" "$START_MODE" "$GEX_START_LOG"
  tmux new-session -d -s "$GEX_TMUX_SESSION" "$tmux_cmd"

  local deadline
  deadline=$((SECONDS + GEX_WAIT_SEC))
  while [ ! -s "$GEX_START_LOG" ]; do
    if [ "$SECONDS" -ge "$deadline" ]; then
      die "$START_SCRIPT did not write output within ${GEX_WAIT_SEC}s"
    fi
    sleep 1
  done

  while ! grep -q "All running" "$GEX_START_LOG" 2>/dev/null; do
    if grep -q "❌" "$GEX_START_LOG" 2>/dev/null; then
      tail -n 120 "$GEX_START_LOG" 2>/dev/null || true
      die "$START_SCRIPT $GEX_MODE reported an error"
    fi
    if [ "$SECONDS" -ge "$deadline" ]; then
      tail -n 120 "$GEX_START_LOG" 2>/dev/null || true
      die "$START_SCRIPT $GEX_MODE did not finish within ${GEX_WAIT_SEC}s"
    fi
    sleep 1
  done

  tail -n 80 "$GEX_START_LOG" 2>/dev/null || true
}

wait_for_gex_stack() {
  if [ "$DRY_RUN" = "1" ] || [ "$DRY_RUN" = "true" ]; then
    log "DRY_RUN: would wait for GEX UI 8050 and Macro UI 8051"
    return 0
  fi

  wait_for_port 127.0.0.1 8050 "GEX UI"
  wait_for_port 127.0.0.1 8051 "Macro UI"
}

case "$GEX_MODE" in
  trade|live)
    START_MODE="live"
    ;;
  paper|dry|dry-run|dry_run)
    START_MODE="dry"
    ;;
  *)
    die "unknown gex mode: $GEX_MODE (expected trade, dry, or live)"
    ;;
esac

[ -r "$START_SCRIPT" ] || die "start script is not readable: $START_SCRIPT"

if [ ! -x "$IB_RESTART_SCRIPT" ]; then
  if [ -x "$ROOT_DIR/scripts/ib_gateway_premarket_restart.sh" ]; then
    IB_RESTART_SCRIPT="$ROOT_DIR/scripts/ib_gateway_premarket_restart.sh"
  else
    die "IB restart script is not executable: $IB_RESTART_SCRIPT"
  fi
fi

log "Step 1/3: restart IB Gateway via $IB_RESTART_SCRIPT"
run_ib_restart
wait_for_ib_port

cd "$ROOT_DIR"

log "Step 2/3: stop gex project"
run_cmd /bin/bash "$START_SCRIPT" stop

log "Step 3/3: start gex project mode=$GEX_MODE start_mode=$START_MODE"
run_gex_start
wait_for_gex_stack

log "Done"
