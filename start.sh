#!/bin/bash
# GEX Monitor + KDJ Trader + Macro Dashboard 一键启动
# 用法：
#   ./start.sh              # 默认 dry-run
#   ./start.sh live         # 实盘下单
#   ./start.sh stop         # 停止所有

cd "$(dirname "$0")"

CONDA_ENV="gex"
CONFIG_FILE="../config/config.yaml"
GEX_LOG="logs/gex_$(date +%Y%m%d).log"
KDJ_LOG="logs/kdj_trader_$(date +%Y%m%d).log"
MACRO_LOG="logs/macro_$(date +%Y%m%d).log"
QQQ_BAR_LOG="logs/qqq_bars_$(date +%Y%m%d).log"
PID_FILE="logs/.pids"
QQQ_BAR_CLIENT_ID="${GEX_QQQ_BAR_CLIENT_ID:-210}"

mkdir -p logs

# 注意：不要换成 pgrep —— 本机 app 环境下 sysmond 不可用，
# pgrep 报 "Cannot get process list" 静默返回空，守卫会失效
find_matching_pids() {
  local my_pid=$$
  while read -r pid cmd; do
    [ -z "$pid" ] && continue
    [ "$pid" = "$my_pid" ] && continue
    for pattern in "$@"; do
      case "$cmd" in
        *find_matching_pids*) continue 2 ;;  # 跳过自身 subshell
        *"$pattern"*)
          echo "$pid"
          break
          ;;
      esac
    done
  done < <(ps -axo pid=,command=)
}

# conda activate 在脚本里需要 source
eval "$(conda shell.bash hook)"
conda activate $CONDA_ENV

case "${1:-dry}" in
  stop)
    stopped_any=0
    if [ -f "$PID_FILE" ]; then
      while read pid; do
        [ -z "$pid" ] && continue
        kill "$pid" 2>/dev/null && echo "Stopped PID $pid" && stopped_any=1
      done < "$PID_FILE"
      rm "$PID_FILE"
    fi
    stale_pids="$(find_matching_pids \
      "python -m gex_monitor.main" \
      "python -m gex_monitor.kdj_live_trader" \
      "python -m gex_monitor.macro_app" \
      "python -m gex_monitor.qqq_bars")"
    if [ -n "$stale_pids" ]; then
      echo "$stale_pids" | while read -r pid; do
        kill "$pid" 2>/dev/null && echo "Stopped stale PID $pid"
      done
      stopped_any=1
    fi
    if [ "$stopped_any" = "0" ]; then
      echo "No running processes found"
    fi
    exit 0
    ;;
  live|trade)
    DRY_FLAG=""
    GEX_HEDGE_FLAG=""
    echo "⚠️  LIVE MODE — 实盘下单"
    ;;
  *)
    DRY_FLAG="--dry-run"
    GEX_HEDGE_FLAG="--no-hedge"
    echo "📋 DRY RUN — 只看信号不下单"
    ;;
esac

echo ""
echo "═══════════════════════════════════════"
echo "  GEX Monitor + KDJ Trader"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "═══════════════════════════════════════"

EXISTING_PIDS="$(find_matching_pids \
  "python -m gex_monitor.main" \
  "python -m gex_monitor.kdj_live_trader" \
  "python -m gex_monitor.macro_app" \
  "python -m gex_monitor.qqq_bars")"
if [ -n "$EXISTING_PIDS" ]; then
  echo "❌ Existing GEX/KDJ/Macro process found: $EXISTING_PIDS"
  echo "   Run ./start.sh stop first, then start again."
  exit 1
fi

# 启动 GEX Monitor
echo "🔧 Starting GEX Monitor..."
cd src
python -m gex_monitor.main -c "$CONFIG_FILE" $GEX_HEDGE_FLAG > "../$GEX_LOG" 2>&1 &
GEX_PID=$!
cd ..
echo "   PID=$GEX_PID  Log=$GEX_LOG"

# 等 GEX 连上 IB（给 5 秒）
sleep 5
if ! kill -0 "$GEX_PID" 2>/dev/null; then
  echo "❌ GEX Monitor exited during startup; aborting."
  echo "   Last log lines:"
  tail -60 "$GEX_LOG" 2>/dev/null || true
  exit 1
fi

# 启动 QQQ 官方 1 分钟 Bar 采集。它独立使用一个 IB clientId，收盘后
# 会继续等待下一个交易日；stop 时通过 PID/进程模式一起回收。
echo "🔧 Starting official QQQ 1m bars..."
python -m gex_monitor.qqq_bars \
  -c "$CONFIG_FILE" --live --client-id "$QQQ_BAR_CLIENT_ID" \
  > "../$QQQ_BAR_LOG" 2>&1 &
QQQ_BAR_PID=$!
cd ..
echo "   PID=$QQQ_BAR_PID  Log=$QQQ_BAR_LOG  clientId=$QQQ_BAR_CLIENT_ID"
sleep 2
if ! kill -0 "$QQQ_BAR_PID" 2>/dev/null; then
  echo "❌ QQQ Bar collector exited during startup; aborting."
  tail -60 "$QQQ_BAR_LOG" 2>/dev/null || true
  kill "$GEX_PID" 2>/dev/null || true
  exit 1
fi

# 启动 KDJ Trader
echo "🔧 Starting KDJ Trader..."
cd src
python -m gex_monitor.kdj_live_trader $DRY_FLAG --qty 500 --port 4002 >> "../$KDJ_LOG" 2>&1 &
KDJ_PID=$!
cd ..
echo "   PID=$KDJ_PID  Log=$KDJ_LOG"
sleep 1
if ! kill -0 "$KDJ_PID" 2>/dev/null; then
  echo "❌ KDJ Trader exited during startup; aborting."
  echo "   Last log lines:"
  tail -60 "$KDJ_LOG" 2>/dev/null || true
  kill "$GEX_PID" 2>/dev/null || true
  kill "$QQQ_BAR_PID" 2>/dev/null || true
  exit 1
fi

# 启动 Macro Dashboard
echo "🔧 Starting Macro Dashboard..."
cd src
python -m gex_monitor.macro_app > "../$MACRO_LOG" 2>&1 &
MACRO_PID=$!
cd ..
echo "   PID=$MACRO_PID  Log=$MACRO_LOG"
sleep 1
if ! kill -0 "$MACRO_PID" 2>/dev/null; then
  echo "❌ Macro Dashboard exited during startup; aborting."
  echo "   Last log lines:"
  tail -60 "$MACRO_LOG" 2>/dev/null || true
  kill "$GEX_PID" 2>/dev/null || true
  kill "$QQQ_BAR_PID" 2>/dev/null || true
  kill "$KDJ_PID" 2>/dev/null || true
  exit 1
fi

# 保存 PID
echo "$GEX_PID" > "$PID_FILE"
echo "$QQQ_BAR_PID" >> "$PID_FILE"
echo "$KDJ_PID" >> "$PID_FILE"
echo "$MACRO_PID" >> "$PID_FILE"

echo ""
echo "✅ All running!"
echo "   GEX UI:    http://localhost:8050"
echo "   Macro UI:  http://localhost:8051"
echo "   KDJ Log:   tail -f $KDJ_LOG"
echo ""
echo "停止: ./start.sh stop"
