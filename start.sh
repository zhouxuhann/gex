#!/bin/bash
# GEX Monitor + KDJ Trader 一键启动
# 用法：
#   ./start.sh              # 默认 dry-run
#   ./start.sh live         # 实盘下单
#   ./start.sh stop         # 停止所有

cd "$(dirname "$0")"

CONDA_ENV="gex"
GEX_LOG="logs/gex_$(date +%Y%m%d).log"
KDJ_LOG="logs/kdj_trader_$(date +%Y%m%d).log"
PID_FILE="logs/.pids"

mkdir -p logs

# conda activate 在脚本里需要 source
eval "$(conda shell.bash hook)"
conda activate $CONDA_ENV

case "${1:-dry}" in
  stop)
    if [ -f "$PID_FILE" ]; then
      while read pid; do
        kill "$pid" 2>/dev/null && echo "Stopped PID $pid"
      done < "$PID_FILE"
      rm "$PID_FILE"
    else
      echo "No running processes found"
    fi
    exit 0
    ;;
  live)
    DRY_FLAG=""
    echo "⚠️  LIVE MODE — 实盘下单"
    ;;
  *)
    DRY_FLAG="--dry-run"
    echo "📋 DRY RUN — 只看信号不下单"
    ;;
esac

echo ""
echo "═══════════════════════════════════════"
echo "  GEX Monitor + KDJ Trader"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "═══════════════════════════════════════"

# 启动 GEX Monitor
echo "🔧 Starting GEX Monitor..."
cd src
python -m gex_monitor.main > "../$GEX_LOG" 2>&1 &
GEX_PID=$!
cd ..
echo "   PID=$GEX_PID  Log=$GEX_LOG"

# 等 GEX 连上 IB（给 5 秒）
sleep 5

# 启动 KDJ Trader
echo "🔧 Starting KDJ Trader..."
cd src
python -m gex_monitor.kdj_live_trader $DRY_FLAG --qty 500 --port 4002 >> "../$KDJ_LOG" 2>&1 &
KDJ_PID=$!
cd ..
echo "   PID=$KDJ_PID  Log=$KDJ_LOG"

# 保存 PID
echo "$GEX_PID" > "$PID_FILE"
echo "$KDJ_PID" >> "$PID_FILE"

echo ""
echo "✅ Both running!"
echo "   GEX UI: http://localhost:8050"
echo "   KDJ Log: tail -f $KDJ_LOG"
echo ""
echo "停止: ./start.sh stop"
echo "查看: tail -f $KDJ_LOG"
