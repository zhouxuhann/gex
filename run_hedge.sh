#!/bin/bash
# 每日对冲信号 — cron 调用
# 日本时间 04:30 = 美东 15:30 (收盘前30分钟)

cd /Users/fanzhouxu/Downloads/gex

LOG_DIR="logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/hedge_$(date +%Y%m%d).log"

echo "=== $(date) ===" >> "$LOG_FILE"

# 激活 conda 环境
eval "$(conda shell.bash hook)"
conda activate base

# 执行: Paper 账户自动下单
python -m gex_monitor.hedge \
    --execute \
    -s QQQ \
    -s SPY \
    --ib-port 4002 \
    >> "$LOG_FILE" 2>&1

echo "Exit code: $?" >> "$LOG_FILE"
