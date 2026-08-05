# 多标的三信号动量 Universal v2.1：Python 版

实现位于 `src/gex_monitor/multi_asset_momentum.py`，接受按时间升序排列的 OHLCV 数据，输出
DEMA、VWAP、Kalman、ATR、投票结果和进场事件。所有计算均为逐根 K 线的因果计算。

## DataFrame 用法

```python
import pandas as pd

from gex_monitor.multi_asset_momentum import MomentumConfig, calculate_momentum_signals

bars = pd.read_csv("qqq_1m.csv", parse_dates=["timestamp"]).set_index("timestamp")
# 如果 CSV 时间没有时区，必须按数据实际时区本地化；不要直接假设 UTC。
bars.index = bars.index.tz_localize("America/New_York")

result = calculate_momentum_signals(
    bars,
    symbol="QQQ",
    config=MomentumConfig(),
)

new_orders = result.loc[
    result["long_entry"] | result["short_entry"] | result["early_long_entry"] | result["early_short_entry"]
]
```

`MomentumConfig` 中的 `None` 对应 Pine 参数里的“自动”。可通过 `asset="SPY"` 等参数覆盖
代码自动识别的标的类型。

## CSV 命令行

安装当前项目后：

```bash
universal-momentum qqq_1m.csv qqq_signals.csv --symbol QQQ --timezone America/New_York
```

输入 CSV 默认需要 `timestamp,open,high,low,close,volume` 六列。若 `timestamp` 已包含时区，
无需传 `--timezone`。

## 主要输出

- `sig1`：DEMA 方向，`sig2`：VWAP 偏离方向，`sig3`：Kalman 动量方向。
- `score`：三信号之和；`effective_vote`：当前所需票数。
- `long_signal` / `short_signal`：本根 K 线是否持续满足完整条件。
- `long_entry` / `short_entry`：完整信号由假变真的单根事件，对应 Pine 的进场 alert。
- `early_long_entry` / `early_short_entry`：Kalman 预警后的 DEMA 与量能确认事件。
- `in_session` / `atr_filter`：时段和 ATR 过滤器状态。

美股和期货时段使用 `America/New_York`，会自动处理夏令时；加密货币时段使用 UTC。
期货 `rth` VWAP 只在 09:30–16:15 ET 累计，`globex` VWAP 在 18:00 ET 重置。
