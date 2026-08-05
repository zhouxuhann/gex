# Universal v2.1：IB Paper 0DTE 自动交易

交易器读取 IB Gateway 的 QQQ、SPY 一分钟已完成 K 线，并分别运行 Universal v2.1 指标。

## 交易规则

- `long_entry` 首次出现：买入当日到期 Call。
- `short_entry` 首次出现：买入当日到期 Put。
- 每个方向选择三档且默认各买 1 张：执行价约 `$1 ITM`、ATM、约 `$1 OTM`。
- 当前完整投票箭头消失：卖出该标的所有策略持仓；成交结果记录为
  `take_profit` 或 `stop_loss`。
- 箭头直接反转：先平旧方向；只有全部平仓成功才允许买新方向。
- 黄色早进场、Kalman 预警小箭头及 DEMA 圆点不触发订单。
- 指标在 15:45 ET 后因原有时段过滤停止产生有效箭头；15:50 ET 另有故障兜底清仓。

这里的“ITM/OTM $1”表示目标执行价相对现货约一美元。程序会按当日实际存在的执行价选择最接近且互不重复的三档。

## 先做 dry-run

确认 IB Gateway Paper 已启动、API 已启用，默认 paper 端口为 `4002`：

```bash
python -m gex_monitor.momentum_0dte_paper_trader \
  --host 127.0.0.1 \
  --port 4002 \
  --client-id 71
```

不带 `--execute-paper` 时不会提交订单，状态和日志文件也与实际 paper 执行分开。

## 在 Paper 账户执行

```bash
python -m gex_monitor.momentum_0dte_paper_trader \
  --host 127.0.0.1 \
  --port 4002 \
  --client-id 71 \
  --qty-per-leg 1 \
  --execute-paper
```

实际执行有不可绕过的保护：

- 只接受 IB Gateway paper 端口 `4002` 或 TWS paper 端口 `7497`。
- 必须只连接到一个以 `DU` 开头的 Paper 账户。
- 发现无法证明属于本策略的 QQQ/SPY 期权持仓时，阻止对应标的新开仓，不会擅自平掉该持仓。
- 开仓前要求三档合约都存在有效报价，默认买卖价差不得超过 30%。
- 使用 IOC 限价单；部分成交会被写入状态并继续管理，反转时不会在旧仓尚未平完的情况下开新仓。

成交记录写入 `logs/momentum_0dte_paper_trades_YYYYMMDD.csv`，持仓恢复状态写入
`logs/momentum_0dte_paper_state.json`。

## 每日复盘与优化

交易器会把每天完整的一分钟行情和信号状态保存到 `logs/momentum_0dte_bars/`。启动栈同时运行
优化守护进程，并在每个交易日 16:10 ET 自动完成：

- 按下一根 K 线开盘成交回放，避免使用未来数据。
- 使用最近 20 个交易日，较早日期训练、最近日期做样本外验证。
- 比较净点数、胜率、Profit Factor、最大回撤和正收益日比例。
- 联合读取现有转折点影子评分器，比较动量延续一致率、反转冲突率和事后方向正确率。
- 只有候选参数在验证集同时改善综合评分及净点数时，才标记为 `recommended`。
- 汇总当天真实 Paper 平仓损益。

输出位置：

- `logs/momentum_optimization/latest_review.md`：最新中文复盘报告。
- `logs/momentum_optimization/latest_recommendations.json`：下一交易日参数建议。
- 同目录带日期的文件：历史报告和建议快照。

默认只生成建议，不改变次日 live 参数。确认要自动应用通过验证的建议时，在 runtime `.env` 加入：

```bash
GEX_MOMENTUM_AUTO_APPLY_OPTIMIZATION=1
```

其他可调环境变量：

```bash
GEX_MOMENTUM_OPTIMIZER_RUN_TIME=16:10
GEX_MOMENTUM_OPTIMIZER_LOOKBACK_DAYS=20
GEX_MOMENTUM_OPTIMIZER_MINIMUM_DAYS=5
GEX_ENABLE_MOMENTUM_OPTIMIZER=1
```

当前搜索覆盖 DEMA 快慢周期、VWAP z-score 上下限和标准差窗口、Kalman 的 Q/R 噪声、
速度阈值、ATR 周期和阈值、投票门槛及开盘/收盘过滤时间。无效或危险参数会在交易器加载时再次被拒绝。
