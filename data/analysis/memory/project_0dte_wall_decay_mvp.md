# 0DTE Gamma Wall Decay — MVP 设计

**日期**: 2026-04-22
**状态**: 设计阶段，未实施
**假设**: positive γ 日 dealers 多 gamma → price pins near max_pain → 两侧卖 0DTE premium 收 theta

---

## 为什么做这个

用户项目独有资源：实时 GEX（call_wall / put_wall / max_pain / positive_gamma / total_gex / atm_iv_pct）。

KDJ 5-min 方向预测策略零滑点仅 +2 bps/天，被交易成本吞没。**0DTE decay 不要求方向判断，只要求"不突破"**，天然与 pinning 的机制契合。

但必须承认：0DTE 卖方在 2024-2025 已高度 crowded，tail risk 显著。**这不是 "answer"，是"另一个值得试的方向"**，收益预期不应高于下面的 go/no-go 阈值。

---

## 策略逻辑

### 核心假设
1. `positive_gamma == True` → dealer market-maker 被动 hedge 使 price 回归 OI 密集区
2. `call_wall` 和 `put_wall` 之间 spot 运动受约束
3. 从 10:00 ET 入场到 15:30 ET 出场约 5.5 小时 theta，短期 IV 通常 high (>25% 年化对应日 decay 占权 ~0.8%)

### 触发条件 (全部需满足)

| 条件 | 阈值 | 数据源 |
|---|---|---|
| GEX 正 γ 且强 | `positive_gamma=True` AND `total_gex > 20B` | gex_snapshots |
| spot 居中 | `put_wall × 1.003 ≤ spot ≤ call_wall × 0.997` | 实时 + snapshot |
| 时间窗口 | 10:00 ≤ ET ≤ 13:30 | 本地时间 |
| IV 合理 | `15% ≤ atm_iv_pct ≤ 35%` | gex_snapshots |
| 距离合理 | `(call_wall - put_wall) / spot ≥ 0.4%` | snapshot |
| 无重大事件 | FOMC / CPI / NFP / earnings 当天跳过 | 手动黑名单 |
| VIX 正常 | VIX < P90 of trailing 60 day | yfinance |
| 已开仓 0 | 同日不重复 | 内部 state |

### 交易结构 (MVP)

**0DTE Iron Condor (defined risk)** — 推荐 MVP 从这个开始：

```
short call @ (call_wall - 1 strike)
long  call @ (call_wall + 1 strike)  ← defined risk upper
short put  @ (put_wall + 1 strike)
long  put  @ (put_wall - 1 strike)   ← defined risk lower
```

**为什么 Iron Condor 而不是 naked strangle**:
- naked strangle 需 cash margin，single-leg tail risk 无限
- defined risk = 已知最大亏损，可严格 position size
- paper 试验阶段优先风控

**为什么离 wall 1 strike 而非 ATM**:
- 越靠近 ATM premium 越大（payoff 看起来诱人）
- 但被动 hedge 只保护 wall 附近，ATM 是波动最大的地方
- 1 strike 外 = 留安全边际，接受较低 premium

### 仓位与风控

| 参数 | MVP 值 | 理由 |
|---|---|---|
| 单日最大 risk | 1% account | 极小 (paper 阶段) |
| 单次 risk 定义 | (long leg - short leg) × qty × 100 | IC 的 defined risk |
| 50% profit target | 达到即全平 | 经典 theta harvest |
| Hard stop loss | 单腿 touched (spot 碰 short strike) | 不等 assign |
| Delta hedge | 无 (MVP 不做) | 简化 |
| EOD 强平 | 15:45 ET | 防 assignment risk |

### 出场决策树

```
每 1 min 轮询:
  if spot <= short_put OR spot >= short_call:
      → HARD STOP (单腿 touched)
  elif current_pnl >= 0.5 × max_profit:
      → 50% target (全平)
  elif now >= 15:45 ET:
      → EOD 强平
  elif positive_gamma 翻成 False:
      → regime 变化 (全平，设 2-min stale tolerance)
  else:
      → hold
```

---

## Go/No-go 阈值

这是关键：**必须设前置标准，否则会掉"ddput 项目的坑"再来一次**（小样本过度解读）。

### MVP 存活条件 (30 trade 后评估)

| 指标 | 达标 | 不达标 |
|---|---|---|
| 胜率 | ≥ 55% | < 55% → 关闭 |
| 平均盈亏比 | profit × wins > loss × losses | 否则 → 关闭 |
| Max drawdown | ≤ 2× avg daily risk | 超过 → 立即停 |
| 单笔最大亏损 | ≤ 1.5× defined max loss | 超过 = 风控失败 → 立即停 |
| Regime gate 贡献 | 在 negative γ 日禁止进场后 PnL 显著改善 | 否则 regime gate 无用 |

**30 trade** 约需 60-90 个交易日积累（每天不一定符合进场条件）。这是 3-5 个月的耐心期。

### 早期 kill switch (前 10 笔就该停)

- 连续 3 笔 hit hard stop
- 单日 drawdown > 2% account
- 任何 bid-ask spread 在开仓时 > 15% premium (流动性太差)

---

## 实施阶段

### Phase 1: 手工 paper (1-2 周)
- 每天用 dashboard 人工看 GEX 状态
- 满足触发条件时**手动在 TWS 下 IC**
- Excel 记录：entry/exit price, PnL, 触发原因，出场原因
- 目标：**5-10 笔**，验证判断逻辑是否正确

### Phase 2: 半自动 (2-4 周)
- Python 脚本 poll 进场条件
- 满足时**发邮件/桌面通知** "建议开 IC: short_call=X short_put=Y"
- 人工复核后在 TWS 点击
- 出场规则脚本监控，邮件警报
- 目标：**10-20 笔**，验证自动化逻辑

### Phase 3: 全自动 (仅在 Phase 1+2 通过 Go 阈值后)
- ib_insync 自动下单
- 5-min 轮询 + 实时 GEX
- 完整 CSV + state persistence
- 目标：**持续 30+ 笔**，严格对照 go/no-go 阈值

**绝对不跳过 Phase 1**。手工阶段的作用是**发现所有你代码里没考虑到的边缘情况** (earnings 事件、divert 事件、strike 不齐、bid-ask 大、open interest 变化)。

---

## 可能的失败模式 (诚实清单)

| 模式 | 可能性 | 影响 | 应对 |
|---|---|---|---|
| Wall 被突破，dealer 止损追涨/杀跌 | 中 | 大亏 (single trade max loss) | IC defined risk |
| IV crush (event passes) 中途 | 中 | 中等亏损 | 50% target 早退 |
| Liquidity collapse (bid-ask 爆炸) | 中 | 出场价差 | Phase 1 手工时观察，不适合就跳过 |
| regime 误判 (positive γ 读反) | 低 | 中等亏损 | 2-min stale tolerance + regime change 强平 |
| Retail crowded (edge 已消失) | 高 | 持续 break-even | 30 笔后数据说话，不达标就关 |
| Black swan (Aug 2024 式暴动) | 低 | 灾难性亏损 | VIX 门槛 + hard stop + position size 1% |

---

## 与 KDJ 项目的关系

- KDJ 继续 paper 跑，**加 GEX regime gate 开关**，让 gate 假设 forward-only 验证
- 0DTE decay 作为 **独立轨道**，不替代 KDJ
- 资源配置: KDJ 工程投入冻结（除了 gate 实装），研究精力转 0DTE
- 如果 0DTE Phase 1 (5-10 手工笔) 显示 hit rate < 30%，快速放弃，不投更多工程

---

## 第一步要做的事

**立刻**: 无代码。用 dashboard 观察 10 个正 γ 日，记录：
- call_wall / put_wall / spot / atm_iv
- 日内 spot 最大偏离 wall 的距离
- 是否触发"单腿穿透"条件

**这 10 天的观察才是真正的 Phase 0** — 搞清楚规律真的存在吗，再谈代码。

---

## Open questions

1. **QQQ vs SPX**: SPX 0DTE premium 更大但 strike 间隔 $5 太粗，QQQ strike $0.5 / $1 更灵活但 premium 小。MVP 先选哪个？
2. **Weekly 0DTE 是否也算**: SPX/QQQ 周二/四也有 0DTE, 但 OI 不如 Mon/Wed/Fri 集中，gate 阈值可能要放宽
3. **Earnings blackout 范围**: 个股 earnings 通常 AH，但 QQQ/SPX 不直接 expose。要挂 blackout 吗？
4. **GEX data 5 分钟刷新 vs 实时**: 实时 spot 可能越过 wall 但 GEX 快照还没更新，会误判 regime. 要不要结合实时 spot 计算 spot 距 wall 比率作为次要 gate？

未回答。第一步观察后再说。
