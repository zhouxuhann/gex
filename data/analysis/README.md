# GEX 数据分析工具

本目录有两个 CLI 工具，覆盖**数据健康检查、特征探索、信号分析、可视化、长期观察**全套。

| 脚本 | 用途 |
|---|---|
| [`gex_tools.py`](#gex_toolspy--12-个子命令) | 全功能数据分析 (12 个子命令) |
| [`gamma_band_observer.py`](#gamma_band_observerpy--upside--downside-band-观察) | 长期观察上方/下方 γ band, 每日累积数据 |

---

## 通用约定

所有 `gex_tools.py` 子命令统一接受 `<symbol> [<day>...]`：

| 参数 | 含义 | 例子 |
|---|---|---|
| `symbol` | QQQ / SPX / SPY / ... | `QQQ` |
| `day` | YYYYMMDD | `20260417` |
| `latest` | 最新一天 | `latest` |
| `all` 或省略 | 所有可用数据 | (不写) |
| `-N` | N 天前 | `-1` 昨天, `-7` 一周前 |

**多个 day 可以连写**：`quality QQQ 20260417 20260418 latest`

数据源：`src/data/gex_{SYMBOL}_{YYYYMMDD}.parquet` 主表 + `strikes_*.parquet` + `ohlc_*.parquet`

---

## `gex_tools.py` — 12 个子命令

### 🟢 日常用 (高频)

#### 1. `quality` — **数据质量检查 (最常用)**

```bash
gex_tools.py quality QQQ latest
gex_tools.py quality QQQ 20260417 20260418
gex_tools.py quality QQQ                   # 所有日期
```

输出每天的：
- 行数、时间起止
- 采样间隔（中位/p95/max）+ >30 秒 gap 列表
- spot 区间
- total_gex 均值/标准差
- +γ 占比 + γ 切换次数
- 各字段 NaN 率（flip / call_wall / put_wall / atm_iv 等）
- strike 覆盖（上下方百分比）
- ohlc 涨跌 + 当日振幅

**典型场景**：每天收盘后跑一次确认数据完整、无大 gap、字段无异常 NaN

---

#### 2. `list` — 列出所有可用数据

```bash
gex_tools.py list
```

显示 `src/data/` 下每个 symbol 都有哪些日期的 parquet。

---

#### 3. `compare` — 跨日对比总表

```bash
gex_tools.py compare QQQ
gex_tools.py compare QQQ 20260420 20260424
```

一张表横向对比多日的 day_ret / total_gex 均值 / +γ 占比 / 信号触发数。**找规律首选**。

---

### 🟡 信号 / 特征研究 (中频)

#### 4. `signal` — 信号 → 未来收益相关性

```bash
gex_tools.py signal QQQ                              # 默认 ddput, 15min horizon, 5 bins
gex_tools.py signal QQQ --feature ddgex --horizon 30
gex_tools.py signal QQQ --zscore                     # 按日 z-score 归一化再分箱
```

把指定 feature 分成 N 个 bin（按值大小）, 计算每个 bin 的**未来 N 分钟收益均值**和胜率。Q5-Q1 差分 = 信号 edge。

**参数**：
- `--feature ddput` (默认) / ddcall / ddgex / dput / dcall / dgex
- `--horizon 15` (默认, 分钟)
- `--bins 5`
- `--zscore` 按日内 z-score 归一化再分箱

---

#### 5. `thresholds` — 特征分位阈值

```bash
gex_tools.py thresholds QQQ --feature ddput
```

打印 ddput / ddgex / 等特征的 P5 / P25 / P50 / P75 / P95 阈值，用于设 alert 触发线。

---

#### 6. `events` — 极值事件 top/bottom N

```bash
gex_tools.py events QQQ 20260417 --top 10
gex_tools.py events QQQ --feature ddgex --top 20
```

列出指定特征的 top N 和 bottom N 时刻（带时间、值、对应 spot），方便回看图找规律。

---

#### 7. `signals` — 信号触发时刻列表 (K 线对照)

```bash
gex_tools.py signals QQQ 20260421 --feature ddput --threshold 1.5
gex_tools.py signals QQQ --top-pct 5 --only-long --csv
gex_tools.py signals QQQ --live                         # 用 signals_live_*.parquet
```

输出每个触发的 ts / spot / feature value，可以**直接复制到 TradingView 标注 K 线**。

**参数**：
- `--threshold 1.5` 绝对值阈值
- `--top-pct 5` 取前 5% 极值
- `--only-long` / `--only-short` 仅一侧
- `--exclude-eod` / `--skip-open` 时段过滤
- `--csv` 输出 CSV 而非表格
- `--live` 读 `signals_live_*.parquet` (实盘 ddput 触发记录)

---

### 🔵 深度分析 (低频)

#### 8. `corr` — 全特征 × 全 horizon 相关矩阵

```bash
gex_tools.py corr QQQ
gex_tools.py corr QQQ --features ddput,ddgex,dgex --horizons 5,15,30
```

打印一个矩阵，**横**向是各 horizon (5/10/15/30/60min forward return)，**纵**向是特征。

---

#### 9. `regression` — OLS + partial R²

```bash
gex_tools.py regression QQQ
gex_tools.py regression QQQ --features total_sm,dgex,ddgex --horizons 5,15
```

用 OLS 拟合 fwd_return = a×feature1 + b×feature2 + ..., 输出系数 + p 值 + partial R²（每个特征的独立贡献）。

---

#### 10. `coverage` — strike 覆盖诊断

```bash
gex_tools.py coverage QQQ 20260417
gex_tools.py coverage QQQ -v                          # verbose
```

每天的 strike range 是不是足够覆盖 spot 上下 4%, 有没有"缺角"导致 GEX 计算偏差。

---

#### 11. `plot` — 3 列标准图

```bash
gex_tools.py plot QQQ 20260421
gex_tools.py plot QQQ --feature ddput --smooth 5
```

生成 PNG, 三列：
1. spot + total_gex
2. dGEX (一阶导)
3. 第三个特征 (默认 ddgex, 可换 ddput)

输出到 `figs/` 目录。

---

#### 12. `daily_fingerprint` — 日度指纹追加

```bash
gex_tools.py daily_fingerprint QQQ                    # 增量 merge
gex_tools.py daily_fingerprint QQQ --overwrite        # 重写
```

把每天的 25 个核心指标（total_gex 均值、+γ 占比、波动、最大/最小、各种 z 等）凝练成**一行**, append 到 `daily_fingerprint_QQQ.parquet`。**长期跨周/跨月分析的基础数据**。

---

## `gamma_band_observer.py` — Upside / Downside band 观察

### 用途

把 `strikes_*.parquet` 的逐 strike GEX 按 spot 相对 band 重切片：

| Band | 含义 |
|---|---|
| `up_50bps`  | spot 上方 0.5% 区间净 GEX |
| `up_100bps` | spot 上方 1% |
| `up_200bps` | spot 上方 2% |
| `up_300bps` | spot 上方 3% |
| `dn_*`      | 下方 (镜像) |
| `up_iv_*` / `dn_iv_*` | 该 band 内平均 IV (剔除 >200% 噪声) |

**判读**：
- `up_100bps < 0` → upside short γ → squeeze fuel
- `up_100bps > 0` → upside long γ → 阻尼上涨 (pinning)
- IV 列用于解构 "γ weaken 是结构变化 vs IV 涨"

### 用法

```bash
# 一次性 backfill 历史
python3 gamma_band_observer.py --backfill 20260414 20260429 --symbol QQQ
python3 gamma_band_observer.py --backfill 20260417 20260429 --symbol SPX

# 指定一天
python3 gamma_band_observer.py --date 20260424 --symbol QQQ

# 今天 (ET)
python3 gamma_band_observer.py --today --symbol QQQ
```

输出: `data/gamma_bands/gamma_bands_{symbol}_{YYYYMMDD}.parquet`

### Cron 自动追加

每天 ET 16:30 (= JST 05:30) 收盘后自动跑：

```bash
# crontab -e (假设系统 JST 时区)
30 5 * * 2-6 cd /Users/fanzhouxu/Downloads/gex/data/analysis && /opt/homebrew/anaconda3/bin/python3 gamma_band_observer.py --today --symbol QQQ >> /Users/fanzhouxu/Downloads/gex/logs/gamma_bands.log 2>&1
35 5 * * 2-6 cd /Users/fanzhouxu/Downloads/gex/data/analysis && /opt/homebrew/anaconda3/bin/python3 gamma_band_observer.py --today --symbol SPX >> /Users/fanzhouxu/Downloads/gex/logs/gamma_bands.log 2>&1
```

`* * 2-6` = JST 周二到周六 (= ET 周一到周五收盘后)。

### 累积分析示例

```python
import pandas as pd, glob
files = sorted(glob.glob('/Users/fanzhouxu/Downloads/gex/data/gamma_bands/gamma_bands_QQQ_*.parquet'))
all_df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
print(f"{len(all_df)} rows, {len(files)} days")
# 按日 summary
for d, day in all_df.groupby(all_df['ts'].dt.date):
    pct_neg = (day['up_100bps'] < 0).mean() * 100
    print(f"{d}: upside avg γ {day.up_100bps.mean():+.2e}, %_neg={pct_neg:.0f}%")
```

---

## 工作流推荐

### 每天收盘后 (5 分钟)

```bash
# 1. 数据完整性
python3 gex_tools.py quality QQQ latest
python3 gex_tools.py quality SPX latest

# 2. 当日指纹追加
python3 gex_tools.py daily_fingerprint QQQ
python3 gex_tools.py daily_fingerprint SPX

# 3. gamma band 追加 (cron 已设的话自动)
python3 gamma_band_observer.py --today --symbol QQQ
python3 gamma_band_observer.py --today --symbol SPX
```

### 每周回顾 (15 分钟)

```bash
# 跨日对比看趋势
python3 gex_tools.py compare QQQ

# 信号 edge 重新评估
python3 gex_tools.py signal QQQ --feature ddput --horizon 15

# 特征间相关性
python3 gex_tools.py corr QQQ --horizons 15,30
```

### 临时排查 (5 分钟)

```bash
# 某天数据可疑?
python3 gex_tools.py quality QQQ 20260423        # 04-23 那天数据缺失分析
python3 gex_tools.py coverage QQQ 20260423 -v    # strike 覆盖看有没有 0DTE 不全

# 某个信号触发时市场反应?
python3 gex_tools.py signals QQQ 20260421 --feature ddput --threshold 2.5
python3 gex_tools.py plot QQQ 20260421           # 出图直观看
```

---

## 常用特征名速查

| 特征 | 含义 | 单位 |
|---|---|---|
| `total_gex` | 总 dealer γ exposure | $/% spot move |
| `flip` | gamma flip level | strike $ |
| `call_wall` / `put_wall` | call/put OI 最大堆积 strike | $ |
| `atm_iv_pct` | ATM IV (年化) | % |
| `dgex` | total_gex 一阶导 | per-tick |
| `ddgex` | total_gex 二阶导 | per-tick² |
| `dput` | put_gex 一阶导 | |
| `ddput` | put_gex 二阶导 (**已 closed-failed 信号**) | |
| `total_sm` | total_gex 平滑 | |

---

## 输出文件位置

| 文件 | 内容 |
|---|---|
| `figs/*.png` | `plot` 子命令出的图 |
| `daily_fingerprint_{SYM}.parquet` | 每日指纹 |
| `signals_{SYM}_ddput.csv` | `signals --csv` 输出 |
| `data/gamma_bands/gamma_bands_*.parquet` | gamma_band_observer 输出 |
| `per_day_correlations.csv` | `corr` 子命令的扩展输出 |
| `daily_summary.csv` | `compare` 的扩展 |

---

## 补充

**没在这里的工具** (在 `src/gex_monitor/` 下，运行时使用):
- `kdj_backtest.py` — KDJ 策略回测引擎
- `gex_regime_reader.py` / `vix_regime_reader.py` — 信号 gate
- `ib_error_watcher.py` — IB 关键 error 邮件报警

**相关 memory** (在 `data/analysis/memory/`):
- `project_ddput_signal_hypothesis.md` — ddput 项目状态 (closed-failed)
- `project_0dte_wall_decay_mvp.md` — 0DTE 设计文档
- `feedback_*.md` — 各种验证教训
