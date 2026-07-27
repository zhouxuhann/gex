# GEX Monitor v2.0

QQQ/SPY 0DTE Gamma Exposure 实时监控系统

## 功能

- **实时 GEX 监控**: 连接 IB Gateway，计算 dealer gamma exposure
- **多标的支持**: QQQ、SPY、SPX 等可配置
- **历史回放**: 收盘后拖动时间轴复盘任意时刻的 GEX 分布
- **分段标注**: 在 K 线上框选区间，标注市场 regime (trend/chop)
- **容器化部署**: 支持 Docker Compose

## 快速开始

```bash
# 安装
pip install -e .

# 验证安装
python test_setup.py

# 启动（需要 IB Gateway 运行在 7496 端口）
python -m gex_monitor.main

# 或指定配置
python -m gex_monitor.main --config config/config.yaml
```

访问 http://localhost:8050

## Docker 运行

```bash
docker compose up -d
```

## 项目结构

```
gex/
├── config/
│   ├── config.yaml           # 本地配置
│   └── config.docker.yaml    # Docker 配置
├── src/gex_monitor/
│   ├── main.py               # 入口
│   ├── config.py             # Pydantic 配置
│   ├── gex_calc.py           # GEX 计算（纯函数）
│   ├── ib_client.py          # IB 数据采集 Worker
│   ├── state.py              # 状态管理 (StateManager + Registry)
│   ├── storage.py            # Parquet 存储
│   ├── time_utils.py         # 时区、交易日历
│   └── ui/
│       ├── app.py            # Dash app 工厂
│       ├── callbacks.py      # 回调逻辑
│       └── layout.py         # 布局定义
├── data/                     # 数据目录（自动创建）
├── Dockerfile
├── docker-compose.yaml
├── pyproject.toml
└── test_setup.py             # 安装验证脚本
```

## 配置说明

编辑 `config/config.yaml`:

```yaml
ib:
  host: "127.0.0.1"           # Docker 内用 host.docker.internal
  port: 7496                   # Live 端口 (Paper: 7497)
  client_id_base: 10

symbols:
  - name: QQQ
    trading_class: QQQ
    strike_range: 0.04
    enabled: true
  - name: SPY
    enabled: true             # 启用 SPY
  - name: SPX
    trading_class: SPXW
    sec_type: IND
    enabled: false            # SPX 默认关闭

storage:
  data_dir: "./data"
  max_history: 8000

server:
  host: "0.0.0.0"
  port: 8050
```

## 数据存储

```
data/
├── gex_QQQ_20240115.parquet      # 聚合 GEX（每 3s）
├── ohlc_QQQ_20240115.parquet     # 1 分钟 OHLC
├── strikes_QQQ_20240115.parquet  # strike-level（每分钟，用于回放）
└── segments.parquet              # 分段标注
```

## GEX 计算公式

```
gex = sign × gamma × OI × multiplier × spot² × 0.01
```

- sign: +1 (call), -1 (put)
- 假设: dealers short puts, long calls
- 单位: 美元 per 1% spot 变动
- 注意: OI 是 T-1 数据，盘中不更新
- `total_gex/call_gex/put_gex/gex` 始终使用 OI；当日累计成交量另存为
  `volume_gamma`，只表示活动度，不代表未平仓 dealer 仓位
- Gamma Flip 使用固定各合约 IV 的 Black-Scholes 重新定价网格求根；没有
  有效零点时返回空值，不使用 strike 累积值或窗口边界伪造结果

### 日内模型使用约定

- `gross_gex = |call_gex| + |put_gex|`，`net_gex_ratio = total_gex / gross_gex`；
  GEX 的 5/15 分钟变化统一用 gross GEX 归一化，避免净 GEX 接近零时比率失真。
- `gross_volume_gamma / gross_gex` 仅作为当日成交活跃度确认项，不替代 OI 仓位结构。
- VRP 会记录 `long_gamma / neutral_gamma / short_gamma` 和策略偏好，但当前
  `gex_hard_gate_enabled=false`，不会仅凭 GEX 阻止 paper order。
- 影子评分仅接受 `gex_method=oi_position_v2`。旧混合 OI/Volume 阈值已经停用，
  当前规则由 QQQ 66 日训练/验证/测试并用 SPY 48 日做外部验证。

历史 OI 口径回算（不会覆盖默认研究目录）：

```bash
python scripts/build_intraday_turning_points.py --symbol QQQ \
  --gex-method oi_position_v2 --output-dir data/analysis/gex_v2
python scripts/build_intraday_turning_points.py --symbol SPY \
  --gex-method oi_position_v2 --output-dir data/analysis/gex_v2
python scripts/analyze_intraday_turning_points.py \
  --input-dir data/analysis/gex_v2 --output-dir data/analysis/gex_v2
```

旧 strike 文件没有 volume，因此历史回算只重建 OI position 通道；Volume 通道从
新版本上线后前向积累。

## API 端点

- `GET /health` - 健康检查，返回各标的连接状态

## 开发历史

- **v1.0**: 单文件版本 (gex_monitor_v2.py)
- **v2.0**: 模块化重构
  - 拆分为独立模块
  - Pydantic 配置
  - 多标的并行支持
  - 历史回放功能
  - Docker 支持

## 依赖

- Python 3.11+
- IB Gateway / TWS
- 主要库: ib_insync, dash, pandas, pydantic
