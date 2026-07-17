"""配置模块"""
import os
from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import BaseModel, Field, model_validator


class IBConfig(BaseModel):
    """IB 连接配置"""
    host: str = "127.0.0.1"
    port: int = 4002
    client_id_base: int = 10
    connect_timeout: int = 20  # 连接超时秒数
    max_retries: int = 3  # 最大重试次数


class SymbolConfig(BaseModel):
    """单个标的配置"""
    name: str
    trading_class: str | None = None  # 默认等于 name
    strike_range: float = 0.04
    enabled: bool = True
    sec_type: Literal["STK", "IND"] = "STK"
    multiplier: int | None = None  # 覆盖默认乘数
    # Cboe GTH/Curb 延伸时段 (20:15-9:25 / 16:15-17:00 ET)。
    # 仅 SPX/XSP 等指数期权有此时段；股票期权必须保持 False，
    # 否则 worker 会在盘前空连 6 小时并每 3s 打无效数据警告。
    extended_hours: bool = False

    @model_validator(mode='after')
    def set_trading_class_default(self) -> Self:
        """trading_class 默认等于 name"""
        if self.trading_class is None:
            object.__setattr__(self, 'trading_class', self.name)
        return self


class StorageConfig(BaseModel):
    """存储配置"""
    data_dir: str = "./data"
    max_history: int = 8000


class ServerConfig(BaseModel):
    """服务器配置"""
    host: str = "0.0.0.0"
    port: int = 8050


class TimingConfig(BaseModel):
    """时间间隔配置"""
    tick_interval_sec: int = 3           # 主循环间隔
    persist_interval_sec: int = 60       # 持久化间隔
    reconnect_delay_sec: int = 10        # 重连等待
    market_closed_check_sec: int = 300   # 非交易时段检查间隔
    max_sleep_sec: int = 1800            # 最长休眠时间


class DatabaseConfig(BaseModel):
    """PostgreSQL 数据库配置（复用 ibkr-data-store）"""
    enabled: bool = True
    host: str = "127.0.0.1"
    port: int = 5433
    dbname: str = "ibkr_market_data"
    user: str = "ibkr_user"
    password: str = ""

    @model_validator(mode='after')
    def load_password_from_env(self) -> Self:
        """密码优先从环境变量读取"""
        if not self.password:
            object.__setattr__(
                self, 'password',
                os.environ.get('POSTGRES_PASSWORD', 'ibkr_secure_password_2026')
            )
        return self


class MonitoringConfig(BaseModel):
    """监控配置"""
    stale_seconds: int = 15
    reconnect_stale_seconds: int = 60
    spot_sanity_pct: float = 0.01
    quality_min_contracts: int = 20
    quality_max_missing_ratio: float = 0.25
    quality_min_rth_coverage: float = 0.95
    quality_max_gap_seconds: int = 60
    quality_max_derived_null_ratio: float = 0.10


class IntradayVRPConfig(BaseModel):
    """0DTE 日内波动率风险溢价观测器（只采集，不交易）。"""
    enabled: bool = False
    observation_only: bool = True
    symbols: list[str] = Field(default_factory=lambda: ["QQQ"])
    schedule_et: list[str] = Field(default_factory=lambda: [
        "09:35", "10:00", "10:30", "11:00", "12:00",
        "13:00", "14:00", "14:30", "15:00",
    ])
    sample_window_seconds: int = 180
    candidate_strikes_each_side: int = 2
    wing_strikes_each_side: int = 5
    iron_fly_widths: list[float] = Field(default_factory=lambda: [1, 2, 3, 5])
    max_quote_age_seconds: int = 10
    max_combined_spread_ratio: float = 0.10
    commission_per_straddle: float = 1.30


class EmailAlertConfig(BaseModel):
    """通用邮件告警配置"""
    enabled: bool = False
    sender: str = Field(
        default_factory=lambda: os.environ.get("EMAIL_SENDER", "fzhouxu615@gmail.com")
    )
    password_env: str = Field(
        default_factory=lambda: os.environ.get("EMAIL_PASSWORD_ENV", "GMAIL_APP_PASSWORD")
    )
    recipients: list[str] = Field(
        default_factory=lambda: [
            x.strip()
            for x in os.environ.get("EMAIL_RECIPIENTS", "wenyi.hann@gmail.com").split(",")
            if x.strip()
        ]
    )
    smtp_host: str = Field(
        default_factory=lambda: os.environ.get("EMAIL_SMTP_HOST", "smtp.gmail.com")
    )
    smtp_port: int = Field(default_factory=lambda: int(os.environ.get("EMAIL_SMTP_PORT", "465")))
    cooldown_sec: int = 600
    subject_prefix: str = "[GEX-IB]"


class IBErrorAlertConfig(BaseModel):
    """IB 系统错误告警"""
    email: EmailAlertConfig = Field(default_factory=EmailAlertConfig)


class AlertsConfig(BaseModel):
    """系统告警配置"""
    ib_errors: IBErrorAlertConfig = Field(default_factory=IBErrorAlertConfig)


class AppConfig(BaseModel):
    """应用总配置"""
    ib: IBConfig = Field(default_factory=IBConfig)
    symbols: list[SymbolConfig] = Field(default_factory=list)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    intraday_vrp: IntradayVRPConfig = Field(default_factory=IntradayVRPConfig)
    timing: TimingConfig = Field(default_factory=TimingConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    alerts: AlertsConfig = Field(default_factory=AlertsConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AppConfig":
        """从 YAML 文件加载配置，并稳定解析相对数据目录。

        相对路径优先以包含 ``pyproject.toml`` 的项目根目录为基准；若配置
        文件不在项目内，则以配置文件所在目录为基准。这样服务从不同 cwd
        启动时不会分别写入 ``data`` 和 ``src/data``。
        """
        config_path = Path(path).expanduser().resolve()
        with open(config_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        config = cls(**(data or {}))
        config.resolve_storage_path(config_path)
        return config

    @classmethod
    def default(cls) -> "AppConfig":
        """返回默认配置（单标的 QQQ）"""
        config = cls(
            symbols=[
                SymbolConfig(name="QQQ", trading_class="QQQ"),
            ]
        )
        config.resolve_storage_path()
        return config

    def resolve_storage_path(self, config_path: str | Path | None = None) -> Path:
        """把 storage.data_dir 解析成绝对路径并写回配置。"""
        raw = Path(self.storage.data_dir).expanduser()
        if raw.is_absolute():
            resolved = raw.resolve()
        else:
            base = Path(__file__).resolve().parents[2]
            if config_path is not None:
                path = Path(config_path).expanduser().resolve()
                base = path.parent
                for candidate in (path.parent, *path.parents):
                    if (candidate / 'pyproject.toml').is_file():
                        base = candidate
                        break
            resolved = (base / raw).resolve()
        self.storage.data_dir = str(resolved)
        return resolved

    def get_enabled_symbols(self) -> list[SymbolConfig]:
        """获取所有启用的标的"""
        return [s for s in self.symbols if s.enabled]
