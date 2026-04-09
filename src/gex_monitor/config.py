"""配置模块"""
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


class MonitoringConfig(BaseModel):
    """监控配置"""
    stale_seconds: int = 15
    spot_sanity_pct: float = 0.01


class AppConfig(BaseModel):
    """应用总配置"""
    ib: IBConfig = Field(default_factory=IBConfig)
    symbols: list[SymbolConfig] = Field(default_factory=list)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AppConfig":
        """从 YAML 文件加载配置"""
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        return cls(**data)

    @classmethod
    def default(cls) -> "AppConfig":
        """返回默认配置（单标的 QQQ）"""
        return cls(
            symbols=[
                SymbolConfig(name="QQQ", trading_class="QQQ"),
            ]
        )

    def get_enabled_symbols(self) -> list[SymbolConfig]:
        """获取所有启用的标的"""
        return [s for s in self.symbols if s.enabled]
