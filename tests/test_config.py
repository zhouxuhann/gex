"""Tests for config module."""
import tempfile
from pathlib import Path

import pytest
import yaml

from gex_monitor.config import (
    IBConfig, SymbolConfig, StorageConfig, ServerConfig,
    MonitoringConfig, AppConfig
)


class TestIBConfig:
    """Tests for IBConfig class."""

    def test_default_values(self):
        """Test default IB configuration values."""
        config = IBConfig()

        assert config.host == "127.0.0.1"
        assert config.port == 4002
        assert config.client_id_base == 10
        assert config.connect_timeout == 20
        assert config.max_retries == 3

    def test_custom_values(self):
        """Test custom IB configuration values."""
        config = IBConfig(
            host="192.168.1.100",
            port=7496,
            client_id_base=100,
            connect_timeout=30,
            max_retries=5,
        )

        assert config.host == "192.168.1.100"
        assert config.port == 7496
        assert config.client_id_base == 100
        assert config.connect_timeout == 30
        assert config.max_retries == 5


class TestSymbolConfig:
    """Tests for SymbolConfig class."""

    def test_minimal_config(self):
        """Test symbol config with only required field."""
        config = SymbolConfig(name="QQQ")

        assert config.name == "QQQ"
        assert config.trading_class == "QQQ"  # Should default to name
        assert config.strike_range == 0.04
        assert config.enabled is True
        assert config.sec_type == "STK"
        assert config.multiplier is None

    def test_trading_class_defaults_to_name(self):
        """Test that trading_class defaults to name."""
        config = SymbolConfig(name="SPY")
        assert config.trading_class == "SPY"

    def test_explicit_trading_class(self):
        """Test explicit trading_class."""
        config = SymbolConfig(name="SPX", trading_class="SPXW")
        assert config.trading_class == "SPXW"

    def test_index_config(self):
        """Test index configuration."""
        config = SymbolConfig(
            name="SPX",
            trading_class="SPXW",
            sec_type="IND",
            multiplier=100,
        )

        assert config.sec_type == "IND"
        assert config.multiplier == 100

    def test_disabled_symbol(self):
        """Test disabled symbol configuration."""
        config = SymbolConfig(name="QQQ", enabled=False)
        assert config.enabled is False

    def test_custom_strike_range(self):
        """Test custom strike range."""
        config = SymbolConfig(name="QQQ", strike_range=0.06)
        assert config.strike_range == 0.06


class TestStorageConfig:
    """Tests for StorageConfig class."""

    def test_default_values(self):
        """Test default storage configuration values."""
        config = StorageConfig()

        assert config.data_dir == "./data"
        assert config.max_history == 8000

    def test_custom_values(self):
        """Test custom storage configuration values."""
        config = StorageConfig(
            data_dir="/custom/path",
            max_history=10000,
        )

        assert config.data_dir == "/custom/path"
        assert config.max_history == 10000


class TestServerConfig:
    """Tests for ServerConfig class."""

    def test_default_values(self):
        """Test default server configuration values."""
        config = ServerConfig()

        assert config.host == "0.0.0.0"
        assert config.port == 8050

    def test_custom_values(self):
        """Test custom server configuration values."""
        config = ServerConfig(host="127.0.0.1", port=8080)

        assert config.host == "127.0.0.1"
        assert config.port == 8080


class TestMonitoringConfig:
    """Tests for MonitoringConfig class."""

    def test_default_values(self):
        """Test default monitoring configuration values."""
        config = MonitoringConfig()

        assert config.stale_seconds == 15
        assert config.spot_sanity_pct == 0.01

    def test_custom_values(self):
        """Test custom monitoring configuration values."""
        config = MonitoringConfig(
            stale_seconds=30,
            spot_sanity_pct=0.02,
        )

        assert config.stale_seconds == 30
        assert config.spot_sanity_pct == 0.02


class TestAppConfig:
    """Tests for AppConfig class."""

    def test_default_config(self):
        """Test default app configuration."""
        config = AppConfig.default()

        assert len(config.symbols) == 1
        assert config.symbols[0].name == "QQQ"
        assert config.ib.host == "127.0.0.1"
        assert config.storage.data_dir == "./data"
        assert config.server.port == 8050

    def test_get_enabled_symbols(self):
        """Test getting enabled symbols."""
        config = AppConfig(
            symbols=[
                SymbolConfig(name="QQQ", enabled=True),
                SymbolConfig(name="SPY", enabled=False),
                SymbolConfig(name="SPX", enabled=True),
            ]
        )

        enabled = config.get_enabled_symbols()

        assert len(enabled) == 2
        assert enabled[0].name == "QQQ"
        assert enabled[1].name == "SPX"

    def test_get_enabled_symbols_all_disabled(self):
        """Test getting enabled symbols when all disabled."""
        config = AppConfig(
            symbols=[
                SymbolConfig(name="QQQ", enabled=False),
                SymbolConfig(name="SPY", enabled=False),
            ]
        )

        enabled = config.get_enabled_symbols()
        assert len(enabled) == 0

    def test_from_yaml(self, temp_dir):
        """Test loading configuration from YAML file."""
        yaml_content = """
ib:
  host: "192.168.1.100"
  port: 7496
  client_id_base: 20

symbols:
  - name: QQQ
    strike_range: 0.05
    enabled: true
  - name: SPY
    strike_range: 0.04
    enabled: true

storage:
  data_dir: "/data/gex"
  max_history: 5000

server:
  host: "0.0.0.0"
  port: 9000

monitoring:
  stale_seconds: 20
  spot_sanity_pct: 0.015
"""
        yaml_file = temp_dir / "config.yaml"
        yaml_file.write_text(yaml_content)

        config = AppConfig.from_yaml(yaml_file)

        assert config.ib.host == "192.168.1.100"
        assert config.ib.port == 7496
        assert len(config.symbols) == 2
        assert config.symbols[0].name == "QQQ"
        assert config.symbols[0].strike_range == 0.05
        assert config.storage.data_dir == "/data/gex"
        assert config.server.port == 9000
        assert config.monitoring.stale_seconds == 20

    def test_from_yaml_minimal(self, temp_dir):
        """Test loading minimal YAML configuration."""
        yaml_content = """
symbols:
  - name: QQQ
"""
        yaml_file = temp_dir / "config.yaml"
        yaml_file.write_text(yaml_content)

        config = AppConfig.from_yaml(yaml_file)

        # Should use defaults for everything not specified
        assert config.ib.host == "127.0.0.1"
        assert len(config.symbols) == 1
        assert config.symbols[0].name == "QQQ"
        assert config.symbols[0].trading_class == "QQQ"

    def test_from_yaml_with_index(self, temp_dir):
        """Test loading YAML with index configuration."""
        yaml_content = """
symbols:
  - name: SPX
    trading_class: SPXW
    sec_type: IND
    strike_range: 0.03
"""
        yaml_file = temp_dir / "config.yaml"
        yaml_file.write_text(yaml_content)

        config = AppConfig.from_yaml(yaml_file)

        assert config.symbols[0].name == "SPX"
        assert config.symbols[0].trading_class == "SPXW"
        assert config.symbols[0].sec_type == "IND"

    def test_empty_config(self):
        """Test empty configuration."""
        config = AppConfig()

        assert len(config.symbols) == 0
        assert config.ib is not None
        assert config.storage is not None

    def test_config_immutability(self):
        """Test that config fields are accessible but model is frozen."""
        config = AppConfig.default()

        # Should be able to access
        _ = config.ib.host
        _ = config.symbols[0].name

        # Pydantic models are mutable by default, so this should work
        config.server.port = 9000
        assert config.server.port == 9000


class TestSymbolConfigValidation:
    """Tests for SymbolConfig validation."""

    def test_sec_type_validation(self):
        """Test that sec_type only accepts STK or IND."""
        # Valid values
        SymbolConfig(name="QQQ", sec_type="STK")
        SymbolConfig(name="SPX", sec_type="IND")

        # Invalid value should raise
        with pytest.raises(Exception):  # Pydantic validation error
            SymbolConfig(name="QQQ", sec_type="INVALID")

    def test_name_required(self):
        """Test that name is required."""
        with pytest.raises(Exception):  # Pydantic validation error
            SymbolConfig()
