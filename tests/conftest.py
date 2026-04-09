"""Pytest fixtures and configuration."""
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

ET = ZoneInfo('America/New_York')


@pytest.fixture
def temp_dir():
    """Provide a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_gex_df():
    """Sample GEX DataFrame for testing."""
    return pd.DataFrame([
        {'strike': 490, 'right': 'C', 'gamma': 0.05, 'oi': 1000, 'gex': 1e6, 'iv': 0.25},
        {'strike': 490, 'right': 'P', 'gamma': 0.05, 'oi': 800, 'gex': -0.8e6, 'iv': 0.26},
        {'strike': 495, 'right': 'C', 'gamma': 0.08, 'oi': 1500, 'gex': 1.5e6, 'iv': 0.22},
        {'strike': 495, 'right': 'P', 'gamma': 0.08, 'oi': 1200, 'gex': -1.2e6, 'iv': 0.23},
        {'strike': 500, 'right': 'C', 'gamma': 0.10, 'oi': 2000, 'gex': 2e6, 'iv': 0.20},
        {'strike': 500, 'right': 'P', 'gamma': 0.10, 'oi': 1800, 'gex': -1.8e6, 'iv': 0.21},
        {'strike': 505, 'right': 'C', 'gamma': 0.07, 'oi': 1200, 'gex': 1.2e6, 'iv': 0.22},
        {'strike': 505, 'right': 'P', 'gamma': 0.07, 'oi': 1400, 'gex': -1.4e6, 'iv': 0.23},
    ])


@pytest.fixture
def mock_ib_ticker():
    """Create a mock IB ticker for testing."""
    def _create_ticker(strike, right, gamma, oi, iv=0.25):
        ticker = MagicMock()
        ticker.contract.strike = strike
        ticker.contract.right = right
        ticker.contract.multiplier = '100'
        ticker.modelGreeks = MagicMock()
        ticker.modelGreeks.gamma = gamma
        ticker.modelGreeks.impliedVol = iv
        if right == 'C':
            ticker.callOpenInterest = oi
            ticker.putOpenInterest = 0
        else:
            ticker.callOpenInterest = 0
            ticker.putOpenInterest = oi
        return ticker
    return _create_ticker


@pytest.fixture
def sample_history_data():
    """Sample history data for StateManager testing."""
    now = datetime.now(ET)
    return [
        {
            'ts': now.replace(second=0, microsecond=0),
            'spot': 500.0,
            'total_gex': 1e6,
            'flip': 498.0,
            'call_gex': 2e6,
            'put_gex': -1e6,
            'atm_iv_pct': 22.5,
        },
        {
            'ts': now.replace(second=30, microsecond=0),
            'spot': 500.5,
            'total_gex': 1.1e6,
            'flip': 498.5,
            'call_gex': 2.1e6,
            'put_gex': -1e6,
            'atm_iv_pct': 22.3,
        },
    ]
