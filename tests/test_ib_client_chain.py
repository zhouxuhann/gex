"""Tests for IB option-chain selection."""
from types import SimpleNamespace

from gex_monitor.ib_client import select_option_chain


def _chain(exchange: str, trading_class: str, strikes=None):
    return SimpleNamespace(
        exchange=exchange,
        tradingClass=trading_class,
        strikes=strikes or [],
    )


def test_select_option_chain_prefers_matching_smart_trading_class():
    chains = [
        _chain("SMART", "2SPY", [587.0, 609.0]),
        _chain("SMART", "SPY", [720.0, 721.0, 722.0]),
    ]

    selected = select_option_chain(chains, "SPY")

    assert selected.tradingClass == "SPY"
    assert selected.strikes == [720.0, 721.0, 722.0]


def test_select_option_chain_falls_back_to_matching_trading_class():
    chains = [
        _chain("SMART", "2SPY"),
        _chain("CBOE", "SPY", [720.0]),
    ]

    selected = select_option_chain(chains, "SPY")

    assert selected.exchange == "CBOE"
    assert selected.tradingClass == "SPY"
