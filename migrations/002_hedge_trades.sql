-- 对冲交易记录表
-- 记录每次信号触发后的实际交易，用于验证信号有效性
--
-- 执行方式:
--   docker exec -i ibkr-market-data psql -U ibkr_user -d ibkr_market_data < migrations/002_hedge_trades.sql

CREATE TABLE IF NOT EXISTS hedge_trades (
    id BIGSERIAL PRIMARY KEY,

    -- 信号来源
    signal_ts TIMESTAMP NOT NULL,           -- 信号时间
    symbol VARCHAR(20) NOT NULL,
    action VARCHAR(20) NOT NULL,            -- HEDGE_NOW, HEDGE_SPREAD, REDUCE
    structure VARCHAR(20) NOT NULL,         -- outright_put, put_spread

    -- 入场
    entry_ts TIMESTAMP,                     -- 成交时间
    entry_spot DECIMAL(12, 4),              -- 入场时 spot
    entry_cost DECIMAL(12, 4),              -- 总成本 (debit)，正数

    -- 出场
    exit_ts TIMESTAMP,
    exit_spot DECIMAL(12, 4),
    exit_proceeds DECIMAL(12, 4),           -- 平仓收入，正数
    exit_reason VARCHAR(50),                -- signal_reduce, expiry, manual, stop_loss

    -- 合约腿
    legs JSONB NOT NULL DEFAULT '[]',
    -- 格式: [{"strike": 475, "right": "P", "expiry": "20260528",
    --         "side": "BUY", "qty": 1, "fill_price": 3.50}, ...]

    -- P&L
    realized_pnl DECIMAL(12, 4),            -- 已实现 P&L (exit_proceeds - entry_cost)
    max_pnl DECIMAL(12, 4),                 -- 持仓期间最大浮盈
    min_pnl DECIMAL(12, 4),                 -- 持仓期间最大浮亏

    -- 信号上下文（用于回测分析）
    urgency DECIMAL(4, 2),
    skew_cheapness DECIMAL(6, 2),
    gex_regime VARCHAR(20),
    term_structure VARCHAR(20),
    rr_25_at_entry DECIMAL(10, 6),

    -- 状态
    status VARCHAR(20) NOT NULL DEFAULT 'OPEN',  -- OPEN, CLOSED, EXPIRED
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_trades_symbol_status
    ON hedge_trades(symbol, status);

CREATE INDEX IF NOT EXISTS idx_trades_signal_ts
    ON hedge_trades(signal_ts DESC);

-- 信号验证视图: 每次信号 → 交易结果
CREATE OR REPLACE VIEW hedge_signal_validation AS
SELECT
    t.id as trade_id,
    t.signal_ts,
    t.symbol,
    t.action,
    t.structure,
    t.entry_cost,
    t.exit_proceeds,
    t.realized_pnl,
    t.status,
    t.gex_regime,
    t.skew_cheapness,
    t.term_structure,
    t.urgency,
    -- 持仓天数
    EXTRACT(EPOCH FROM (COALESCE(t.exit_ts, NOW()) - t.entry_ts)) / 86400.0
        AS holding_days,
    -- 收益率 (%)
    CASE WHEN t.entry_cost > 0
        THEN (t.realized_pnl / t.entry_cost) * 100
        ELSE NULL
    END AS return_pct
FROM hedge_trades t
ORDER BY t.signal_ts DESC;

-- 信号有效性汇总
CREATE OR REPLACE VIEW hedge_signal_stats AS
SELECT
    symbol,
    action,
    COUNT(*) as total_trades,
    COUNT(*) FILTER (WHERE status = 'CLOSED') as closed_trades,
    COUNT(*) FILTER (WHERE realized_pnl > 0) as winners,
    COUNT(*) FILTER (WHERE realized_pnl <= 0) as losers,
    ROUND(AVG(realized_pnl)::numeric, 2) as avg_pnl,
    ROUND(SUM(realized_pnl)::numeric, 2) as total_pnl,
    ROUND(AVG(CASE WHEN realized_pnl > 0 THEN realized_pnl END)::numeric, 2) as avg_win,
    ROUND(AVG(CASE WHEN realized_pnl <= 0 THEN realized_pnl END)::numeric, 2) as avg_loss,
    ROUND(
        COUNT(*) FILTER (WHERE realized_pnl > 0)::numeric /
        NULLIF(COUNT(*) FILTER (WHERE status = 'CLOSED'), 0) * 100, 1
    ) as win_rate_pct
FROM hedge_trades
GROUP BY symbol, action;
