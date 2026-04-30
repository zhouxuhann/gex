-- GEX Snapshots 表
-- 存储聚合 GEX 指标，与 market_data_bars 同库，方便 JOIN 回测
--
-- 执行方式:
--   psql -h localhost -p 5433 -U ibkr_user -d ibkr_market_data -f migrations/001_gex_snapshots.sql

CREATE TABLE IF NOT EXISTS gex_snapshots (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    datetime TIMESTAMP NOT NULL,

    -- 价格
    spot DECIMAL(12, 4) NOT NULL,

    -- GEX 核心指标
    total_gex DECIMAL(18, 4),
    call_gex DECIMAL(18, 4),
    put_gex DECIMAL(18, 4),
    gamma_flip DECIMAL(12, 4),

    -- 关键价位
    call_wall DECIMAL(12, 4),
    put_wall DECIMAL(12, 4),
    max_pain DECIMAL(12, 4),

    -- 波动率
    atm_iv_pct DECIMAL(8, 4),
    positive_gamma BOOLEAN DEFAULT FALSE,

    -- Regime 分类
    regime_code VARCHAR(100),

    -- Skew 指标
    rr_25 DECIMAL(10, 6),
    skew_slope DECIMAL(10, 6),
    rr_25_zscore DECIMAL(8, 4),
    skew_signal VARCHAR(50),

    -- 元数据
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- 去重：同一标的同一时刻只有一条
    UNIQUE(symbol, datetime)
);

-- 索引
CREATE INDEX IF NOT EXISTS idx_gex_symbol_time
    ON gex_snapshots(symbol, datetime DESC);

CREATE INDEX IF NOT EXISTS idx_gex_symbol_regime
    ON gex_snapshots(symbol, regime_code);

CREATE INDEX IF NOT EXISTS idx_gex_lookup
    ON gex_snapshots(symbol, datetime DESC)
    INCLUDE (spot, total_gex, gamma_flip, call_wall, put_wall, rr_25);

-- Upsert 函数
CREATE OR REPLACE FUNCTION upsert_gex_snapshot(
    p_symbol VARCHAR(20),
    p_datetime TIMESTAMP,
    p_spot DECIMAL(12, 4),
    p_total_gex DECIMAL(18, 4),
    p_call_gex DECIMAL(18, 4),
    p_put_gex DECIMAL(18, 4),
    p_gamma_flip DECIMAL(12, 4),
    p_call_wall DECIMAL(12, 4),
    p_put_wall DECIMAL(12, 4),
    p_max_pain DECIMAL(12, 4),
    p_atm_iv_pct DECIMAL(8, 4),
    p_positive_gamma BOOLEAN,
    p_regime_code VARCHAR(100),
    p_rr_25 DECIMAL(10, 6),
    p_skew_slope DECIMAL(10, 6),
    p_rr_25_zscore DECIMAL(8, 4),
    p_skew_signal VARCHAR(50)
) RETURNS TEXT AS $$
DECLARE
    result TEXT;
BEGIN
    INSERT INTO gex_snapshots (
        symbol, datetime, spot,
        total_gex, call_gex, put_gex, gamma_flip,
        call_wall, put_wall, max_pain,
        atm_iv_pct, positive_gamma, regime_code,
        rr_25, skew_slope, rr_25_zscore, skew_signal
    ) VALUES (
        p_symbol, p_datetime, p_spot,
        p_total_gex, p_call_gex, p_put_gex, p_gamma_flip,
        p_call_wall, p_put_wall, p_max_pain,
        p_atm_iv_pct, p_positive_gamma, p_regime_code,
        p_rr_25, p_skew_slope, p_rr_25_zscore, p_skew_signal
    )
    ON CONFLICT (symbol, datetime) DO UPDATE SET
        spot = EXCLUDED.spot,
        total_gex = EXCLUDED.total_gex,
        call_gex = EXCLUDED.call_gex,
        put_gex = EXCLUDED.put_gex,
        gamma_flip = EXCLUDED.gamma_flip,
        call_wall = EXCLUDED.call_wall,
        put_wall = EXCLUDED.put_wall,
        max_pain = EXCLUDED.max_pain,
        atm_iv_pct = EXCLUDED.atm_iv_pct,
        positive_gamma = EXCLUDED.positive_gamma,
        regime_code = EXCLUDED.regime_code,
        rr_25 = EXCLUDED.rr_25,
        skew_slope = EXCLUDED.skew_slope,
        rr_25_zscore = EXCLUDED.rr_25_zscore,
        skew_signal = EXCLUDED.skew_signal;

    GET DIAGNOSTICS result = ROW_COUNT;
    RETURN result;
END;
$$ LANGUAGE plpgsql;

-- 便捷视图：最新 GEX + 价格
CREATE OR REPLACE VIEW latest_gex AS
SELECT DISTINCT ON (symbol)
    g.symbol, g.datetime, g.spot,
    g.total_gex, g.gamma_flip,
    g.call_wall, g.put_wall, g.max_pain,
    g.atm_iv_pct, g.positive_gamma, g.regime_code,
    g.rr_25, g.skew_slope, g.rr_25_zscore, g.skew_signal
FROM gex_snapshots g
ORDER BY symbol, datetime DESC;

-- 便捷视图：GEX + OHLC JOIN（1 分钟对齐）
CREATE OR REPLACE VIEW gex_with_price AS
SELECT
    g.symbol,
    g.datetime,
    b.open, b.high, b.low, b.close, b.volume,
    g.spot, g.total_gex, g.gamma_flip,
    g.call_gex, g.put_gex,
    g.call_wall, g.put_wall, g.max_pain,
    g.atm_iv_pct, g.positive_gamma, g.regime_code,
    g.rr_25, g.skew_slope, g.rr_25_zscore, g.skew_signal
FROM gex_snapshots g
LEFT JOIN market_data_bars b
    ON g.symbol = b.symbol
    AND date_trunc('minute', g.datetime) = b.datetime
    AND b.bar_size = '1 min';
