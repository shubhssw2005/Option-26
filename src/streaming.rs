/// SQLite persistence for collected data
use rusqlite::{Connection, params};
use crate::models::{OptionItem, WsGreekItem};
use crate::error::Result;

pub fn init_db(path: &str) -> Result<Connection> {
    let conn = Connection::open(path)?;
    conn.execute_batch("
        PRAGMA journal_mode=WAL;

        CREATE TABLE IF NOT EXISTS option_chain_snapshot (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            collected_at INTEGER NOT NULL,
            asset       TEXT NOT NULL,
            exchange    TEXT NOT NULL,
            expiry      TEXT NOT NULL,
            option_type TEXT NOT NULL,  -- 'CE' or 'PE'
            inst_id     INTEGER,
            strike      REAL,
            ltp         REAL,
            iv          REAL,
            delta       REAL,
            gamma       REAL,
            theta       REAL,
            vega        REAL,
            oi          INTEGER,
            volume      INTEGER
        );

        CREATE TABLE IF NOT EXISTS greeks_tick (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ts          INTEGER NOT NULL,
            inst_id     INTEGER NOT NULL,
            strike      REAL,
            ltp         REAL,
            iv          REAL,
            delta       REAL,
            gamma       REAL,
            theta       REAL,
            vega        REAL,
            oi          INTEGER,
            volume      INTEGER,
            prev_oi     INTEGER
        );

        CREATE TABLE IF NOT EXISTS historical_candle (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol      TEXT NOT NULL,
            exchange    TEXT NOT NULL,
            interval    TEXT NOT NULL,
            ts          INTEGER NOT NULL,
            open        REAL,
            high        REAL,
            low         REAL,
            close       REAL,
            volume      INTEGER,
            UNIQUE(symbol, exchange, interval, ts)
        );

        CREATE INDEX IF NOT EXISTS idx_greeks_inst ON greeks_tick(inst_id, ts);
        CREATE INDEX IF NOT EXISTS idx_chain_asset ON option_chain_snapshot(asset, expiry, collected_at);
        CREATE INDEX IF NOT EXISTS idx_candle_sym ON historical_candle(symbol, exchange, interval, ts);
    ")?;
    Ok(conn)
}

pub fn insert_option_chain_snapshot(
    conn: &Connection,
    collected_at: i64,
    asset: &str,
    exchange: &str,
    expiry: &str,
    option_type: &str,
    items: &[OptionItem],
) -> Result<()> {
    let mut stmt = conn.prepare_cached("
        INSERT INTO option_chain_snapshot
            (collected_at, asset, exchange, expiry, option_type, inst_id, strike, ltp, iv, delta, gamma, theta, vega, oi, volume)
        VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14,?15)
    ")?;

    for item in items {
        stmt.execute(params![
            collected_at, asset, exchange, expiry, option_type,
            item.inst_id,
            item.sp.map(|v| v as f64 / 100.0),
            item.ltp.map(|v| v as f64 / 100.0),
            item.iv,
            item.delta, item.gamma, item.theta, item.vega,
            item.oi, item.volume
        ])?;
    }
    Ok(())
}

pub fn insert_greek_tick(conn: &Connection, tick: &WsGreekItem) -> Result<()> {
    conn.execute("
        INSERT INTO greeks_tick (ts, inst_id, strike, ltp, iv, delta, gamma, theta, vega, oi, volume, prev_oi)
        VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12)
    ", params![
        tick.ts, tick.inst_id,
        tick.sp as f64 / 100.0,
        tick.ltp as f64 / 100.0,
        tick.iv, tick.delta, tick.gamma, tick.theta, tick.vega,
        tick.oi, tick.volume, tick.prev_oi
    ])?;
    Ok(())
}

pub fn insert_historical_candle(
    conn: &Connection,
    symbol: &str,
    exchange: &str,
    interval: &str,
    ts: i64,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    volume: i64,
) -> Result<()> {
    conn.execute("
        INSERT OR REPLACE INTO historical_candle (symbol, exchange, interval, ts, open, high, low, close, volume)
        VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9)
    ", params![symbol, exchange, interval, ts, open, high, low, close, volume])?;
    Ok(())
}
