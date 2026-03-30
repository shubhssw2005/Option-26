use std::sync::Arc;
use std::io::{self, Write};
use tracing::info;
use dotenv::dotenv;

use nubra_collector::{
    api::nubario::NubraClient,
    cache::TickCache,
    collector::{start_ws_collector, WsCommand},
    streaming::{init_db},
    models::HistoricalQuery,
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenv().ok();
    tracing_subscriber::fmt::init();

    let phone = std::env::var("NUBRA_PHONE").unwrap_or_else(|_| "9372056598".to_string());
    let mpin  = std::env::var("NUBRA_MPIN").unwrap_or_else(|_| "1211".to_string());
    let production = std::env::var("NUBRA_ENV").map(|v| v == "production").unwrap_or(false);
    let db_path = std::env::var("DB_PATH").unwrap_or_else(|_| "data.db".to_string());

    let client = if production { NubraClient::production() } else { NubraClient::uat() };

    // ── Auth ──────────────────────────────────────────────────────────────────
    info!("Sending OTP to {phone}...");
    let temp_token = client.send_otp(&phone).await?;

    print!("Enter OTP: ");
    io::stdout().flush()?;
    let mut otp = String::new();
    io::stdin().read_line(&mut otp)?;
    let otp = otp.trim();

    let auth_token = client.verify_otp(&phone, otp, &temp_token).await?;
    let session_token = client.verify_pin(&auth_token, &mpin).await?;
    info!("Authenticated successfully");

    // ── DB ────────────────────────────────────────────────────────────────────
    let conn = init_db(&db_path)?;
    info!("Database ready at {db_path}");

    // ── Cache ─────────────────────────────────────────────────────────────────
    let cache = Arc::new(TickCache::new());
    cache.set_token(session_token.clone());

    // ── WebSocket collector ───────────────────────────────────────────────────
    let ws_tx = start_ws_collector(session_token.clone(), cache.clone(), production).await;

    // Subscribe to NIFTY index
    ws_tx.send(WsCommand::SubscribeIndex {
        symbols: vec!["NIFTY".to_string(), "BANKNIFTY".to_string()],
        exchange: "NSE".to_string(),
    }).await?;

    // Subscribe to NIFTY option chain (nearest expiry — update as needed)
    ws_tx.send(WsCommand::SetInterval {
        channel: "option".to_string(),
        interval: "1m".to_string(),
    }).await?;

    // ── Fetch historical data example ─────────────────────────────────────────
    info!("Fetching historical NIFTY data...");
    let hist = client.get_historical(&session_token, HistoricalQuery {
        exchange: "NSE".to_string(),
        instrument_type: "INDEX".to_string(),
        values: vec!["NIFTY".to_string()],
        fields: vec!["open".to_string(), "high".to_string(), "low".to_string(), "close".to_string(), "tick_volume".to_string()],
        start_date: "2025-01-01T03:45:00.000Z".to_string(),
        end_date: "2025-06-01T09:30:00.000Z".to_string(),
        interval: "1d".to_string(),
        intra_day: false,
        real_time: false,
    }).await;

    match hist {
        Ok(h) => info!("Historical fetch OK: {} result sets", h.result.len()),
        Err(e) => tracing::warn!("Historical fetch error: {e}"),
    }

    // ── Keep running ──────────────────────────────────────────────────────────
    info!("Collector running. Press Ctrl+C to stop.");
    tokio::signal::ctrl_c().await?;
    ws_tx.send(WsCommand::Disconnect).await.ok();
    drop(conn);
    Ok(())
}
