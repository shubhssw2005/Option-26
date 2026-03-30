use std::sync::Arc;
use tokio::sync::mpsc;
use tokio_tungstenite::{connect_async, tungstenite::Message};
use futures_util::{SinkExt, StreamExt};
use tracing::{info, warn, error};
use serde_json::Value;

use crate::api::nubario::{ws_url, subscribe_cmd};
use crate::cache::TickCache;
use crate::models::{WsGreekItem, WsIndexTick};

/// Commands sent to the WebSocket task
#[derive(Debug, Clone)]
pub enum WsCommand {
    SubscribeGreeks { instruments: Vec<i64> },
    SubscribeIndex { symbols: Vec<String>, exchange: String },
    SubscribeOptionChain { asset: String, exchange: String, expiry: String },
    SetInterval { channel: String, interval: String },
    Disconnect,
}

/// Spawn the WebSocket collector. Returns a command sender.
pub async fn start_ws_collector(
    token: String,
    cache: Arc<TickCache>,
    production: bool,
) -> mpsc::Sender<WsCommand> {
    let (cmd_tx, mut cmd_rx) = mpsc::channel::<WsCommand>(64);
    let url = ws_url(production);

    tokio::spawn(async move {
        loop {
            info!("Connecting to WebSocket: {}", url);
            match connect_async(url).await {
                Err(e) => {
                    error!("WS connect failed: {e}. Retrying in 5s...");
                    tokio::time::sleep(std::time::Duration::from_secs(5)).await;
                    continue;
                }
                Ok((ws_stream, _)) => {
                    let (mut write, mut read) = ws_stream.split();

                    loop {
                        tokio::select! {
                            // Incoming tick from server
                            msg = read.next() => {
                                match msg {
                                    Some(Ok(Message::Text(txt))) => {
                                        handle_text_message(&txt, &cache);
                                    }
                                    Some(Ok(Message::Binary(bin))) => {
                                        handle_binary_message(&bin, &cache);
                                    }
                                    Some(Ok(Message::Ping(p))) => {
                                        let _ = write.send(Message::Pong(p)).await;
                                    }
                                    Some(Err(e)) => {
                                        warn!("WS read error: {e}");
                                        break;
                                    }
                                    None => {
                                        warn!("WS stream closed");
                                        break;
                                    }
                                    _ => {}
                                }
                            }
                            // Command from application
                            cmd = cmd_rx.recv() => {
                                match cmd {
                                    Some(WsCommand::Disconnect) | None => break,
                                    Some(WsCommand::SubscribeGreeks { instruments }) => {
                                        let ids: Vec<String> = instruments.iter().map(|i| i.to_string()).collect();
                                        let payload = format!(r#"{{"instruments":[{}]}}"#, ids.join(","));
                                        let msg = subscribe_cmd(&token, "greeks", &payload);
                                        info!("WS -> {msg}");
                                        let _ = write.send(Message::Text(msg)).await;
                                    }
                                    Some(WsCommand::SubscribeIndex { symbols, exchange }) => {
                                        let syms: Vec<String> = symbols.iter().map(|s| format!(r#""{}""#, s)).collect();
                                        let payload = format!(r#"{{"indexes":[{}]}}"#, syms.join(","));
                                        let msg = format!("batch_subscribe {} index {} {}", token, payload, exchange);
                                        info!("WS -> {msg}");
                                        let _ = write.send(Message::Text(msg)).await;
                                    }
                                    Some(WsCommand::SubscribeOptionChain { asset, exchange, expiry }) => {
                                        let payload = format!(
                                            r#"[{{"exchange":"{}","asset":"{}","expiry":"{}"}}]"#,
                                            exchange, asset, expiry
                                        );
                                        let msg = subscribe_cmd(&token, "option", &payload);
                                        info!("WS -> {msg}");
                                        let _ = write.send(Message::Text(msg)).await;
                                    }
                                    Some(WsCommand::SetInterval { channel, interval }) => {
                                        let msg = format!("batch_subscribe {} socket_interval {} {}", token, channel, interval);
                                        info!("WS -> {msg}");
                                        let _ = write.send(Message::Text(msg)).await;
                                    }
                                }
                            }
                        }
                    }
                    // Reconnect after disconnect
                    tokio::time::sleep(std::time::Duration::from_secs(3)).await;
                }
            }
        }
    });

    cmd_tx
}

fn handle_text_message(txt: &str, cache: &Arc<TickCache>) {
    // Try to parse as JSON — Nubra may send JSON for some channels
    match serde_json::from_str::<Value>(txt) {
        Ok(v) => {
            // Greeks array
            if let Some(instruments) = v.get("instruments").and_then(|i| i.as_array()) {
                for item in instruments {
                    if let Ok(greek) = serde_json::from_value::<WsGreekItem>(item.clone()) {
                        cache.update_greek(greek);
                    }
                }
            }
            // Index array
            if let Some(indexes) = v.get("indexes").and_then(|i| i.as_array()) {
                for item in indexes {
                    if let Ok(tick) = serde_json::from_value::<WsIndexTick>(item.clone()) {
                        cache.update_index(tick);
                    }
                }
            }
        }
        Err(_) => {
            // Protobuf binary wrapped in text — log and skip (handled in binary handler)
            warn!("Non-JSON WS text message (likely proto): {} bytes", txt.len());
        }
    }
}

fn handle_binary_message(bin: &[u8], _cache: &Arc<TickCache>) {
    // Protobuf decoding would go here once .proto files are compiled with prost-build
    // For now log the size
    info!("Binary WS message: {} bytes", bin.len());
}
