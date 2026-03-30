use serde::{Deserialize, Serialize};

// ── Auth ──────────────────────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
pub struct PhoneOtpRequest {
    pub phone: String,
    pub skip_totp: bool,
}

#[derive(Debug, Deserialize)]
pub struct PhoneOtpResponse {
    pub temp_token: String,
    pub message: String,
}

#[derive(Debug, Serialize)]
pub struct VerifyOtpRequest {
    pub phone: String,
    pub otp: String,
}

#[derive(Debug, Deserialize)]
pub struct VerifyOtpResponse {
    pub auth_token: String,
    pub message: String,
}

#[derive(Debug, Serialize)]
pub struct VerifyPinRequest {
    pub pin: String,
}

#[derive(Debug, Deserialize)]
pub struct VerifyPinResponse {
    pub session_token: String,
    pub message: String,
    #[serde(rename = "userId")]
    pub user_id: Option<u64>,
}

// ── Option Chain (REST) ───────────────────────────────────────────────────────

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct OptionChainResponse {
    pub chain: OptionChain,
    pub message: String,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct OptionChain {
    pub asset: String,
    pub exchange: String,
    pub expiry: String,
    pub ce: Vec<OptionItem>,
    pub pe: Vec<OptionItem>,
    pub atm: Option<i64>,
    pub cp: Option<i64>,
    pub all_expiries: Vec<String>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct OptionItem {
    pub ref_id: Option<i64>,
    pub inst_id: Option<i64>,
    pub ts: Option<i64>,
    pub sp: Option<i64>,       // strike price (scaled)
    pub ls: Option<i32>,       // lot size
    pub ltp: Option<i64>,      // last traded price (scaled)
    pub ltpchg: Option<f64>,
    pub iv: Option<f64>,
    pub delta: Option<f64>,
    pub gamma: Option<f64>,
    pub theta: Option<f64>,
    pub vega: Option<f64>,
    pub oi: Option<i64>,
    pub volume: Option<i64>,
}

// ── Historical (REST) ─────────────────────────────────────────────────────────

#[derive(Debug, Serialize, Clone)]
pub struct HistoricalQuery {
    pub exchange: String,
    #[serde(rename = "type")]
    pub instrument_type: String,
    pub values: Vec<String>,
    pub fields: Vec<String>,
    #[serde(rename = "startDate")]
    pub start_date: String,
    #[serde(rename = "endDate")]
    pub end_date: String,
    pub interval: String,
    #[serde(rename = "intraDay")]
    pub intra_day: bool,
    #[serde(rename = "realTime")]
    pub real_time: bool,
}

#[derive(Debug, Serialize, Clone)]
pub struct HistoricalRequest {
    pub query: Vec<HistoricalQuery>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct HistoricalResponse {
    pub market_time: Option<String>,
    pub message: String,
    pub result: Vec<serde_json::Value>,
}

// ── WebSocket tick (Greeks / Option Chain) ────────────────────────────────────

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct WsGreekItem {
    pub inst_id: i64,
    pub ts: i64,
    pub sp: i64,
    pub ls: i32,
    pub ltp: i64,
    pub ltpchg: f32,
    pub iv: f32,
    pub delta: f32,
    pub gamma: f32,
    pub theta: f32,
    pub vega: f32,
    pub oi: i64,
    pub volume: i64,
    pub ref_id: i64,
    pub prev_oi: i64,
    pub price_pcp: i64,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct WsIndexTick {
    pub indexname: String,
    pub timestamp: i64,
    pub index_value: i64,
    pub high_index_value: i64,
    pub low_index_value: i64,
    pub volume: i64,
    pub changepercent: f32,
    pub tick_volume: i64,
    pub prev_close: i64,
    pub exchange: String,
    pub volume_oi: i64,
}
