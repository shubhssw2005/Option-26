use reqwest::Client;
use crate::error::Result;
use crate::models::*;

const DEVICE_ID: &str = "TS123";

pub struct NubraClient {
    http: Client,
    base_url: String,
}

impl NubraClient {
    pub fn new(base_url: &str) -> Self {
        Self {
            http: Client::new(),
            base_url: base_url.trim_end_matches('/').to_string(),
        }
    }

    pub fn production() -> Self { Self::new("https://api.nubra.io") }
    pub fn uat() -> Self { Self::new("https://uatapi.nubra.io") }

    // ── Auth ──────────────────────────────────────────────────────────────────

    /// Step 1 + 2: send OTP, returns temp_token
    pub async fn send_otp(&self, phone: &str) -> Result<String> {
        // Step 1: initiate
        let resp: PhoneOtpResponse = self.http
            .post(format!("{}/sendphoneotp", self.base_url))
            .json(&PhoneOtpRequest { phone: phone.to_string(), skip_totp: false })
            .send().await?
            .json().await?;
        let temp1 = resp.temp_token;

        // Step 2: trigger OTP send
        let resp2: PhoneOtpResponse = self.http
            .post(format!("{}/sendphoneotp", self.base_url))
            .header("x-temp-token", &temp1)
            .json(&PhoneOtpRequest { phone: phone.to_string(), skip_totp: true })
            .send().await?
            .json().await?;

        Ok(resp2.temp_token)
    }

    /// Step 3: verify OTP, returns auth_token
    pub async fn verify_otp(&self, phone: &str, otp: &str, temp_token: &str) -> Result<String> {
        let resp: VerifyOtpResponse = self.http
            .post(format!("{}/verifyphoneotp", self.base_url))
            .header("x-temp-token", temp_token)
            .header("x-device-id", DEVICE_ID)
            .json(&VerifyOtpRequest { phone: phone.to_string(), otp: otp.to_string() })
            .send().await?
            .json().await?;
        Ok(resp.auth_token)
    }

    /// Step 4: verify PIN, returns session_token
    pub async fn verify_pin(&self, auth_token: &str, pin: &str) -> Result<String> {
        let resp: VerifyPinResponse = self.http
            .post(format!("{}/verifypin", self.base_url))
            .header("x-device-id", DEVICE_ID)
            .bearer_auth(auth_token)
            .json(&VerifyPinRequest { pin: pin.to_string() })
            .send().await?
            .json().await?;
        Ok(resp.session_token)
    }

    // ── Option Chain ──────────────────────────────────────────────────────────

    pub async fn get_option_chain(
        &self,
        session_token: &str,
        instrument: &str,
        exchange: &str,
        expiry: &str,
    ) -> Result<OptionChainResponse> {
        let url = format!(
            "{}/optionchains/{}?exchange={}&expiry={}",
            self.base_url, instrument, exchange, expiry
        );
        let resp: OptionChainResponse = self.http
            .get(&url)
            .header("x-device-id", DEVICE_ID)
            .bearer_auth(session_token)
            .send().await?
            .json().await?;
        Ok(resp)
    }

    // ── Historical ────────────────────────────────────────────────────────────

    pub async fn get_historical(
        &self,
        session_token: &str,
        query: HistoricalQuery,
    ) -> Result<HistoricalResponse> {
        let body = HistoricalRequest { query: vec![query] };
        let resp: HistoricalResponse = self.http
            .post(format!("{}/charts/timeseries", self.base_url))
            .header("x-device-id", DEVICE_ID)
            .header("Content-Type", "application/json")
            .bearer_auth(session_token)
            .json(&body)
            .send().await?
            .json().await?;
        Ok(resp)
    }
}

/// Build the WebSocket URL for the given environment
pub fn ws_url(production: bool) -> &'static str {
    if production {
        "wss://api.nubra.io/apibatch/ws"
    } else {
        "wss://uatapi.nubra.io/apibatch/ws"
    }
}

/// Format a subscribe command for the WebSocket
pub fn subscribe_cmd(token: &str, channel: &str, payload: &str) -> String {
    format!("batch_subscribe {} {} {}", token, channel, payload)
}

pub fn unsubscribe_cmd(token: &str, channel: &str, payload: &str) -> String {
    format!("batch_unsubscribe {} {} {}", token, channel, payload)
}
