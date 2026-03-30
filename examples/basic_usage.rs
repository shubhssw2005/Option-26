/// Basic usage example: authenticate and fetch an option chain snapshot
use nubra_collector::api::nubario::NubraClient;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let client = NubraClient::uat();

    // Step 1+2: send OTP
    let temp_token = client.send_otp("9372056598").await?;
    println!("OTP sent. temp_token: {}", &temp_token[..20]);

    // Step 3: verify OTP (enter manually)
    let mut otp = String::new();
    std::io::stdin().read_line(&mut otp)?;
    let auth_token = client.verify_otp("9372056598", otp.trim(), &temp_token).await?;

    // Step 4: verify PIN
    let session = client.verify_pin(&auth_token, "1211").await?;
    println!("Session token: {}...", &session[..20]);

    // Fetch NIFTY option chain
    let chain = client.get_option_chain(&session, "NIFTY", "NSE", "20260327").await?;
    println!("ATM: {:?}, CE strikes: {}", chain.chain.atm, chain.chain.ce.len());

    Ok(())
}
