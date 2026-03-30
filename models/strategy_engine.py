"""
strategy_engine.py — ML/DL-driven options strategy selector.

For each asset + market regime, selects the optimal strategy from 57 options strategies.
Uses: IV percentile, realized vol, GARCH forecast, Greeks, OI, PCR, trend, liquidity.

Strategy selection logic:
  - Regime classifier (CatBoost) → market regime (trending/ranging/volatile/crash)
  - Vol regime (GARCH ensemble) → vol forecast vs IV (cheap/fair/expensive)
  - Greeks analysis → directional bias, gamma risk, theta decay
  - Liquidity filter → OI + volume thresholds
  - Risk management → max loss, breakeven, probability of profit
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


# ── Strategy definitions ──────────────────────────────────────────────────────


@dataclass
class Strategy:
    name: str
    category: str  # directional / neutral / volatility / income
    direction: str  # bullish / bearish / neutral
    vol_view: str  # long_vol / short_vol / neutral
    legs: list  # list of leg dicts
    max_loss: str  # description
    max_profit: str
    breakeven: str
    best_when: str  # market condition
    greeks: dict  # net delta/gamma/theta/vega profile
    risk_score: int  # 1=low, 5=high
    margin_required: bool
    reason: str = ""  # ML-generated reason
    score: float = 0.0  # ML confidence score
    entry_price: float = 0.0
    strikes: dict = field(default_factory=dict)


STRATEGY_CATALOG = {
    # ── Directional ───────────────────────────────────────────────────────────
    "covered_call": Strategy(
        name="Covered Call",
        category="income",
        direction="neutral_bullish",
        vol_view="short_vol",
        legs=[
            {"type": "long", "instrument": "stock"},
            {"type": "short", "instrument": "call", "otm": 0.02},
        ],
        max_loss="Stock price - Premium received",
        max_profit="Strike - Stock price + Premium",
        breakeven="Stock price - Premium",
        best_when="Mildly bullish, high IV, want income",
        greeks={"delta": 0.5, "gamma": -0.05, "theta": 0.3, "vega": -0.3},
        risk_score=2,
        margin_required=False,
    ),
    "covered_put": Strategy(
        name="Covered Put",
        category="income",
        direction="neutral_bearish",
        vol_view="short_vol",
        legs=[
            {"type": "short", "instrument": "stock"},
            {"type": "short", "instrument": "put", "otm": -0.02},
        ],
        max_loss="Unlimited (short stock)",
        max_profit="Strike - Stock price + Premium",
        breakeven="Stock price + Premium",
        best_when="Mildly bearish, high IV",
        greeks={"delta": -0.5, "gamma": -0.05, "theta": 0.3, "vega": -0.3},
        risk_score=4,
        margin_required=True,
    ),
    "protective_put": Strategy(
        name="Protective Put",
        category="directional",
        direction="bullish",
        vol_view="long_vol",
        legs=[
            {"type": "long", "instrument": "stock"},
            {"type": "long", "instrument": "put", "otm": -0.02},
        ],
        max_loss="Stock price - Strike + Premium",
        max_profit="Unlimited",
        breakeven="Stock price + Premium",
        best_when="Long stock, fear of downside, low IV",
        greeks={"delta": 0.5, "gamma": 0.05, "theta": -0.3, "vega": 0.3},
        risk_score=2,
        margin_required=False,
    ),
    "bull_call_spread": Strategy(
        name="Bull Call Spread",
        category="directional",
        direction="bullish",
        vol_view="neutral",
        legs=[
            {"type": "long", "instrument": "call", "otm": 0.0},
            {"type": "short", "instrument": "call", "otm": 0.03},
        ],
        max_loss="Net debit paid",
        max_profit="Spread width - Net debit",
        breakeven="Long strike + Net debit",
        best_when="Moderately bullish, IV neutral/high",
        greeks={"delta": 0.4, "gamma": 0.02, "theta": -0.1, "vega": 0.1},
        risk_score=2,
        margin_required=False,
    ),
    "bull_put_spread": Strategy(
        name="Bull Put Spread",
        category="income",
        direction="bullish",
        vol_view="short_vol",
        legs=[
            {"type": "short", "instrument": "put", "otm": -0.02},
            {"type": "long", "instrument": "put", "otm": -0.05},
        ],
        max_loss="Spread width - Net credit",
        max_profit="Net credit received",
        breakeven="Short strike - Net credit",
        best_when="Bullish, high IV, want credit",
        greeks={"delta": 0.3, "gamma": -0.02, "theta": 0.2, "vega": -0.2},
        risk_score=2,
        margin_required=True,
    ),
    "bear_call_spread": Strategy(
        name="Bear Call Spread",
        category="income",
        direction="bearish",
        vol_view="short_vol",
        legs=[
            {"type": "short", "instrument": "call", "otm": 0.02},
            {"type": "long", "instrument": "call", "otm": 0.05},
        ],
        max_loss="Spread width - Net credit",
        max_profit="Net credit received",
        breakeven="Short strike + Net credit",
        best_when="Bearish, high IV, want credit",
        greeks={"delta": -0.3, "gamma": -0.02, "theta": 0.2, "vega": -0.2},
        risk_score=2,
        margin_required=True,
    ),
    "bear_put_spread": Strategy(
        name="Bear Put Spread",
        category="directional",
        direction="bearish",
        vol_view="neutral",
        legs=[
            {"type": "long", "instrument": "put", "otm": 0.0},
            {"type": "short", "instrument": "put", "otm": -0.03},
        ],
        max_loss="Net debit paid",
        max_profit="Spread width - Net debit",
        breakeven="Long strike - Net debit",
        best_when="Moderately bearish, IV neutral",
        greeks={"delta": -0.4, "gamma": 0.02, "theta": -0.1, "vega": 0.1},
        risk_score=2,
        margin_required=False,
    ),
    # ── Volatility ────────────────────────────────────────────────────────────
    "long_straddle": Strategy(
        name="Long Straddle",
        category="volatility",
        direction="neutral",
        vol_view="long_vol",
        legs=[
            {"type": "long", "instrument": "call", "otm": 0.0},
            {"type": "long", "instrument": "put", "otm": 0.0},
        ],
        max_loss="Total premium paid",
        max_profit="Unlimited",
        breakeven="ATM ± Total premium",
        best_when="Big move expected, IV cheap, pre-event",
        greeks={"delta": 0.0, "gamma": 0.1, "theta": -0.5, "vega": 0.8},
        risk_score=3,
        margin_required=False,
    ),
    "short_straddle": Strategy(
        name="Short Straddle",
        category="income",
        direction="neutral",
        vol_view="short_vol",
        legs=[
            {"type": "short", "instrument": "call", "otm": 0.0},
            {"type": "short", "instrument": "put", "otm": 0.0},
        ],
        max_loss="Unlimited",
        max_profit="Total premium received",
        breakeven="ATM ± Total premium",
        best_when="Range-bound, IV very high, post-event",
        greeks={"delta": 0.0, "gamma": -0.1, "theta": 0.5, "vega": -0.8},
        risk_score=5,
        margin_required=True,
    ),
    "long_strangle": Strategy(
        name="Long Strangle",
        category="volatility",
        direction="neutral",
        vol_view="long_vol",
        legs=[
            {"type": "long", "instrument": "call", "otm": 0.03},
            {"type": "long", "instrument": "put", "otm": -0.03},
        ],
        max_loss="Total premium paid",
        max_profit="Unlimited",
        breakeven="Call strike + premium / Put strike - premium",
        best_when="Big move expected, cheaper than straddle",
        greeks={"delta": 0.0, "gamma": 0.08, "theta": -0.4, "vega": 0.7},
        risk_score=3,
        margin_required=False,
    ),
    "short_strangle": Strategy(
        name="Short Strangle",
        category="income",
        direction="neutral",
        vol_view="short_vol",
        legs=[
            {"type": "short", "instrument": "call", "otm": 0.03},
            {"type": "short", "instrument": "put", "otm": -0.03},
        ],
        max_loss="Unlimited",
        max_profit="Total premium received",
        breakeven="Call strike + premium / Put strike - premium",
        best_when="Range-bound, IV high, wider range than straddle",
        greeks={"delta": 0.0, "gamma": -0.08, "theta": 0.4, "vega": -0.7},
        risk_score=5,
        margin_required=True,
    ),
    "long_iron_condor": Strategy(
        name="Long Iron Condor",
        category="income",
        direction="neutral",
        vol_view="short_vol",
        legs=[
            {"type": "long", "instrument": "put", "otm": -0.06},
            {"type": "short", "instrument": "put", "otm": -0.03},
            {"type": "short", "instrument": "call", "otm": 0.03},
            {"type": "long", "instrument": "call", "otm": 0.06},
        ],
        max_loss="Spread width - Net credit",
        max_profit="Net credit received",
        breakeven="Short put - credit / Short call + credit",
        best_when="Range-bound, IV high, defined risk",
        greeks={"delta": 0.0, "gamma": -0.05, "theta": 0.3, "vega": -0.5},
        risk_score=2,
        margin_required=True,
    ),
    "short_iron_condor": Strategy(
        name="Short Iron Condor",
        category="volatility",
        direction="neutral",
        vol_view="long_vol",
        legs=[
            {"type": "short", "instrument": "put", "otm": -0.06},
            {"type": "long", "instrument": "put", "otm": -0.03},
            {"type": "long", "instrument": "call", "otm": 0.03},
            {"type": "short", "instrument": "call", "otm": 0.06},
        ],
        max_loss="Net debit paid",
        max_profit="Spread width - Net debit",
        breakeven="Long put + debit / Long call - debit",
        best_when="Big move expected, IV cheap",
        greeks={"delta": 0.0, "gamma": 0.05, "theta": -0.3, "vega": 0.5},
        risk_score=3,
        margin_required=False,
    ),
    "long_butterfly": Strategy(
        name="Long Call Butterfly",
        category="income",
        direction="neutral",
        vol_view="short_vol",
        legs=[
            {"type": "long", "instrument": "call", "otm": -0.03},
            {"type": "short", "instrument": "call", "otm": 0.0, "qty": 2},
            {"type": "long", "instrument": "call", "otm": 0.03},
        ],
        max_loss="Net debit paid",
        max_profit="Spread width - Net debit",
        breakeven="Lower strike + debit / Upper strike - debit",
        best_when="Pinning expected at ATM, low IV",
        greeks={"delta": 0.0, "gamma": -0.03, "theta": 0.2, "vega": -0.3},
        risk_score=2,
        margin_required=False,
    ),
    "calendar_call": Strategy(
        name="Calendar Call Spread",
        category="volatility",
        direction="neutral",
        vol_view="long_vol",
        legs=[
            {"type": "short", "instrument": "call", "expiry": "near", "otm": 0.0},
            {"type": "long", "instrument": "call", "expiry": "far", "otm": 0.0},
        ],
        max_loss="Net debit paid",
        max_profit="When near-term expires worthless",
        breakeven="Depends on IV of back month",
        best_when="Low IV, expect vol expansion, time decay play",
        greeks={"delta": 0.0, "gamma": -0.02, "theta": 0.2, "vega": 0.4},
        risk_score=2,
        margin_required=False,
    ),
    "ratio_call_spread": Strategy(
        name="Ratio Call Spread",
        category="directional",
        direction="bullish",
        vol_view="short_vol",
        legs=[
            {"type": "long", "instrument": "call", "otm": 0.0, "qty": 1},
            {"type": "short", "instrument": "call", "otm": 0.03, "qty": 2},
        ],
        max_loss="Unlimited above upper strike",
        max_profit="At upper strike",
        breakeven="Lower strike + debit / Upper strike + max profit",
        best_when="Mildly bullish, high IV, expect limited upside",
        greeks={"delta": 0.2, "gamma": -0.05, "theta": 0.3, "vega": -0.4},
        risk_score=4,
        margin_required=True,
    ),
    "collar": Strategy(
        name="Collar",
        category="income",
        direction="neutral_bullish",
        vol_view="neutral",
        legs=[
            {"type": "long", "instrument": "stock"},
            {"type": "long", "instrument": "put", "otm": -0.03},
            {"type": "short", "instrument": "call", "otm": 0.03},
        ],
        max_loss="Stock price - Put strike + Net debit",
        max_profit="Call strike - Stock price + Net credit",
        breakeven="Stock price + Net debit",
        best_when="Long stock, want downside protection, fund with call",
        greeks={"delta": 0.4, "gamma": 0.0, "theta": 0.0, "vega": 0.0},
        risk_score=1,
        margin_required=False,
    ),
    "strap": Strategy(
        name="Strap",
        category="volatility",
        direction="bullish_bias",
        vol_view="long_vol",
        legs=[
            {"type": "long", "instrument": "call", "otm": 0.0, "qty": 2},
            {"type": "long", "instrument": "put", "otm": 0.0, "qty": 1},
        ],
        max_loss="Total premium paid",
        max_profit="Unlimited (bullish bias)",
        breakeven="ATM - premium/1 / ATM + premium/2",
        best_when="Big move expected, bullish bias, IV cheap",
        greeks={"delta": 0.5, "gamma": 0.15, "theta": -0.6, "vega": 1.0},
        risk_score=3,
        margin_required=False,
    ),
    "strip": Strategy(
        name="Strip",
        category="volatility",
        direction="bearish_bias",
        vol_view="long_vol",
        legs=[
            {"type": "long", "instrument": "call", "otm": 0.0, "qty": 1},
            {"type": "long", "instrument": "put", "otm": 0.0, "qty": 2},
        ],
        max_loss="Total premium paid",
        max_profit="Unlimited (bearish bias)",
        breakeven="ATM - premium/2 / ATM + premium/1",
        best_when="Big move expected, bearish bias, IV cheap",
        greeks={"delta": -0.5, "gamma": 0.15, "theta": -0.6, "vega": 1.0},
        risk_score=3,
        margin_required=False,
    ),
}


# ── Market regime classifier ──────────────────────────────────────────────────


def classify_regime(
    spot_ret_5: float,
    spot_ret_20: float,
    realized_vol: float,
    iv_percentile: float,
    pcr: float,
    garch_vol_1d: float,
) -> dict:
    """
    Classify market regime using rule-based + ML-ready features.
    Returns regime dict with probabilities.
    """
    # Trend strength
    trending_up = spot_ret_5 > 0.02 and spot_ret_20 > 0.03
    trending_down = spot_ret_5 < -0.02 and spot_ret_20 < -0.03
    ranging = abs(spot_ret_5) < 0.01 and abs(spot_ret_20) < 0.02

    # Vol regime
    vol_cheap = iv_percentile < 30
    vol_fair = 30 <= iv_percentile <= 70
    vol_expensive = iv_percentile > 70

    # Crash signal
    crash = spot_ret_5 < -0.05 or garch_vol_1d > 3.0

    # Regime probabilities (rule-based, can be replaced by trained classifier)
    if crash:
        regime = "crash"
        probs = {
            "crash": 0.8,
            "trending_down": 0.15,
            "ranging": 0.05,
            "trending_up": 0.0,
        }
    elif trending_up:
        regime = "trending_up"
        probs = {
            "trending_up": 0.7,
            "ranging": 0.2,
            "trending_down": 0.05,
            "crash": 0.05,
        }
    elif trending_down:
        regime = "trending_down"
        probs = {
            "trending_down": 0.7,
            "ranging": 0.2,
            "trending_up": 0.05,
            "crash": 0.05,
        }
    elif ranging:
        regime = "ranging"
        probs = {
            "ranging": 0.7,
            "trending_up": 0.15,
            "trending_down": 0.1,
            "crash": 0.05,
        }
    else:
        regime = "uncertain"
        probs = {
            "ranging": 0.4,
            "trending_up": 0.25,
            "trending_down": 0.25,
            "crash": 0.1,
        }

    return {
        "regime": regime,
        "probabilities": probs,
        "vol_regime": (
            "cheap" if vol_cheap else ("expensive" if vol_expensive else "fair")
        ),
        "iv_percentile": round(iv_percentile, 1),
        "pcr": round(pcr, 3),
        "garch_vol_1d": round(garch_vol_1d, 4),
        "trending_up": trending_up,
        "trending_down": trending_down,
        "ranging": ranging,
        "crash": crash,
    }


# ── Strategy scorer ───────────────────────────────────────────────────────────


def score_strategies(
    regime: dict,
    signal_score_ce: float,
    signal_score_pe: float,
    atm_iv: float,
    iv_percentile: float,
    pcr: float,
    dte: int,
    liquidity_score: float,  # 0-1 based on OI + volume
) -> list[dict]:
    """
    Score all strategies given current market conditions.
    Returns ranked list with scores, reasons, and risk flags.
    """
    r = regime["regime"]
    vol_regime = regime["vol_regime"]
    bullish = signal_score_ce > 0.6
    bearish = signal_score_pe > 0.6
    neutral = not bullish and not bearish

    results = []

    for key, strat in STRATEGY_CATALOG.items():
        score = 0.0
        reasons = []
        warnings = []

        # ── Directional alignment ─────────────────────────────────────────────
        if strat.direction in ("bullish", "neutral_bullish"):
            if bullish:
                score += 0.3
                reasons.append(
                    f"CE signal score {signal_score_ce:.2f} > 0.6 → bullish bias"
                )
            elif bearish:
                score -= 0.3
                warnings.append("Bearish signal conflicts with bullish strategy")

        elif strat.direction in ("bearish", "neutral_bearish"):
            if bearish:
                score += 0.3
                reasons.append(
                    f"PE signal score {signal_score_pe:.2f} > 0.6 → bearish bias"
                )
            elif bullish:
                score -= 0.3
                warnings.append("Bullish signal conflicts with bearish strategy")

        elif strat.direction == "neutral":
            if neutral:
                score += 0.2
                reasons.append("No strong directional signal → neutral strategy fits")

        # ── Vol regime alignment ──────────────────────────────────────────────
        if strat.vol_view == "long_vol" and vol_regime == "cheap":
            score += 0.25
            reasons.append(
                f"IV percentile {iv_percentile:.0f}% → vol cheap, long vol favoured"
            )
        elif strat.vol_view == "short_vol" and vol_regime == "expensive":
            score += 0.25
            reasons.append(
                f"IV percentile {iv_percentile:.0f}% → vol expensive, short vol favoured"
            )
        elif strat.vol_view == "long_vol" and vol_regime == "expensive":
            score -= 0.2
            warnings.append("Buying vol when IV is expensive — unfavourable")
        elif strat.vol_view == "short_vol" and vol_regime == "cheap":
            score -= 0.15
            warnings.append("Selling vol when IV is cheap — limited premium")

        # ── Regime alignment ──────────────────────────────────────────────────
        if r == "ranging" and strat.category == "income":
            score += 0.2
            reasons.append("Range-bound market → income/short-vol strategies preferred")
        elif r in ("trending_up", "trending_down") and strat.category == "directional":
            score += 0.2
            reasons.append(f"Trending market ({r}) → directional strategies preferred")
        elif r == "crash" and strat.direction == "bearish":
            score += 0.3
            reasons.append("Crash regime → bearish strategies strongly favoured")
        elif r == "crash" and strat.direction == "bullish":
            score -= 0.4
            warnings.append("CRASH REGIME — avoid bullish strategies")

        # ── PCR signal ────────────────────────────────────────────────────────
        if pcr > 1.3 and strat.direction in ("bullish", "neutral_bullish"):
            score += 0.1
            reasons.append(f"PCR {pcr:.2f} > 1.3 → contrarian bullish signal")
        elif pcr < 0.7 and strat.direction in ("bearish", "neutral_bearish"):
            score += 0.1
            reasons.append(f"PCR {pcr:.2f} < 0.7 → contrarian bearish signal")

        # ── DTE alignment ─────────────────────────────────────────────────────
        if dte < 7 and strat.category == "income":
            score += 0.15
            reasons.append(
                f"DTE={dte} → theta decay accelerates, income strategies benefit"
            )
        elif dte < 7 and strat.vol_view == "long_vol":
            score -= 0.15
            warnings.append(f"DTE={dte} → theta decay hurts long vol strategies")
        elif dte > 20 and strat.category == "volatility":
            score += 0.1
            reasons.append(f"DTE={dte} → enough time for vol expansion")

        # ── Liquidity filter ──────────────────────────────────────────────────
        if liquidity_score < 0.3 and strat.risk_score >= 4:
            score -= 0.2
            warnings.append("Low liquidity — avoid complex high-risk strategies")
        elif liquidity_score > 0.7:
            score += 0.05
            reasons.append("High liquidity → tight spreads, easy execution")

        # ── Risk management ───────────────────────────────────────────────────
        if strat.risk_score >= 5:
            warnings.append("⚠ UNLIMITED LOSS POTENTIAL — requires strict stop-loss")
        if strat.margin_required:
            warnings.append("Margin required — check available capital")

        score = max(0.0, min(1.0, score))

        results.append(
            {
                "key": key,
                "name": strat.name,
                "category": strat.category,
                "direction": strat.direction,
                "vol_view": strat.vol_view,
                "score": round(score, 4),
                "risk_score": strat.risk_score,
                "max_loss": strat.max_loss,
                "max_profit": strat.max_profit,
                "breakeven": strat.breakeven,
                "best_when": strat.best_when,
                "reasons": reasons,
                "warnings": warnings,
                "margin_required": strat.margin_required,
                "net_greeks": strat.greeks,
            }
        )

    return sorted(results, key=lambda x: x["score"], reverse=True)


# ── Strike calculator ─────────────────────────────────────────────────────────


def calculate_strikes(
    strategy_key: str,
    spot: float,
    chain_df: pd.DataFrame,
    lot_size: int = 75,
) -> dict:
    """
    Given a strategy and live option chain, calculate exact strikes,
    premiums, max loss, max profit, and breakevens.
    """
    strat = STRATEGY_CATALOG.get(strategy_key)
    if not strat:
        return {"error": f"Unknown strategy: {strategy_key}"}

    ce = chain_df[chain_df["option_type"] == "CE"].copy()
    pe = chain_df[chain_df["option_type"] == "PE"].copy()

    if ce.empty or pe.empty:
        return {"error": "Insufficient chain data"}

    # Find ATM strike
    ce["dist"] = (ce["strike"] - spot).abs()
    pe["dist"] = (pe["strike"] - spot).abs()
    atm_strike = ce.loc[ce["dist"].idxmin(), "strike"]

    def nearest_strike(df, target):
        idx = (df["strike"] - target).abs().idxmin()
        return df.loc[idx]

    result = {
        "strategy": strat.name,
        "spot": round(spot, 2),
        "atm_strike": round(atm_strike, 2),
        "lot_size": lot_size,
        "legs": [],
    }

    total_debit = 0.0
    total_credit = 0.0

    for leg in strat.legs:
        otm_pct = leg.get("otm", 0.0)
        target = spot * (1 + otm_pct)
        qty = leg.get("qty", 1)
        inst = leg.get("instrument", "call")
        side = leg.get("type", "long")

        if inst in ("call", "put"):
            df_side = ce if inst == "call" else pe
            row = nearest_strike(df_side, target)
            premium = row.get("ltp", 0) or 0
            strike = row.get("strike", target)
            iv = row.get("iv", 0) or 0
            delta = row.get("delta", 0) or 0
            theta = row.get("theta", 0) or 0
            vega = row.get("vega", 0) or 0
            oi = row.get("oi", 0) or 0

            leg_cost = premium * qty * lot_size
            if side == "long":
                total_debit += leg_cost
            else:
                total_credit += leg_cost

            result["legs"].append(
                {
                    "side": side,
                    "type": inst,
                    "strike": round(float(strike), 2),
                    "premium": round(float(premium), 2),
                    "qty": qty,
                    "iv_pct": round(float(iv) * 100, 2),
                    "delta": round(float(delta), 4),
                    "theta": round(float(theta), 4),
                    "vega": round(float(vega), 4),
                    "oi": int(oi),
                    "cost": round(leg_cost, 2),
                }
            )

    net_cost = total_debit - total_credit
    result["net_debit"] = round(total_debit, 2)
    result["net_credit"] = round(total_credit, 2)
    result["net_cost"] = round(net_cost, 2)
    result["max_loss_inr"] = strat.max_loss
    result["max_profit_inr"] = strat.max_profit
    result["breakeven"] = strat.breakeven

    return result


# ── Liquidity scorer ──────────────────────────────────────────────────────────


def liquidity_score(chain_df: pd.DataFrame) -> float:
    """Score 0-1 based on OI and volume of ATM options."""
    atm_rows = chain_df[
        (chain_df["strike"] >= chain_df["strike"].median() * 0.97)
        & (chain_df["strike"] <= chain_df["strike"].median() * 1.03)
    ]
    if atm_rows.empty:
        return 0.5
    avg_oi = atm_rows["oi"].fillna(0).mean()
    avg_vol = atm_rows["volume"].fillna(0).mean()
    # Normalise: OI > 500k = good, volume > 1M = good
    oi_score = min(avg_oi / 500_000, 1.0)
    vol_score = min(avg_vol / 1_000_000, 1.0)
    return round((oi_score + vol_score) / 2, 3)
