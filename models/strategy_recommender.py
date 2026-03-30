import numpy as np
from scipy.stats import norm
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class MarketRegime(Enum):
    BULL_LOW_VOL = "bull_low_vol"
    BULL_HIGH_VOL = "bull_high_vol"
    BEAR_LOW_VOL = "bear_low_vol"
    BEAR_HIGH_VOL = "bear_high_vol"
    NEUTRAL_LOW_VOL = "neutral_low_vol"
    NEUTRAL_HIGH_VOL = "neutral_high_vol"
    HIGH_IV_PREMIUM = "high_iv_premium"
    LOW_IV_DISCOUNT = "low_iv_discount"
    SKEW_BEARISH = "skew_bearish"
    SKEW_BULLISH = "skew_bullish"

@dataclass
class StrategyInfo:
    name: str
    category: str
    legs: List[Dict]
    description: str
    best_for: str
    risk_profile: str
    max_profit: Optional[float]
    max_loss: Optional[float]
    breakeven: List[float]
    bullish: bool
    bearish: bool
    neutral: bool
    volatility_trade: bool

class StrategyEngine:
    def __init__(self, spot: float, strike: float, iv: float, dte: float, rate: float = 0.065):
        self.spot = spot
        self.strike = strike
        self.iv = iv / 100
        self.dte = dte
        self.rate = rate
        self.T = dte / 365

    def black_scholes(self, S: float, K: float, T: float, r: float, sigma: float, option_type: str) -> float:
        if T <= 0:
            if option_type == "CE":
                return max(S - K, 0)
            else:
                return max(K - S, 0)
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type == "CE":
            return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    def payoff_at_expiry(self, spot_price: float, legs: List[Dict]) -> float:
        total = 0
        for leg in legs:
            K = leg['strike']
            qty = leg['qty']
            opt_type = leg['type']
            premium = leg.get('premium', 0)
            
            if opt_type == "CE":
                pnl = qty * max(spot_price - K, 0)
            else:
                pnl = qty * max(K - spot_price, 0)
            
            total += pnl - qty * premium
        return total

    def get_payoff_range(self, legs: List[Dict], spot_low: float, spot_high: float, steps: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        spots = np.linspace(spot_low, spot_high, steps)
        payoffs = np.array([self.payoff_at_expiry(s, legs) for s in spots])
        return spots, payoffs

    def get_breakevens(self, legs: List[Dict], spot_low: float, spot_high: float) -> List[float]:
        spots, payoffs = self.get_payoff_range(legs, spot_low, spot_high)
        breakevens = []
        for i in range(len(payoffs) - 1):
            if (payoffs[i] < 0 and payoffs[i+1] >= 0) or (payoffs[i] >= 0 and payoffs[i+1] < 0):
                x1, x2 = spots[i], spots[i+1]
                y1, y2 = payoffs[i], payoffs[i+1]
                if y2 != y1:
                    be = x1 - y1 * (x2 - x1) / (y2 - y1)
                    breakevens.append(round(be, 2))
        return breakevens

class StrategyLibrary:
    def __init__(self):
        self.strategies = self._build_all_strategies()

    def _build_all_strategies(self) -> Dict[str, StrategyInfo]:
        strategies = {}

        strategies['covered_call'] = StrategyInfo(
            name="Covered Call",
            category="Income",
            legs=[],
            description="Sell OTM call against long stock",
            best_for="Income in sideways market",
            risk_profile="Limited upside, unlimited downside on stock",
            max_profit=None, max_loss=None,
            breakeven=[],
            bullish=False, bearish=False, neutral=True, volatility_trade=False
        )

        strategies['covered_put'] = StrategyInfo(
            name="Covered Put",
            category="Income",
            legs=[],
            description="Sell OTM put against short stock",
            best_for="Income when expecting stock to decline",
            risk_profile="Limited downside, unlimited upside on stock",
            max_profit=None, max_loss=None,
            breakeven=[],
            bullish=False, bearish=True, neutral=False, volatility_trade=False
        )

        strategies['protective_put'] = StrategyInfo(
            name="Protective Put",
            category="Hedging",
            legs=[],
            description="Buy put to protect long stock position",
            best_for="Protecting against downside",
            risk_profile="Limited downside, pays premium",
            max_profit=None, max_loss=None,
            breakeven=[],
            bullish=True, bearish=False, neutral=False, volatility_trade=True
        )

        strategies['protective_call'] = StrategyInfo(
            name="Protective Call",
            category="Hedging",
            legs=[],
            description="Buy call to protect short stock position",
            best_for="Protecting against upside in short position",
            risk_profile="Limited upside, pays premium",
            max_profit=None, max_loss=None,
            breakeven=[],
            bullish=False, bearish=True, neutral=False, volatility_trade=True
        )

        strategies['bull_call_spread'] = StrategyInfo(
            name="Bull Call Spread",
            category="Vertical Spread",
            legs=[],
            description="Buy lower strike call, sell higher strike call",
            best_for="Moderate bullish view",
            risk_profile="Limited risk, limited reward",
            max_profit=None, max_loss=None,
            breakeven=[],
            bullish=True, bearish=False, neutral=False, volatility_trade=False
        )

        strategies['bull_put_spread'] = StrategyInfo(
            name="Bull Put Spread",
            category="Vertical Spread",
            legs=[],
            description="Sell higher strike put, buy lower strike put",
            best_for="Credit spread on bullish view",
            risk_profile="Limited risk, limited reward",
            max_profit=None, max_loss=None,
            breakeven=[],
            bullish=True, bearish=False, neutral=False, volatility_trade=False
        )

        strategies['bear_call_spread'] = StrategyInfo(
            name="Bear Call Spread",
            category="Vertical Spread",
            legs=[],
            description="Sell lower strike call, buy higher strike call",
            best_for="Moderate bearish view",
            risk_profile="Limited risk, limited reward",
            max_profit=None, max_loss=None,
            breakeven=[],
            bullish=False, bearish=True, neutral=False, volatility_trade=False
        )

        strategies['bear_put_spread'] = StrategyInfo(
            name="Bear Put Spread",
            category="Vertical Spread",
            legs=[],
            description="Buy higher strike put, sell lower strike put",
            best_for="Bearish view with defined risk",
            risk_profile="Limited risk, limited reward",
            max_profit=None, max_loss=None,
            breakeven=[],
            bullish=False, bearish=True, neutral=False, volatility_trade=False
        )

        strategies['long_synthetic_forward'] = StrategyInfo(
            name="Long Synthetic Forward",
            category="Synthetic",
            legs=[],
            description="Long call + short put at same strike",
            best_for="Mimicking long stock position",
            risk_profile="Unlimited upside and downside",
            max_profit=None, max_loss=None,
            breakeven=[],
            bullish=True, bearish=False, neutral=False, volatility_trade=False
        )

        strategies['short_synthetic_forward'] = StrategyInfo(
            name="Short Synthetic Forward",
            category="Synthetic",
            legs=[],
            description="Short call + long put at same strike",
            best_for="Mimicking short stock position",
            risk_profile="Unlimited upside and downside",
            max_profit=None, max_loss=None,
            breakeven=[],
            bullish=False, bearish=True, neutral=False, volatility_trade=False
        )

        strategies['long_combo'] = StrategyInfo(
            name="Long Combo",
            category="Synthetic",
            legs=[],
            description="Long call + short put, different strikes",
            best_for="Cheaper than stock with leverage",
            risk_profile="Unlimited upside, substantial downside",
            max_profit=None, max_loss=None,
            breakeven=[],
            bullish=True, bearish=False, neutral=False, volatility_trade=False
        )

        strategies['short_combo'] = StrategyInfo(
            name="Short Combo",
            category="Synthetic",
            legs=[],
            description="Short call + long put, different strikes",
            best_for="Short stock replacement",
            risk_profile="Unlimited upside risk, limited downside",
            max_profit=None, max_loss=None,
            breakeven=[],
            bullish=False, bearish=True, neutral=False, volatility_trade=False
        )

        strategies['bull_call_ladder'] = StrategyInfo(
            name="Bull Call Ladder",
            category="Ladder Spread",
            legs=[],
            description="Bull call spread + short higher call",
            best_for="Bullish when expecting limited upside",
            risk_profile="Limited downside, limited upside",
            max_profit=None, max_loss=None,
            breakeven=[],
            bullish=True, bearish=False, neutral=False, volatility_trade=False
        )

        strategies['bull_put_ladder'] = StrategyInfo(
            name="Bull Put Ladder",
            category="Ladder Spread",
            legs=[],
            description="Bull put spread + long lower put",
            best_for="Bullish with downside protection",
            risk_profile="Defined risk, potential for higher reward",
            max_profit=None, max_loss=None,
            breakeven=[],
            bullish=True, bearish=False, neutral=False, volatility_trade=False
        )

        strategies['bear_call_ladder'] = StrategyInfo(
            name="Bear Call Ladder",
            category="Ladder Spread",
            legs=[],
            description="Bear call spread + long higher call",
            best_for="Bearish when expecting limited downside",
            risk_profile="Limited upside, defined downside",
            max_profit=None, max_loss=None,
            breakeven=[],
            bullish=False, bearish=True, neutral=False, volatility_trade=False
        )

        strategies['bear_put_ladder'] = StrategyInfo(
            name="Bear Put Ladder",
            category="Ladder Spread",
            legs=[],
            description="Bear put spread + short lower put",
            best_for="Bearish with capped loss",
            risk_profile="Limited upside, limited downside",
            max_profit=None, max_loss=None,
            breakeven=[],
            bullish=False, bearish=True, neutral=False, volatility_trade=False
        )

        strategies['calendar_call_spread'] = StrategyInfo(
            name="Calendar Call Spread",
            category="Calendar",
            legs=[],
            description="Sell near-term call, buy longer-term call at same strike",
            best_for="Time decay differential",
            risk_profile="Limited risk, profit if near-term decays faster",
            max_profit=None, max_loss=None,
            breakeven=[],
            bullish=False, bearish=False, neutral=True, volatility_trade=True
        )

        strategies['calendar_put_spread'] = StrategyInfo(
            name="Calendar Put Spread",
            category="Calendar",
            legs=[],
            description="Sell near-term put, buy longer-term put at same strike",
            best_for="Time decay differential",
            risk_profile="Limited risk, profit if near-term decays faster",
            max_profit=None, max_loss=None,
            breakeven=[],
            bullish=False, bearish=False, neutral=True, volatility_trade=True
        )

        strategies['diagonal_call_spread'] = StrategyInfo(
            name="Diagonal Call Spread",
            category="Diagonal",
            legs=[],
            description="Buy longer-term lower strike call, sell near-term higher strike call",
            best_for="Balanced bullish with time decay benefit",
            risk_profile="Defined risk with time decay advantage",
            max_profit=None, max_loss=None,
            breakeven=[],
            bullish=True, bearish=False, neutral=False, volatility_trade=True
        )

        strategies['diagonal_put_spread'] = StrategyInfo(
            name="Diagonal Put Spread",
            category="Diagonal",
            legs=[],
            description="Buy longer-term higher strike put, sell near-term lower strike put",
            best_for="Bullish put strategy with time advantage",
            risk_profile="Defined risk with time decay advantage",
            max_profit=None, max_loss=None,
            breakeven=[],
            bullish=True, bearish=False, neutral=False, volatility_trade=True
        )

        strategies['long_straddle'] = StrategyInfo(
            name="Long Straddle",
            category="Volatility",
            legs=[],
            description="Buy ATM call + buy ATM put",
            best_for="Big moves in either direction",
            risk_profile="Limited loss (premium paid), unlimited profit potential",
            max_profit=None, max_loss=None,
            breakeven=[],
            bullish=False, bearish=False, neutral=False, volatility_trade=True
        )

        strategies['long_strangle'] = StrategyInfo(
            name="Long Strangle",
            category="Volatility",
            legs=[],
            description="Buy OTM call + buy OTM put",
            best_for="Cheaper volatility play than straddle",
            risk_profile="Limited loss, unlimited profit potential",
            max_profit=None, max_loss=None,
            breakeven=[],
            bullish=False, bearish=False, neutral=False, volatility_trade=True
        )

        strategies['long_guts'] = StrategyInfo(
            name="Long Guts",
            category="Volatility",
            legs=[],
            description="Buy ITM call + buy ITM put (ITM options)",
            best_for="Cheaper than straddle with similar payoff",
            risk_profile="Limited loss if both expire worthless",
            max_profit=None, max_loss=None,
            breakeven=[],
            bullish=False, bearish=False, neutral=False, volatility_trade=True
        )

        strategies['short_straddle'] = StrategyInfo(
            name="Short Straddle",
            category="Volatility",
            legs=[],
            description="Sell ATM call + sell ATM put",
            best_for="Range-bound market, collecting premium",
            risk_profile="Unlimited loss, limited profit (premium)",
            max_profit=None, max_loss=None,
            breakeven=[],
            bullish=False, bearish=False, neutral=True, volatility_trade=True
        )

        strategies['short_strangle'] = StrategyInfo(
            name="Short Strangle",
            category="Volatility",
            legs=[],
            description="Sell OTM call + sell OTM put",
            best_for="Wide range-bound, high premium collection",
            risk_profile="Unlimited loss, limited profit",
            max_profit=None, max_loss=None,
            breakeven=[],
            bullish=False, bearish=False, neutral=True, volatility_trade=True
        )

        strategies['short_guts'] = StrategyInfo(
            name="Short Guts",
            category="Volatility",
            legs=[],
            description="Sell ITM call + sell ITM put",
            best_for="Selling high IV in ITM options",
            risk_profile="Unlimited loss potential",
            max_profit=None, max_loss=None,
            breakeven=[],
            bullish=False, bearish=False, neutral=True, volatility_trade=True
        )

        strategies['long_call_synthetic_straddle'] = StrategyInfo(
            name="Long Call Synthetic Straddle",
            category="Synthetic",
            legs=[],
            description="Long stock + long put = long call position",
            best_for="Bullish volatility",
            risk_profile="Like long call",
            max_profit=None, max_loss=None,
            breakeven=[],
            bullish=True, bearish=False, neutral=False, volatility_trade=True
        )

        strategies['long_put_synthetic_straddle'] = StrategyInfo(
            name="Long Put Synthetic Straddle",
            category="Synthetic",
            legs=[],
            description="Short stock + long call = long put position",
            best_for="Bearish volatility",
            risk_profile="Like long put",
            max_profit=None, max_loss=None,
            breakeven=[],
            bullish=False, bearish=True, neutral=False, volatility_trade=True
        )

        strategies['short_call_synthetic_straddle'] = StrategyInfo(
            name="Short Call Synthetic Straddle",
            category="Synthetic",
            legs=[],
            description="Short stock + short put = short call position",
            best_for="Bearish volatility",
            risk_profile="Like short call",
            max_profit=None, max_loss=None,
            breakeven=[],
            bullish=False, bearish=True, neutral=False, volatility_trade=True
        )

        strategies['short_put_synthetic_straddle'] = StrategyInfo(
            name="Short Put Synthetic Straddle",
            category="Synthetic",
            legs=[],
            description="Long stock + short call = short put position",
            best_for="Bullish volatility",
            risk_profile="Like short put",
            max_profit=None, max_loss=None,
            breakeven=[],
            bullish=True, bearish=False, neutral=False, volatility_trade=True
        )

        strategies['covered_short_straddle'] = StrategyInfo(
            name="Covered Short Straddle",
            category="Income",
            legs=[],
            description="Long stock + short ATM call + short ATM put",
            best_for="Enhanced income in range-bound market",
            risk_profile="High risk, max profit at strike",
            max_profit=None, max_loss=None,
            breakeven=[],
            bullish=False, bearish=False, neutral=True, volatility_trade=True
        )

        strategies['covered_short_strangle'] = StrategyInfo(
            name="Covered Short Strangle",
            category="Income",
            legs=[],
            description="Long stock + short OTM call + short OTM put",
            best_for="Wide range income strategy",
            risk_profile="High risk, profit if stock stays in range",
            max_profit=None, max_loss=None,
            breakeven=[],
            bullish=False, bearish=False, neutral=True, volatility_trade=True
        )

        strategies['strap'] = StrategyInfo(
            name="Strap",
            category="Volatility",
            legs=[],
            description="Long 2 calls + long 1 put at same strike",
            best_for="Expecting bullish move with volatility",
            risk_profile="Unlimited upside, limited downside",
            max_profit=None, max_loss=None,
            breakeven=[],
            bullish=True, bearish=False, neutral=False, volatility_trade=True
        )

        strategies['strip'] = StrategyInfo(
            name="Strip",
            category="Volatility",
            legs=[],
            description="Long 1 call + long 2 puts at same strike",
            best_for="Expecting bearish move with volatility",
            risk_profile="Unlimited downside, limited upside",
            max_profit=None, max_loss=None,
            breakeven=[],
            bullish=False, bearish=True, neutral=False, volatility_trade=True
        )

        strategies['call_ratio_backspread'] = StrategyInfo(
            name="Call Ratio Backspread",
            category="Ratio",
            legs=[],
            description="Sell 1 ITM call, buy 2 OTM calls",
            best_for="Bullish move with no downside risk",
            risk_profile="Limited loss, unlimited upside",
            max_profit=None, max_loss=None,
            breakeven=[],
            bullish=True, bearish=False, neutral=False, volatility_trade=True
        )

        strategies['put_ratio_backspread'] = StrategyInfo(
            name="Put Ratio Backspread",
            category="Ratio",
            legs=[],
            description="Sell 1 ITM put, buy 2 OTM puts",
            best_for="Bearish move with no upside risk",
            risk_profile="Limited loss, unlimited downside",
            max_profit=None, max_loss=None,
            breakeven=[],
            bullish=False, bearish=True, neutral=False, volatility_trade=True
        )

        strategies['ratio_call_spread'] = StrategyInfo(
            name="Ratio Call Spread",
            category="Ratio",
            legs=[],
            description="Buy call at lower strike, sell 2 calls at higher strike",
            best_for="Bullish with capped upside",
            risk_profile="Credit received, but uncapped loss above upper strike",
            max_profit=None, max_loss=None,
            breakeven=[],
            bullish=True, bearish=False, neutral=False, volatility_trade=True
        )

        strategies['ratio_put_spread'] = StrategyInfo(
            name="Ratio Put Spread",
            category="Ratio",
            legs=[],
            description="Buy put at higher strike, sell 2 puts at lower strike",
            best_for="Bearish with capped downside",
            risk_profile="Credit received, but uncapped loss below lower strike",
            max_profit=None, max_loss=None,
            breakeven=[],
            bullish=False, bearish=True, neutral=False, volatility_trade=True
        )

        strategies['long_call_butterfly'] = StrategyInfo(
            name="Long Call Butterfly",
            category="Butterfly",
            legs=[],
            description="Buy 1 ITM call, sell 2 ATM calls, buy 1 OTM call",
            best_for="Minimal move expected (low risk/reward)",
            risk_profile="Limited loss, limited profit",
            max_profit=None, max_loss=None,
            breakeven=[],
            bullish=False, bearish=False, neutral=True, volatility_trade=True
        )

        strategies['modified_call_butterfly'] = StrategyInfo(
            name="Modified Call Butterfly",
            category="Butterfly",
            legs=[],
            description="Uneven wing butterfly with calls",
            best_for="Specific price target",
            risk_profile="Defined risk, asymmetric payoff",
            max_profit=None, max_loss=None,
            breakeven=[],
            bullish=False, bearish=False, neutral=True, volatility_trade=True
        )

        strategies['long_put_butterfly'] = StrategyInfo(
            name="Long Put Butterfly",
            category="Butterfly",
            legs=[],
            description="Buy 1 ITM put, sell 2 ATM puts, buy 1 OTM put",
            best_for="Minimal move expected (low risk/reward)",
            risk_profile="Limited loss, limited profit",
            max_profit=None, max_loss=None,
            breakeven=[],
            bullish=False, bearish=False, neutral=True, volatility_trade=True
        )

        strategies['modified_put_butterfly'] = StrategyInfo(
            name="Modified Put Butterfly",
            category="Butterfly",
            legs=[],
            description="Uneven wing butterfly with puts",
            best_for="Specific price target",
            risk_profile="Defined risk, asymmetric payoff",
            max_profit=None, max_loss=None,
            breakeven=[],
            bullish=False, bearish=False, neutral=True, volatility_trade=True
        )

        strategies['short_call_butterfly'] = StrategyInfo(
            name="Short Call Butterfly",
            category="Butterfly",
            legs=[],
            description="Reverse of long call butterfly",
            best_for="Expecting big move",
            risk_profile="Limited profit, unlimited loss",
            max_profit=None, max_loss=None,
            breakeven=[],
            bullish=False, bearish=False, neutral=False, volatility_trade=True
        )

        strategies['short_put_butterfly'] = StrategyInfo(
            name="Short Put Butterfly",
            category="Butterfly",
            legs=[],
            description="Reverse of long put butterfly",
            best_for="Expecting big move",
            risk_profile="Limited profit, unlimited loss",
            max_profit=None, max_loss=None,
            breakeven=[],
            bullish=False, bearish=False, neutral=False, volatility_trade=True
        )

        strategies['long_iron_butterfly'] = StrategyInfo(
            name="Long Iron Butterfly",
            category="Iron",
            legs=[],
            description="Short straddle + long strangle for protection",
            best_for="Range-bound with volatility spike expected",
            risk_profile="Limited loss, limited profit",
            max_profit=None, max_loss=None,
            breakeven=[],
            bullish=False, bearish=False, neutral=True, volatility_trade=True
        )

        strategies['short_iron_butterfly'] = StrategyInfo(
            name="Short Iron Butterfly",
            category="Iron",
            legs=[],
            description="Long straddle + short strangle",
            best_for="Big move expected, sell wings",
            risk_profile="Limited profit, wide loss zones",
            max_profit=None, max_loss=None,
            breakeven=[],
            bullish=False, bearish=False, neutral=False, volatility_trade=True
        )

        strategies['long_call_condor'] = StrategyInfo(
            name="Long Call Condor",
            category="Condor",
            legs=[],
            description="Bull call spread + bear call spread combined",
            best_for="Wider range than butterfly",
            risk_profile="Limited loss, limited profit",
            max_profit=None, max_loss=None,
            breakeven=[],
            bullish=False, bearish=False, neutral=True, volatility_trade=True
        )

        strategies['long_put_condor'] = StrategyInfo(
            name="Long Put Condor",
            category="Condor",
            legs=[],
            description="Bull put spread + bear put spread combined",
            best_for="Wider range with defined risk",
            risk_profile="Limited loss, limited profit",
            max_profit=None, max_loss=None,
            breakeven=[],
            bullish=False, bearish=False, neutral=True, volatility_trade=True
        )

        strategies['short_call_condor'] = StrategyInfo(
            name="Short Call Condor",
            category="Condor",
            legs=[],
            description="Reverse of long call condor",
            best_for="Big move expected",
            risk_profile="Limited profit, wide loss",
            max_profit=None, max_loss=None,
            breakeven=[],
            bullish=False, bearish=False, neutral=False, volatility_trade=True
        )

        strategies['short_put_condor'] = StrategyInfo(
            name="Short Put Condor",
            category="Condor",
            legs=[],
            description="Reverse of long put condor",
            best_for="Big move expected",
            risk_profile="Limited profit, wide loss",
            max_profit=None, max_loss=None,
            breakeven=[],
            bullish=False, bearish=False, neutral=False, volatility_trade=True
        )

        strategies['long_iron_condor'] = StrategyInfo(
            name="Long Iron Condor",
            category="Iron",
            legs=[],
            description="Short put spread + short call spread",
            best_for="Wide range-bound, premium collection",
            risk_profile="Limited loss, limited profit",
            max_profit=None, max_loss=None,
            breakeven=[],
            bullish=False, bearish=False, neutral=True, volatility_trade=True
        )

        strategies['short_iron_condor'] = StrategyInfo(
            name="Short Iron Condor",
            category="Iron",
            legs=[],
            description="Long put spread + long call spread",
            best_for="Big move expected, buy wings",
            risk_profile="Wide loss zones",
            max_profit=None, max_loss=None,
            breakeven=[],
            bullish=False, bearish=False, neutral=False, volatility_trade=True
        )

        strategies['long_box'] = StrategyInfo(
            name="Long Box",
            category="Box",
            legs=[],
            description="Bull call spread + bull put spread (same strikes)",
            best_for="Risk-free arbitrage if mispriced",
            risk_profile="Risk-free (but costs capital)",
            max_profit=None, max_loss=None,
            breakeven=[],
            bullish=True, bearish=True, neutral=True, volatility_trade=False
        )

        strategies['collar'] = StrategyInfo(
            name="Collar",
            category="Hedging",
            legs=[],
            description="Protective put + covered call",
            best_for="Protected stock position with zero cost",
            risk_profile="Protected downside, capped upside",
            max_profit=None, max_loss=None,
            breakeven=[],
            bullish=True, bearish=False, neutral=True, volatility_trade=True
        )

        strategies['bullish_short_seagull'] = StrategyInfo(
            name="Bullish Short Seagull",
            category="Seagull",
            legs=[],
            description="Bull put spread + short OTM call",
            best_for="Bullish with limited downside",
            risk_profile="Limited risk, capped upside",
            max_profit=None, max_loss=None,
            breakeven=[],
            bullish=True, bearish=False, neutral=False, volatility_trade=True
        )

        strategies['bearish_long_seagull'] = StrategyInfo(
            name="Bearish Long Seagull",
            category="Seagull",
            legs=[],
            description="Bear call spread + long OTM call",
            best_for="Bearish with limited risk",
            risk_profile="Limited loss, profit if stock falls",
            max_profit=None, max_loss=None,
            breakeven=[],
            bullish=False, bearish=True, neutral=False, volatility_trade=True
        )

        strategies['bearish_short_seagull'] = StrategyInfo(
            name="Bearish Short Seagull",
            category="Seagull",
            legs=[],
            description="Bear call spread + short OTM put",
            best_for="Bearish with upside buffer",
            risk_profile="Limited risk, profit if stock falls",
            max_profit=None, max_loss=None,
            breakeven=[],
            bullish=False, bearish=True, neutral=False, volatility_trade=True
        )

        strategies['bullish_long_seagull'] = StrategyInfo(
            name="Bullish Long Seagull",
            category="Seagull",
            legs=[],
            description="Bull put spread + long OTM call",
            best_for="Bullish with floor protection",
            risk_profile="Limited loss, capped profit",
            max_profit=None, max_loss=None,
            breakeven=[],
            bullish=True, bearish=False, neutral=False, volatility_trade=True
        )

        return strategies

    def get_strategy(self, key: str) -> Optional[StrategyInfo]:
        return self.strategies.get(key)

    def get_all_strategies(self) -> Dict[str, StrategyInfo]:
        return self.strategies

    def get_strategies_by_category(self, category: str) -> Dict[str, StrategyInfo]:
        return {k: v for k, v in self.strategies.items() if v.category == category}

    def get_strategies_by_regime(self, regime: MarketRegime) -> List[str]:
        regime_map = {
            MarketRegime.BULL_LOW_VOL: ['bull_call_spread', 'bull_put_spread', 'bullish_long_seagull', 'diagonal_call_spread'],
            MarketRegime.BULL_HIGH_VOL: ['bull_call_spread', 'protective_put', 'long_combo', 'strap'],
            MarketRegime.BEAR_LOW_VOL: ['bear_call_spread', 'bear_put_spread', 'bearish_long_seagull', 'diagonal_put_spread'],
            MarketRegime.BEAR_HIGH_VOL: ['bear_put_spread', 'protective_call', 'short_combo', 'strip'],
            MarketRegime.NEUTRAL_LOW_VOL: ['short_straddle', 'short_strangle', 'long_iron_condor', 'covered_short_straddle'],
            MarketRegime.NEUTRAL_HIGH_VOL: ['short_straddle', 'covered_short_strangle', 'calendar_call_spread', 'calendar_put_spread'],
            MarketRegime.HIGH_IV_PREMIUM: ['short_straddle', 'short_strangle', 'bear_call_spread', 'put_ratio_backspread'],
            MarketRegime.LOW_IV_DISCOUNT: ['long_straddle', 'long_strangle', 'call_ratio_backspread', 'protective_put'],
            MarketRegime.SKEW_BEARISH: ['protective_put', 'bear_put_spread', 'strip', 'long_put_butterfly'],
            MarketRegime.SKEW_BULLISH: ['covered_call', 'bull_call_spread', 'strap', 'long_call_butterfly']
        }
        return regime_map.get(regime, [])

class StrategyRecommender:
    def __init__(self):
        self.library = StrategyLibrary()

    def classify_regime(self, spot: float, strikes: List[float], ivs: List[float], 
                        atm_iv: float, iv_hist: float, trend: str, iv_rank: float) -> Tuple[MarketRegime, Dict]:
        
        vol_level = "high" if atm_iv > 25 else "low"
        
        skew = 0
        if len(strikes) > 5 and len(ivs) > 5:
            otm_ivs = [iv for s, iv in zip(strikes, ivs) if s > spot]
            itm_ivs = [iv for s, iv in zip(strikes, ivs) if s < spot]
            if otm_ivs and itm_ivs:
                skew = np.mean(otm_ivs) - np.mean(itm_ivs)
        
        regime_score = 0
        
        if trend == "bullish":
            regime_score += 2
        elif trend == "bearish":
            regime_score -= 2
        
        if iv_rank > 70:
            regime_score += 1
        elif iv_rank < 30:
            regime_score -= 1
        
        if skew > 3:
            regime_score -= 1
        elif skew < -3:
            regime_score += 1
        
        if regime_score >= 3:
            regime = MarketRegime.BULL_HIGH_VOL if vol_level == "high" else MarketRegime.BULL_LOW_VOL
        elif regime_score <= -3:
            regime = MarketRegime.BEAR_HIGH_VOL if vol_level == "high" else MarketRegime.BEAR_LOW_VOL
        elif abs(regime_score) <= 1:
            regime = MarketRegime.NEUTRAL_HIGH_VOL if vol_level == "high" else MarketRegime.NEUTRAL_LOW_VOL
        else:
            regime = MarketRegime.NEUTRAL_HIGH_VOL if vol_level == "high" else MarketRegime.NEUTRAL_LOW_VOL
        
        if iv_rank > 80:
            regime = MarketRegime.HIGH_IV_PREMIUM
        elif iv_rank < 20:
            regime = MarketRegime.LOW_IV_DISCOUNT
        
        if skew > 5:
            regime = MarketRegime.SKEW_BEARISH
        elif skew < -5:
            regime = MarketRegime.SKEW_BULLISH
        
        context = {
            "vol_level": vol_level,
            "skew": round(skew, 2),
            "iv_rank": round(iv_rank, 1),
            "trend": trend,
            "atm_iv": atm_iv
        }
        
        return regime, context

    def rank_strategies(self, regime: MarketRegime, spot: float, atm_iv: float, dte: float,
                        risk_tolerance: str = "medium") -> List[Dict]:
        
        regime_strategies = self.library.get_strategies_by_regime(regime)
        
        if not regime_strategies:
            regime_strategies = ['bull_call_spread', 'bear_put_spread', 'long_straddle', 'iron_condor']
        
        ranked = []
        
        for strat_key in regime_strategies:
            strat = self.library.get_strategy(strat_key)
            if not strat:
                continue
            
            score = 100
            
            if risk_tolerance == "low":
                if strat.max_loss is None:
                    score -= 30
                score += 20 if "spread" in strat.category.lower() else -20
            elif risk_tolerance == "high":
                if strat.max_loss is None:
                    score += 10
                score += 10 if "straddle" in strat.name.lower() or "strangle" in strat.name.lower() else -10
            
            if strat.volatility_trade and (regime == MarketRegime.HIGH_IV_PREMIUM or regime == MarketRegime.LOW_IV_DISCOUNT):
                score += 20
            
            ranked.append({
                "strategy_key": strat_key,
                "name": strat.name,
                "category": strat.category,
                "description": strat.description,
                "best_for": strat.best_for,
                "risk_profile": strat.risk_profile,
                "score": score,
                "trend_fit": "bullish" if strat.bullish else "bearish" if strat.bearish else "neutral"
            })
        
        ranked.sort(key=lambda x: x['score'], reverse=True)
        
        for i, r in enumerate(ranked):
            r['rank'] = i + 1
        
        return ranked

    def recommend(self, asset: str, spot: float, strikes: List[float], ivs: List[float],
                  atm_iv: float, iv_hist: float, trend: str, iv_rank: float,
                  dte: float = 7, risk_tolerance: str = "medium") -> Dict:
        
        regime, context = self.classify_regime(spot, strikes, ivs, atm_iv, iv_hist, trend, iv_rank)
        
        ranked = self.rank_strategies(regime, spot, atm_iv, dte, risk_tolerance)
        
        return {
            "asset": asset,
            "spot": spot,
            "atm_iv": atm_iv,
            "dte": dte,
            "regime": regime.value,
            "regime_description": self._describe_regime(regime),
            "market_context": context,
            "top_recommendations": ranked[:5],
            "all_strategies": ranked
        }

    def _describe_regime(self, regime: MarketRegime) -> str:
        descriptions = {
            MarketRegime.BULL_LOW_VOL: "Bullish market with low volatility - ideal for debit spreads",
            MarketRegime.BULL_HIGH_VOL: "Bullish market with high volatility - consider hedging",
            MarketRegime.BEAR_LOW_VOL: "Bearish market with low volatility - credit spreads preferred",
            MarketRegime.BEAR_HIGH_VOL: "Bearish market with high volatility - protective strategies",
            MarketRegime.NEUTRAL_LOW_VOL: "Range-bound with low volatility - sell premium strategies",
            MarketRegime.NEUTRAL_HIGH_VOL: "Range-bound with high volatility - sell premium, calendar spreads",
            MarketRegime.HIGH_IV_PREMIUM: "IV Rank high (>80) - sell premium strategies favored",
            MarketRegime.LOW_IV_DISCOUNT: "IV Rank low (<20) - buy premium strategies favored",
            MarketRegime.SKEW_BEARISH: "Put skew elevated - expect downside, protective strategies",
            MarketRegime.SKEW_BULLISH: "Call skew elevated - expect upside, bullish strategies"
        }
        return descriptions.get(regime, "Mixed signals")
