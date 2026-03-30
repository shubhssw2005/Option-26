import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import json
import os

class DeepStrategyRecommender:
    def __init__(self):
        self.strategy_vectors = self._build_strategy_vectors()
        self.condition_embeddings = self._build_condition_embeddings()
        
    def _build_strategy_vectors(self) -> Dict[str, np.ndarray]:
        """Build feature vectors for each strategy based on characteristics"""
        vectors = {}
        
        strategy_features = {
            'covered_call': {'bullish': 0.3, 'bearish': -0.5, 'neutral': 0.7, 'vol_trade': 0.3, 'income': 0.9, 'risk': -0.3},
            'covered_put': {'bullish': -0.5, 'bearish': 0.6, 'neutral': 0.7, 'vol_trade': 0.3, 'income': 0.9, 'risk': -0.3},
            'protective_put': {'bullish': 0.8, 'bearish': -0.6, 'neutral': 0.5, 'vol_trade': 0.8, 'income': 0.0, 'risk': 0.2},
            'protective_call': {'bullish': -0.6, 'bearish': 0.8, 'neutral': 0.5, 'vol_trade': 0.8, 'income': 0.0, 'risk': 0.2},
            'bull_call_spread': {'bullish': 0.9, 'bearish': -0.8, 'neutral': 0.3, 'vol_trade': 0.1, 'income': 0.4, 'risk': -0.2},
            'bull_put_spread': {'bullish': 0.9, 'bearish': -0.8, 'neutral': 0.5, 'vol_trade': 0.1, 'income': 0.6, 'risk': -0.2},
            'bear_call_spread': {'bullish': -0.8, 'bearish': 0.9, 'neutral': 0.3, 'vol_trade': 0.1, 'income': 0.4, 'risk': -0.2},
            'bear_put_spread': {'bullish': -0.8, 'bearish': 0.9, 'neutral': 0.3, 'vol_trade': 0.1, 'income': 0.4, 'risk': -0.2},
            'long_synthetic_forward': {'bullish': 0.95, 'bearish': -0.95, 'neutral': 0.0, 'vol_trade': 0.3, 'income': 0.0, 'risk': 0.9},
            'short_synthetic_forward': {'bullish': -0.95, 'bearish': 0.95, 'neutral': 0.0, 'vol_trade': 0.3, 'income': 0.0, 'risk': 0.9},
            'long_combo': {'bullish': 0.85, 'bearish': -0.7, 'neutral': 0.0, 'vol_trade': 0.4, 'income': 0.2, 'risk': 0.7},
            'short_combo': {'bullish': -0.7, 'bearish': 0.85, 'neutral': 0.0, 'vol_trade': 0.4, 'income': 0.2, 'risk': 0.7},
            'bull_call_ladder': {'bullish': 0.7, 'bearish': -0.5, 'neutral': 0.5, 'vol_trade': 0.3, 'income': 0.5, 'risk': -0.1},
            'bull_put_ladder': {'bullish': 0.8, 'bearish': -0.6, 'neutral': 0.4, 'vol_trade': 0.3, 'income': 0.4, 'risk': -0.1},
            'bear_call_ladder': {'bullish': -0.5, 'bearish': 0.7, 'neutral': 0.5, 'vol_trade': 0.3, 'income': 0.5, 'risk': -0.1},
            'bear_put_ladder': {'bullish': -0.6, 'bearish': 0.8, 'neutral': 0.5, 'vol_trade': 0.3, 'income': 0.4, 'risk': -0.1},
            'calendar_call_spread': {'bullish': 0.4, 'bearish': -0.3, 'neutral': 0.8, 'vol_trade': 0.9, 'income': 0.3, 'risk': -0.2},
            'calendar_put_spread': {'bullish': 0.4, 'bearish': -0.3, 'neutral': 0.8, 'vol_trade': 0.9, 'income': 0.3, 'risk': -0.2},
            'diagonal_call_spread': {'bullish': 0.8, 'bearish': -0.4, 'neutral': 0.5, 'vol_trade': 0.8, 'income': 0.3, 'risk': -0.1},
            'diagonal_put_spread': {'bullish': 0.8, 'bearish': -0.4, 'neutral': 0.5, 'vol_trade': 0.8, 'income': 0.3, 'risk': -0.1},
            'long_straddle': {'bullish': 0.3, 'bearish': 0.3, 'neutral': 0.0, 'vol_trade': 0.95, 'income': 0.0, 'risk': 0.6},
            'long_strangle': {'bullish': 0.2, 'bearish': 0.2, 'neutral': 0.0, 'vol_trade': 0.95, 'income': 0.0, 'risk': 0.5},
            'long_guts': {'bullish': 0.3, 'bearish': 0.3, 'neutral': 0.0, 'vol_trade': 0.9, 'income': 0.0, 'risk': 0.6},
            'short_straddle': {'bullish': -0.3, 'bearish': -0.3, 'neutral': 0.9, 'vol_trade': 0.9, 'income': 0.8, 'risk': 0.8},
            'short_strangle': {'bullish': -0.2, 'bearish': -0.2, 'neutral': 0.9, 'vol_trade': 0.9, 'income': 0.8, 'risk': 0.7},
            'short_guts': {'bullish': -0.3, 'bearish': -0.3, 'neutral': 0.9, 'vol_trade': 0.9, 'income': 0.8, 'risk': 0.8},
            'long_call_synthetic_straddle': {'bullish': 0.7, 'bearish': -0.4, 'neutral': 0.2, 'vol_trade': 0.8, 'income': 0.0, 'risk': 0.7},
            'long_put_synthetic_straddle': {'bullish': -0.4, 'bearish': 0.7, 'neutral': 0.2, 'vol_trade': 0.8, 'income': 0.0, 'risk': 0.7},
            'short_call_synthetic_straddle': {'bullish': -0.4, 'bearish': 0.7, 'neutral': 0.2, 'vol_trade': 0.8, 'income': 0.0, 'risk': 0.7},
            'short_put_synthetic_straddle': {'bullish': 0.7, 'bearish': -0.4, 'neutral': 0.2, 'vol_trade': 0.8, 'income': 0.0, 'risk': 0.7},
            'covered_short_straddle': {'bullish': -0.3, 'bearish': -0.3, 'neutral': 0.8, 'vol_trade': 0.7, 'income': 0.9, 'risk': 0.5},
            'covered_short_strangle': {'bullish': -0.2, 'bearish': -0.2, 'neutral': 0.9, 'vol_trade': 0.7, 'income': 0.9, 'risk': 0.4},
            'strap': {'bullish': 0.7, 'bearish': 0.3, 'neutral': 0.0, 'vol_trade': 0.9, 'income': 0.0, 'risk': 0.5},
            'strip': {'bullish': 0.3, 'bearish': 0.7, 'neutral': 0.0, 'vol_trade': 0.9, 'income': 0.0, 'risk': 0.5},
            'call_ratio_backspread': {'bullish': 0.9, 'bearish': -0.3, 'neutral': 0.2, 'vol_trade': 0.8, 'income': 0.3, 'risk': 0.2},
            'put_ratio_backspread': {'bullish': -0.3, 'bearish': 0.9, 'neutral': 0.2, 'vol_trade': 0.8, 'income': 0.3, 'risk': 0.2},
            'ratio_call_spread': {'bullish': 0.8, 'bearish': -0.4, 'neutral': 0.3, 'vol_trade': 0.6, 'income': 0.5, 'risk': 0.3},
            'ratio_put_spread': {'bullish': -0.4, 'bearish': 0.8, 'neutral': 0.3, 'vol_trade': 0.6, 'income': 0.5, 'risk': 0.3},
            'long_call_butterfly': {'bullish': 0.3, 'bearish': -0.3, 'neutral': 0.95, 'vol_trade': 0.7, 'income': 0.2, 'risk': -0.3},
            'modified_call_butterfly': {'bullish': 0.5, 'bearish': -0.3, 'neutral': 0.8, 'vol_trade': 0.7, 'income': 0.3, 'risk': -0.2},
            'long_put_butterfly': {'bullish': -0.3, 'bearish': 0.3, 'neutral': 0.95, 'vol_trade': 0.7, 'income': 0.2, 'risk': -0.3},
            'modified_put_butterfly': {'bullish': -0.3, 'bearish': 0.5, 'neutral': 0.8, 'vol_trade': 0.7, 'income': 0.3, 'risk': -0.2},
            'short_call_butterfly': {'bullish': 0.4, 'bearish': 0.4, 'neutral': -0.5, 'vol_trade': 0.7, 'income': 0.4, 'risk': 0.6},
            'short_put_butterfly': {'bullish': 0.4, 'bearish': 0.4, 'neutral': -0.5, 'vol_trade': 0.7, 'income': 0.4, 'risk': 0.6},
            'long_iron_butterfly': {'bullish': 0.2, 'bearish': -0.2, 'neutral': 0.9, 'vol_trade': 0.8, 'income': 0.4, 'risk': -0.2},
            'short_iron_butterfly': {'bullish': 0.4, 'bearish': 0.4, 'neutral': -0.5, 'vol_trade': 0.8, 'income': 0.3, 'risk': 0.6},
            'long_call_condor': {'bullish': 0.3, 'bearish': -0.3, 'neutral': 0.9, 'vol_trade': 0.7, 'income': 0.3, 'risk': -0.3},
            'long_put_condor': {'bullish': -0.3, 'bearish': 0.3, 'neutral': 0.9, 'vol_trade': 0.7, 'income': 0.3, 'risk': -0.3},
            'short_call_condor': {'bullish': 0.4, 'bearish': 0.4, 'neutral': -0.5, 'vol_trade': 0.7, 'income': 0.4, 'risk': 0.6},
            'short_put_condor': {'bullish': 0.4, 'bearish': 0.4, 'neutral': -0.5, 'vol_trade': 0.7, 'income': 0.4, 'risk': 0.6},
            'long_iron_condor': {'bullish': 0.2, 'bearish': -0.2, 'neutral': 0.9, 'vol_trade': 0.8, 'income': 0.5, 'risk': -0.2},
            'short_iron_condor': {'bullish': 0.4, 'bearish': 0.4, 'neutral': -0.5, 'vol_trade': 0.8, 'income': 0.3, 'risk': 0.6},
            'long_box': {'bullish': 0.5, 'bearish': 0.5, 'neutral': 0.95, 'vol_trade': 0.1, 'income': 0.0, 'risk': -0.1},
            'collar': {'bullish': 0.6, 'bearish': -0.4, 'neutral': 0.7, 'vol_trade': 0.5, 'income': 0.5, 'risk': -0.1},
            'bullish_short_seagull': {'bullish': 0.8, 'bearish': -0.5, 'neutral': 0.3, 'vol_trade': 0.6, 'income': 0.5, 'risk': 0.0},
            'bearish_long_seagull': {'bullish': -0.5, 'bearish': 0.8, 'neutral': 0.3, 'vol_trade': 0.6, 'income': 0.3, 'risk': 0.1},
            'bearish_short_seagull': {'bullish': -0.5, 'bearish': 0.7, 'neutral': 0.4, 'vol_trade': 0.6, 'income': 0.6, 'risk': 0.0},
            'bullish_long_seagull': {'bullish': 0.7, 'bearish': -0.5, 'neutral': 0.3, 'vol_trade': 0.6, 'income': 0.3, 'risk': 0.1}
        }
        
        for strat, features in strategy_features.items():
            vec = np.array([
                features['bullish'],
                features['bearish'],
                features['neutral'],
                features['vol_trade'],
                features['income'],
                features['risk']
            ])
            vectors[strat] = vec
            
        return vectors
    
    def _build_condition_embeddings(self) -> Dict[str, np.ndarray]:
        """Build embeddings for different market conditions"""
        return {
            'strong_bull_low_vol': np.array([0.95, 0.95, 0.0, 0.2, 0.5, 0.0]),
            'strong_bull_high_vol': np.array([0.9, 0.9, 0.0, 0.8, 0.3, 0.3]),
            'mild_bull_low_vol': np.array([0.6, 0.8, 0.2, 0.3, 0.6, 0.0]),
            'mild_bull_high_vol': np.array([0.5, 0.7, 0.2, 0.7, 0.4, 0.2]),
            'strong_bear_low_vol': np.array([-0.95, -0.95, 0.0, 0.2, 0.5, 0.0]),
            'strong_bear_high_vol': np.array([-0.9, -0.9, 0.0, 0.8, 0.3, 0.3]),
            'mild_bear_low_vol': np.array([-0.6, -0.8, 0.2, 0.3, 0.6, 0.0]),
            'mild_bear_high_vol': np.array([-0.5, -0.7, 0.2, 0.7, 0.4, 0.2]),
            'neutral_low_vol': np.array([0.0, 0.0, 0.9, 0.3, 0.8, 0.0]),
            'neutral_high_vol': np.array([0.0, 0.0, 0.8, 0.9, 0.6, 0.2]),
            'high_iv_rank': np.array([0.2, 0.2, 0.7, 0.95, 0.7, 0.3]),
            'low_iv_rank': np.array([0.3, 0.3, 0.5, 0.1, 0.2, 0.4]),
            'high_put_skew': np.array([-0.3, 0.3, 0.5, 0.6, 0.3, 0.3]),
            'high_call_skew': np.array([0.3, -0.3, 0.5, 0.6, 0.3, 0.3])
        }
    
    def _compute_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors"""
        dot = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(dot / (norm1 * norm2))
    
    def predict_conditions(self, spot_change: float, iv_level: float, iv_rank: float, 
                          skew: float, dte: int, realized_vol: float) -> np.ndarray:
        """Predict market condition vector from features"""
        
        trend_score = np.clip(spot_change / 20.0, -1, 1)
        
        vol_score = 1.0 if iv_level > 30 else 0.0
        vol_score += 1.0 if iv_rank > 70 else (1.0 if iv_rank < 30 else 0.0)
        vol_score = np.clip(vol_score, 0, 1)
        
        skew_score = np.clip(skew / 5.0, -1, 1)
        
        dte_score = 1.0 if dte < 7 else (0.5 if dte < 14 else 0.0)
        
        cond_vec = np.array([
            trend_score * 0.4 + vol_score * 0.3 + skew_score * 0.1,
            -trend_score * 0.4 + vol_score * 0.3 - skew_score * 0.1,
            (1 - abs(trend_score)) * vol_score,
            vol_score * (1 - abs(trend_score)),
            dte_score * 0.5,
            0.0
        ])
        
        return cond_vec
    
    def recommend(self, spot_change: float = 0.0, iv_level: float = 25.0,
                   iv_rank: float = 50.0, skew: float = 0.0, dte: int = 7,
                   realized_vol: float = 20.0, risk_tolerance: str = "medium") -> List[Dict]:
        
        cond_vec = self.predict_conditions(spot_change, iv_level, iv_rank, skew, dte, realized_vol)
        
        strategy_scores = []
        
        for strat_name, strat_vec in self.strategy_vectors.items():
            
            base_score = self._compute_similarity(cond_vec, strat_vec)
            
            risk_adjustment = 0.0
            if risk_tolerance == "low":
                if strat_vec[5] > 0.5:
                    risk_adjustment -= 0.3
                if "spread" in strat_name or "butterfly" in strat_name or "condor" in strat_name:
                    risk_adjustment += 0.15
            elif risk_tolerance == "high":
                if strat_vec[5] < 0.0:
                    risk_adjustment += 0.1
                if "straddle" in strat_name or "strangle" in strat_name:
                    risk_adjustment += 0.15
            
            vol_adjustment = 0.0
            if iv_rank > 80:
                if strat_vec[3] > 0.7 and strat_vec[4] > 0.5:
                    vol_adjustment += 0.2
            elif iv_rank < 20:
                if strat_vec[3] > 0.7 and strat_vec[4] < 0.3:
                    vol_adjustment += 0.2
            
            dte_adjustment = 0.0
            if dte < 7:
                if "calendar" in strat_name:
                    dte_adjustment += 0.15
            elif dte > 30:
                if "diagonal" in strat_name:
                    dte_adjustment += 0.1
            
            final_score = base_score + risk_adjustment + vol_adjustment + dte_adjustment
            final_score = np.clip(final_score, -1, 1)
            
            strategy_scores.append({
                'strategy_key': strat_name,
                'score': round(final_score * 100, 1),
                'base_similarity': round(base_score * 100, 1)
            })
        
        strategy_scores.sort(key=lambda x: x['score'], reverse=True)
        
        for i, s in enumerate(strategy_scores):
            s['rank'] = i + 1
            
        return strategy_scores
    
    def get_condition_analysis(self, spot_change: float, iv_level: float, 
                               iv_rank: float, skew: float) -> Dict:
        """Provide human-readable analysis of current market conditions"""
        
        if spot_change > 1:
            trend = "Strong Bullish"
        elif spot_change > 0.3:
            trend = "Mild Bullish"
        elif spot_change < -1:
            trend = "Strong Bearish"
        elif spot_change < -0.3:
            trend = "Mild Bearish"
        else:
            trend = "Neutral"
        
        if iv_level > 35:
            vol_env = "High Volatility"
        elif iv_level < 20:
            vol_env = "Low Volatility"
        else:
            vol_env = "Normal Volatility"
        
        if iv_rank > 75:
            iv_status = "IV Rank High (Favor selling)"
        elif iv_rank < 25:
            iv_status = "IV Rank Low (Favor buying)"
        else:
            iv_status = "IV Rank Neutral"
        
        if skew > 5:
            skew_status = "High Put Skew (Bearish sentiment)"
        elif skew < -5:
            skew_status = "High Call Skew (Bullish sentiment)"
        else:
            skew_status = "Normal Skew"
        
        return {
            "trend": trend,
            "volatility_environment": vol_env,
            "iv_status": iv_status,
            "skew_analysis": skew_status,
            "overall_outlook": self._get_outlook(trend, vol_env, iv_rank)
        }
    
    def _get_outlook(self, trend: str, vol_env: str, iv_rank: float) -> str:
        if "Bull" in trend and "High" in vol_env:
            return "Caution: Bullish with high volatility - consider hedging with spreads"
        elif "Bear" in trend and "High" in vol_env:
            return "Opportunity: Bearish with high vol - protective strategies or sell premium"
        elif "Bull" in trend and "Low" in vol_env:
            return "Favorable: Bullish with low vol - buy spreads, sell covered calls"
        elif "Bear" in trend and "Low" in vol_env:
            return "Favorable: Bearish with low vol - bear spreads, sell covered puts"
        elif "Neutral" in trend and "High" in vol_env:
            return "Premium Selling Zone: High IV favors selling strangles/straddles"
        elif "Neutral" in trend and "Low" in vol_env:
            return "Wait Mode: Low vol, neutral - consider calendar spreads"
        elif iv_rank > 75:
            return "IV High: Best time to sell premium - iron condors, short strangles"
        elif iv_rank < 25:
            return "IV Low: Best time to buy premium - straddles, calendars"
        else:
            return "Balanced: Mixed signals - use defined-risk strategies"
