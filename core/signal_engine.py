import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
import ta

class SignalEngine:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Strategy parameters (matching Pine Script exactly)
        self.atr_length = 14
        self.fast_ema_length = 9
        self.slow_ema_length = 21
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9
        self.rsi_length = 14
        self.vwap_toggle = True
        self.cooldown_minutes = 30
        self.risk_pct = 1.0
        self.tp1_mult = 0.5
        self.tp2_mult = 1.0
        self.tp3_mult = 1.5
        self.adx_length = 14
        self.vol_rank_sma_len = 20
        
        # Strategy toggles
        self.enable_trap = True
        self.enable_smc = True
        self.enable_scalp = True
        
        # Thresholds
        self.adx_threshold = 25
        self.vol_rank_threshold = 50
        
        # Cooldown tracking
        self.last_signal_time = {}
    
    def calculate_ema(self, df: pd.DataFrame, length: int) -> pd.Series:
        """Calculate Exponential Moving Average"""
        return ta.trend.ema_indicator(df['close'], window=length)
    
    def calculate_macd(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD"""
        macd = ta.trend.MACD(df['close'], window_fast=self.macd_fast, window_slow=self.macd_slow, window_sign=self.macd_signal)
        return macd.macd(), macd.macd_signal(), macd.macd_diff()
    
    def calculate_rsi(self, df: pd.DataFrame) -> pd.Series:
        """Calculate RSI"""
        return ta.momentum.RSI(df['close'], window=self.rsi_length)
    
    def calculate_atr(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Average True Range"""
        return ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=self.atr_length).average_true_range()
    
    def calculate_adx(self, df: pd.DataFrame) -> pd.Series:
        """Calculate ADX"""
        return ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=self.adx_length).adx()
    
    def calculate_vwap(self, df: pd.DataFrame) -> pd.Series:
        """Calculate VWAP"""
        return ta.volume.VolumeWeightedAveragePrice(df['high'], df['low'], df['close'], df['volume']).volume_weighted_average_price()
    
    def calculate_volume_sma(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Volume SMA"""
        return ta.volume.volume_sma(df['volume'], window=self.vol_rank_sma_len)
    
    def calculate_volatility_rank(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Volatility Rank"""
        atr = self.calculate_atr(df)
        close_sma = ta.trend.sma_indicator(df['close'], window=self.vol_rank_sma_len)
        return (atr / close_sma) * 100
    
    def detect_liquidity_grab(self, df: pd.DataFrame) -> bool:
        """Detect liquidity grab (Trap strategy)"""
        if not self.enable_trap:
            return False
        
        vol_sma = self.calculate_volume_sma(df)
        if len(vol_sma) < 2:
            return False
        
        current_volume = df['volume'].iloc[-1]
        current_close = df['close'].iloc[-1]
        highest_20 = df['high'].rolling(window=20).max().iloc[-2]  # Previous bar
        
        return current_volume > vol_sma.iloc[-1] * 2 and current_close > highest_20
    
    def detect_order_block(self, df: pd.DataFrame) -> bool:
        """Detect order block (SMC strategy)"""
        if not self.enable_smc:
            return False
        
        if len(df) < 11:
            return False
        
        current_close = df['close'].iloc[-1]
        lowest_10_prev = df['low'].rolling(window=10).min().iloc[-2]  # Previous bar
        
        return current_close < lowest_10_prev
    
    def detect_fvg_gap(self, df: pd.DataFrame) -> bool:
        """Detect Fair Value Gap (SMC strategy)"""
        if not self.enable_smc:
            return False
        
        if len(df) < 2:
            return False
        
        atr = self.calculate_atr(df)
        prev_high = df['high'].iloc[-2]
        prev_low = df['low'].iloc[-2]
        
        return (prev_high - prev_low) > atr.iloc[-1]
    
    def detect_breaker(self, df: pd.DataFrame) -> bool:
        """Detect breaker (SMC strategy)"""
        if not self.enable_smc:
            return False
        
        if len(df) < 2:
            return False
        
        current_close = df['close'].iloc[-1]
        prev_high = df['high'].iloc[-2]
        
        return current_close > prev_high
    
    def detect_vwap_slope(self, df: pd.DataFrame) -> bool:
        """Detect VWAP slope (Scalping strategy)"""
        if not self.enable_scalp or not self.vwap_toggle:
            return False
        
        vwap = self.calculate_vwap(df)
        if len(vwap) < 2 or pd.isna(vwap.iloc[-1]) or pd.isna(vwap.iloc[-2]):
            return False
        
        # Calculate slope in degrees
        slope_rad = np.arctan((vwap.iloc[-1] - vwap.iloc[-2]) / df['close'].iloc[-1])
        slope_deg = np.degrees(slope_rad)
        
        return slope_deg > 30
    
    def detect_volume_spike(self, df: pd.DataFrame) -> bool:
        """Detect volume spike (Scalping strategy)"""
        if not self.enable_scalp:
            return False
        
        vol_sma = self.calculate_volume_sma(df)
        if len(vol_sma) < 1:
            return False
        
        current_volume = df['volume'].iloc[-1]
        return current_volume > vol_sma.iloc[-1] * 1.5
    
    def validate_ema_crossover(self, df: pd.DataFrame) -> bool:
        """Validate EMA crossover"""
        ema_fast = self.calculate_ema(df, self.fast_ema_length)
        ema_slow = self.calculate_ema(df, self.slow_ema_length)
        
        if len(ema_fast) < 2 or len(ema_slow) < 2:
            return False
        
        # Check for crossover
        return ema_fast.iloc[-1] > ema_slow.iloc[-1] and ema_fast.iloc[-2] <= ema_slow.iloc[-2]
    
    def validate_macd_crossover(self, df: pd.DataFrame) -> bool:
        """Validate MACD crossover"""
        macd_line, signal_line, _ = self.calculate_macd(df)
        
        if len(macd_line) < 2 or len(signal_line) < 2:
            return False
        
        # Check for crossover
        return macd_line.iloc[-1] > signal_line.iloc[-1] and macd_line.iloc[-2] <= signal_line.iloc[-2]
    
    def validate_rsi_reversal(self, df: pd.DataFrame) -> bool:
        """Validate RSI reversal"""
        rsi = self.calculate_rsi(df)
        
        if len(rsi) < 2:
            return False
        
        current_rsi = rsi.iloc[-1]
        prev_rsi = rsi.iloc[-2]
        
        # RSI oversold reversal
        if prev_rsi < 30 and current_rsi >= 30:
            return True
        
        # RSI overbought reversal
        if prev_rsi > 70 and current_rsi <= 70:
            return True
        
        return False
    
    def validate_volume(self, df: pd.DataFrame) -> bool:
        """Validate volume condition"""
        vol_sma = self.calculate_volume_sma(df)
        if len(vol_sma) < 1:
            return False
        
        current_volume = df['volume'].iloc[-1]
        return current_volume > vol_sma.iloc[-1] * 1.5
    
    def validate_adx(self, df: pd.DataFrame) -> bool:
        """Validate ADX condition"""
        adx = self.calculate_adx(df)
        if len(adx) < 1:
            return False
        
        return adx.iloc[-1] > self.adx_threshold
    
    def validate_volatility_rank(self, df: pd.DataFrame) -> bool:
        """Validate volatility rank condition"""
        vol_rank = self.calculate_volatility_rank(df)
        if len(vol_rank) < 1:
            return False
        
        return vol_rank.iloc[-1] > self.vol_rank_threshold
    
    def calculate_strategy_weights(self, df: pd.DataFrame) -> Tuple[float, float, float]:
        """Calculate strategy weights based on market conditions"""
        adx = self.calculate_adx(df)
        vol_rank = self.calculate_volatility_rank(df)
        
        if len(adx) < 1 or len(vol_rank) < 1:
            return 1.0, 1.0, 1.0
        
        if adx.iloc[-1] > self.adx_threshold and vol_rank.iloc[-1] > self.vol_rank_threshold:
            return 1.0, 1.5, 0.5  # wTrap, wSMC, wScalp
        else:
            return 1.0, 0.5, 1.5  # wTrap, wSMC, wScalp
    
    def calculate_trap_score(self, df: pd.DataFrame) -> float:
        """Calculate Trap strategy score"""
        liq_grab = self.detect_liquidity_grab(df)
        valid_ema = self.validate_ema_crossover(df)
        valid_vol = self.validate_volume(df)
        valid_adx = self.validate_adx(df)
        
        score = (1 if liq_grab else 0) + (1 if valid_ema else 0) + (1 if valid_vol else 0) + (1 if valid_adx else 0)
        return score
    
    def calculate_smc_score(self, df: pd.DataFrame) -> float:
        """Calculate SMC strategy score"""
        order_block = self.detect_order_block(df)
        fvg_gap = self.detect_fvg_gap(df)
        breaker = self.detect_breaker(df)
        valid_rsi = self.validate_rsi_reversal(df)
        
        score = (1 if order_block else 0) + (1 if fvg_gap else 0) + (1 if breaker else 0) + (1 if valid_rsi else 0)
        return score
    
    def calculate_scalp_score(self, df: pd.DataFrame) -> float:
        """Calculate Scalping strategy score"""
        vwap_slope = self.detect_vwap_slope(df)
        vol_spike = self.detect_volume_spike(df)
        valid_rsi = self.validate_rsi_reversal(df)
        valid_macd = self.validate_macd_crossover(df)
        
        score = (1 if vwap_slope else 0) + (1 if vol_spike else 0) + (1 if valid_rsi else 0) + (1 if valid_macd else 0)
        return score
    
    def calculate_stage_2_technical_data(self, df: pd.DataFrame) -> Dict:
        """
        Calculate Stage 2 technical indicators for filtering
        Returns: Dictionary with EMA crossover, RSI reversal, ATR, ADX, and volatility rank
        """
        try:
            if len(df) < 50:  # Need sufficient data
                return {
                    'ema_crossover': False,
                    'rsi_reversal': False,
                    'atr_valid': False,
                    'adx': 0,
                    'volatility_rank': 0
                }
            
            # Calculate EMA crossover
            ema_fast = self.calculate_ema(df, self.fast_ema_length)
            ema_slow = self.calculate_ema(df, self.slow_ema_length)
            
            # Check for EMA crossover (fast EMA crosses above slow EMA)
            if len(ema_fast) >= 2 and len(ema_slow) >= 2:
                ema_crossover = (ema_fast.iloc[-2] <= ema_slow.iloc[-2] and 
                               ema_fast.iloc[-1] > ema_slow.iloc[-1])
            else:
                ema_crossover = False
            
            # Calculate RSI and check for reversal
            rsi = self.calculate_rsi(df)
            rsi_reversal = self.validate_rsi_reversal(df)
            
            # Calculate ATR and check if valid
            atr = self.calculate_atr(df)
            atr_valid = len(atr) > 0 and atr.iloc[-1] > 0
            
            # Calculate ADX
            adx = self.calculate_adx(df)
            adx_value = adx.iloc[-1] if len(adx) > 0 else 0
            
            # Calculate volatility rank
            vol_rank = self.calculate_volatility_rank(df)
            vol_rank_value = vol_rank.iloc[-1] if len(vol_rank) > 0 else 0
            
            return {
                'ema_crossover': ema_crossover,
                'rsi_reversal': rsi_reversal,
                'atr_valid': atr_valid,
                'adx': adx_value,
                'volatility_rank': vol_rank_value
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating Stage 2 technical data: {e}")
            return {
                'ema_crossover': False,
                'rsi_reversal': False,
                'atr_valid': False,
                'adx': 0,
                'volatility_rank': 0
            }

    def generate_signal(self, df: pd.DataFrame, symbol: str) -> Optional[Dict]:
        """Generate trading signal based on all strategies"""
        if len(df) < 50:  # Need sufficient data
            return None
        
        # Check cooldown
        current_time = pd.Timestamp.now()
        if symbol in self.last_signal_time:
            time_diff = (current_time - self.last_signal_time[symbol]).total_seconds() / 60
            if time_diff < self.cooldown_minutes:
                return None
        
        # Calculate strategy weights
        w_trap, w_smc, w_scalp = self.calculate_strategy_weights(df)
        
        # Calculate scores
        score_trap = self.calculate_trap_score(df) * w_trap
        score_smc = self.calculate_smc_score(df) * w_smc
        score_scalp = self.calculate_scalp_score(df) * w_scalp
        
        # Find maximum score
        max_score = max(score_trap, score_smc, score_scalp)
        
        # Signal condition: max_score >= 4
        if max_score < 4:
            return None
        
        # Calculate entry and exit prices
        current_price = df['close'].iloc[-1]
        atr = self.calculate_atr(df).iloc[-1]
        
        entry_price = current_price
        stop_loss = entry_price - atr * 1.0
        tp1_price = entry_price * (1 + self.tp1_mult / 100)
        tp2_price = entry_price * (1 + self.tp2_mult / 100)
        tp3_price = entry_price * (1 + self.tp3_mult / 100)
        
        # Calculate risk-reward ratio
        risk_reward = (tp1_price - entry_price) / (entry_price - stop_loss)
        
        # Final validation: risk-reward >= 1.5
        if risk_reward < 1.5:
            return None
        
        # Update cooldown
        self.last_signal_time[symbol] = current_time
        
        # Determine dominant strategy
        if score_trap == max_score:
            strategy = "Trap"
        elif score_smc == max_score:
            strategy = "SMC"
        else:
            strategy = "Scalp"
        
        return {
            'symbol': symbol,
            'strategy': strategy,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'tp1_price': tp1_price,
            'tp2_price': tp2_price,
            'tp3_price': tp3_price,
            'risk_reward': risk_reward,
            'confidence': max_score,
            'timestamp': current_time,
            'scores': {
                'trap': score_trap,
                'smc': score_smc,
                'scalp': score_scalp
            }
        } 