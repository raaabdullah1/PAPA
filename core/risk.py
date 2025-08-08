from typing import Dict, Optional, Tuple
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from collections import defaultdict
from core.logger import SimpleLogger

class RiskManager:
    def __init__(self, account_balance: float = 10000.0, risk_percentage: float = 2.0):
        """
        Initialize Risk Manager
        Args:
            account_balance: Account balance in USD
            risk_percentage: Risk percentage per trade (default 2%)
        """
        self.account_balance = account_balance
        self.risk_percentage = risk_percentage
        self.logger_module = SimpleLogger()
        self.logger = self.logger_module.logger
        
        # Cooldown tracking
        self.cooldowns = {}  # symbol -> cooldown_end_time
        
        # Daily signal tracking
        self.daily_signal_counts = {}  # timeframe -> count
        self.last_reset_date = datetime.now().date()
        
        self.logger.info(f"Risk Manager initialized - Balance: ${account_balance:,.2f}, Risk: {risk_percentage}%")
    
    def calculate_risk_reward_ratio(self, entry_price: float, stop_loss: float, take_profit: float = None) -> float:
        """
        Calculate risk-reward ratio using proper TP1 levels
        Args:
            entry_price: Entry price
            stop_loss: Stop loss price
            take_profit: Take profit price (optional, uses TP1 if not provided)
        Returns:
            Risk-reward ratio
        """
        try:
            risk = abs(entry_price - stop_loss)
            
            if take_profit:
                reward = abs(take_profit - entry_price)
            else:
                # Calculate TP1 as 1.5x risk distance to ensure R:R ≥ 1.2
                reward = risk * 1.5  # TP1 = entry + 1.5x risk distance
            
            if risk <= 0:
                return 0
            
            return reward / risk
            
        except Exception as e:
            self.logger.error(f"Error calculating risk-reward ratio: {e}")
            return 0
    
    def calculate_position_size(self, entry_price: float, stop_loss: float, current_atr: float, max_atr_24h: float) -> Dict:
        """
        Calculate position size with vol_factor and proper R:R validation
        Args:
            entry_price: Entry price
            stop_loss: Stop loss price
            current_atr: Current ATR value
            max_atr_24h: Maximum ATR in last 24 hours
        Returns:
            Dict with position_size, vol_factor, risk_amount, and validation
        """
        try:
            # Calculate vol_factor
            if max_atr_24h > 0:
                vol_factor = 1.5 - (current_atr / max_atr_24h)
                vol_factor = max(0.5, min(1.5, vol_factor))  # Clamp between 0.5 and 1.5
            else:
                vol_factor = 1.0
            
            # Calculate risk amount
            risk_amount = self.account_balance * (self.risk_percentage / 100)
            
            # Calculate price difference
            price_diff = abs(entry_price - stop_loss)
            
            if price_diff <= 0:
                return {
                    'position_size': 0,
                    'vol_factor': vol_factor,
                    'risk_amount': risk_amount,
                    'valid': False,
                    'error': 'Invalid entry/stop loss prices'
                }
            
            # Calculate position size
            position_size = (risk_amount / price_diff) * vol_factor
            
            # Calculate TP1 for proper R:R ratio (1.5x risk distance)
            tp1_price = entry_price + (price_diff * 1.5)  # TP1 = entry + 1.5x risk distance
            
            # Validate minimum R:R ratio using proper TP1
            risk_reward_ratio = self.calculate_risk_reward_ratio(entry_price, stop_loss, tp1_price)
            valid_rr = risk_reward_ratio >= 1.2
            
            return {
                'position_size': position_size,
                'vol_factor': vol_factor,
                'risk_amount': risk_amount,
                'risk_reward_ratio': risk_reward_ratio,
                'tp1_price': tp1_price,
                'valid': valid_rr,
                'error': None if valid_rr else f'R:R ratio {risk_reward_ratio:.2f} below minimum 1.2'
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return {
                'position_size': 0,
                'vol_factor': 0,
                'risk_amount': 0,
                'valid': False,
                'error': str(e)
            }
    
    def check_cooldown(self, symbol: str) -> bool:
        """
        Check if symbol is in cooldown period
        Returns:
            True if cooldown is active, False if signal can be sent
        """
        try:
            current_time = time.time()
            
            if symbol in self.cooldowns:
                cooldown_end_time = self.cooldowns[symbol]
                if current_time < cooldown_end_time:
                    remaining_time = cooldown_end_time - current_time
                    self.logger.info(f"Cooldown active for {symbol}: {remaining_time/60:.1f} minutes remaining")
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking cooldown for {symbol}: {e}")
            return False
    
    def set_cooldown(self, symbol: str):
        """Set cooldown for a symbol"""
        try:
            self.cooldowns[symbol] = time.time() + 15 * 60 # 15 minutes cooldown
            self.logger.info(f"Cooldown set for {symbol} - 15 minutes")
        except Exception as e:
            self.logger.error(f"Error setting cooldown for {symbol}: {e}")
    
    def check_daily_limits(self, timeframe: str) -> Dict:
        """
        Check if daily signal limit has been reached for a timeframe
        Args:
            timeframe: Current timeframe
        Returns:
            Dict with daily_count, adx_threshold, and threshold_adjusted
        """
        try:
            # Reset daily count if it's a new day
            current_date = datetime.now().date()
            if current_date != self.last_reset_date:
                self.daily_signal_counts = {}
                self.last_reset_date = current_date
                self.logger.info(f"Daily signal counts reset for new day: {current_date}")
            
            # Get current count for timeframe
            current_count = self.daily_signal_counts.get(timeframe, 0)
            
            # Daily limits per timeframe (increased for higher confidence threshold)
            max_per_timeframe = 50  # Increased from 20 to 50
            
            # Get ADX threshold
            adx_threshold = self.get_adx_threshold(timeframe)
            threshold_adjusted = current_count < 5
            
            if current_count >= max_per_timeframe:
                self.logger.warning(f"Daily signal limit reached for {timeframe}: {current_count}/{max_per_timeframe}")
                return {
                    'daily_count': current_count,
                    'adx_threshold': adx_threshold,
                    'threshold_adjusted': threshold_adjusted,
                    'limit_reached': True
                }
            
            return {
                'daily_count': current_count,
                'adx_threshold': adx_threshold,
                'threshold_adjusted': threshold_adjusted,
                'limit_reached': False
            }
            
        except Exception as e:
            self.logger.error(f"Error checking daily limits: {e}")
            return {
                'daily_count': 0,
                'adx_threshold': 25,
                'threshold_adjusted': False,
                'limit_reached': False
            }  # Allow signals if error occurs
    
    def increment_daily_count(self, timeframe: str):
        """
        Increment daily signal count for a timeframe
        Args:
            timeframe: Current timeframe
        """
        try:
            # Reset daily count if it's a new day
            current_date = datetime.now().date()
            if current_date != self.last_reset_date:
                self.daily_signal_counts = {}
                self.last_reset_date = current_date
                self.logger.info(f"Daily signal counts reset for new day: {current_date}")
            
            # Increment count for timeframe
            current_count = self.daily_signal_counts.get(timeframe, 0)
            self.daily_signal_counts[timeframe] = current_count + 1
            
            self.logger.info(f"Daily signal count for {timeframe}: {self.daily_signal_counts[timeframe]}")
            
        except Exception as e:
            self.logger.error(f"Error incrementing daily count: {e}")
    
    def get_adx_threshold(self, timeframe: str) -> int:
        """
        Get ADX threshold for a timeframe (can be relaxed if daily count is low)
        Args:
            timeframe: Current timeframe
        Returns:
            ADX threshold (15 if relaxed, 25 if normal)
        """
        try:
            # Reset daily count if it's a new day
            current_date = datetime.now().date()
            if current_date != self.last_reset_date:
                self.daily_signal_counts = {}
                self.last_reset_date = current_date
                self.logger.info(f"Daily signal counts reset for new day: {current_date}")
            
            # Get current count for timeframe
            current_count = self.daily_signal_counts.get(timeframe, 0)
            
            # Relax ADX threshold if daily count is low
            if current_count < 5:  # Relax if less than 5 signals
                self.logger.info(f"Relaxing ADX threshold for {timeframe} - Daily count: {current_count}/20")
                return 15  # Relaxed threshold
            else:
                return 25  # Normal threshold
                
        except Exception as e:
            self.logger.error(f"Error getting ADX threshold: {e}")
            return 25  # Default to normal threshold
    
    def validate_signal(self, symbol: str, timeframe: str, entry_price: float, stop_loss: float, 
                       current_atr: float, max_atr_24h: float) -> Dict:
        """
        Validate signal with risk management rules
        Args:
            symbol: Trading symbol
            timeframe: Current timeframe
            entry_price: Entry price
            stop_loss: Stop loss price
            current_atr: Current ATR value
            max_atr_24h: Maximum ATR in last 24 hours
        Returns:
            Dict with validation results
        """
        try:
            # Check cooldown
            if self.check_cooldown(symbol):
                return {
                    'valid': False,
                    'error': f'Cooldown active for {symbol}'
                }
            
            # Daily limits removed - rely on natural filtering from Steps 1-2
            
            # Calculate position size
            position_result = self.calculate_position_size(entry_price, stop_loss, current_atr, max_atr_24h)
            
            if not position_result['valid']:
                return {
                    'valid': False,
                    'error': position_result['error']
                }
            
            # Increment daily count
            self.increment_daily_count(timeframe)
            
            # Set cooldown
            self.set_cooldown(symbol)
            
            self.logger.info(f"Signal validated for {symbol}: Position size: {position_result['position_size']:.4f}, R:R: {position_result['risk_reward_ratio']:.2f}")
            
            return {
                'valid': True,
                'position_size': position_result['position_size'],
                'vol_factor': position_result['vol_factor'],
                'risk_reward_ratio': position_result['risk_reward_ratio'],
                'tp1_price': position_result['tp1_price']
            }
            
        except Exception as e:
            self.logger.error(f"Error validating signal: {e}")
            return {
                'valid': False,
                'error': str(e)
            } 