#!/usr/bin/env python3
"""
Cooldown Management for Crypto Signal Bot
Manages signal cooldowns and prevents signal spam
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta

class CooldownManager:
    def __init__(self, cooldown_minutes: int = 30):
        self.logger = logging.getLogger(__name__)
        self.cooldown_minutes = cooldown_minutes
        self.last_signals = {}  # symbol -> last_signal_time
        self.daily_signals = {}  # symbol -> [signal_times]
        
    def can_generate_signal(self, symbol: str) -> bool:
        """
        Check if enough time has passed since the last signal for this symbol
        """
        try:
            if symbol not in self.last_signals:
                return True
            
            last_signal_time = self.last_signals[symbol]
            time_since_last = datetime.now() - last_signal_time
            
            # Check if cooldown period has passed
            cooldown_duration = timedelta(minutes=self.cooldown_minutes)
            can_generate = time_since_last >= cooldown_duration
            
            if not can_generate:
                remaining_time = cooldown_duration - time_since_last
                self.logger.info(f"Cooldown active for {symbol}. Remaining: {remaining_time}")
            
            return can_generate
            
        except Exception as e:
            self.logger.error(f"Error checking cooldown for {symbol}: {e}")
            return True  # Allow signal generation on error
    
    def record_signal(self, symbol: str, signal_time: datetime = None):
        """
        Record a signal generation time for cooldown tracking
        """
        try:
            if signal_time is None:
                signal_time = datetime.now()
            
            self.last_signals[symbol] = signal_time
            
            # Track daily signals
            if symbol not in self.daily_signals:
                self.daily_signals[symbol] = []
            
            self.daily_signals[symbol].append(signal_time)
            
            # Clean up old daily signals (older than 24 hours)
            cutoff_time = datetime.now() - timedelta(hours=24)
            self.daily_signals[symbol] = [
                time for time in self.daily_signals[symbol] 
                if time > cutoff_time
            ]
            
            self.logger.info(f"Signal recorded for {symbol} at {signal_time}")
            
        except Exception as e:
            self.logger.error(f"Error recording signal for {symbol}: {e}")
    
    def get_daily_signal_count(self, symbol: str) -> int:
        """
        Get the number of signals generated today for a symbol
        """
        try:
            if symbol not in self.daily_signals:
                return 0
            
            # Count signals from today
            today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            today_signals = [
                time for time in self.daily_signals[symbol] 
                if time >= today_start
            ]
            
            return len(today_signals)
            
        except Exception as e:
            self.logger.error(f"Error getting daily signal count for {symbol}: {e}")
            return 0
    
    def get_remaining_cooldown(self, symbol: str) -> Optional[timedelta]:
        """
        Get remaining cooldown time for a symbol
        """
        try:
            if symbol not in self.last_signals:
                return None
            
            last_signal_time = self.last_signals[symbol]
            time_since_last = datetime.now() - last_signal_time
            cooldown_duration = timedelta(minutes=self.cooldown_minutes)
            
            if time_since_last < cooldown_duration:
                return cooldown_duration - time_since_last
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"Error getting remaining cooldown for {symbol}: {e}")
            return None
    
    def get_cooldown_status(self, symbol: str) -> Dict:
        """
        Get comprehensive cooldown status for a symbol
        """
        try:
            can_generate = self.can_generate_signal(symbol)
            daily_count = self.get_daily_signal_count(symbol)
            remaining_cooldown = self.get_remaining_cooldown(symbol)
            last_signal_time = self.last_signals.get(symbol)
            
            return {
                'symbol': symbol,
                'can_generate': can_generate,
                'daily_count': daily_count,
                'remaining_cooldown': str(remaining_cooldown) if remaining_cooldown else None,
                'last_signal_time': last_signal_time.isoformat() if last_signal_time else None,
                'cooldown_minutes': self.cooldown_minutes
            }
            
        except Exception as e:
            self.logger.error(f"Error getting cooldown status for {symbol}: {e}")
            return {
                'symbol': symbol,
                'can_generate': True,
                'error': str(e)
            }
    
    def reset_cooldown(self, symbol: str):
        """
        Reset cooldown for a symbol (for testing purposes)
        """
        try:
            if symbol in self.last_signals:
                del self.last_signals[symbol]
            if symbol in self.daily_signals:
                del self.daily_signals[symbol]
            
            self.logger.info(f"Cooldown reset for {symbol}")
            
        except Exception as e:
            self.logger.error(f"Error resetting cooldown for {symbol}: {e}")
    
    def get_all_cooldown_status(self) -> Dict[str, Dict]:
        """
        Get cooldown status for all tracked symbols
        """
        try:
            status = {}
            for symbol in self.last_signals.keys():
                status[symbol] = self.get_cooldown_status(symbol)
            
            return status
            
        except Exception as e:
            self.logger.error(f"Error getting all cooldown status: {e}")
            return {} 