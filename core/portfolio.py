#!/usr/bin/env python3
"""
Portfolio Management for Crypto Signal Bot
Manages positions, exposure, and portfolio monitoring
"""

import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass

@dataclass
class Position:
    symbol: str
    side: str  # 'long' or 'short'
    entry_price: float
    quantity: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    take_profit_3: float
    entry_time: datetime
    status: str  # 'open', 'tp1_hit', 'tp2_hit', 'tp3_hit', 'sl_hit', 'closed'
    pnl: float = 0.0
    pnl_percentage: float = 0.0

class PortfolioManager:
    def __init__(self, max_positions: int = 5, max_daily_loss: float = 0.03):
        self.logger = logging.getLogger(__name__)
        self.max_positions = max_positions
        self.max_daily_loss = max_daily_loss
        self.positions = {}  # symbol -> Position
        self.closed_positions = []  # List of closed positions
        self.daily_pnl = 0.0
        self.total_pnl = 0.0
        self.account_balance = 10000.0  # Starting balance
        
    def can_open_position(self, symbol: str) -> Tuple[bool, str]:
        """
        Check if we can open a new position
        Returns: (can_open, reason)
        """
        try:
            # Check if symbol already has an open position
            if symbol in self.positions:
                return False, f"Position already open for {symbol}"
            
            # Check max positions limit
            if len(self.positions) >= self.max_positions:
                return False, f"Maximum positions limit reached ({self.max_positions})"
            
            # Check daily loss limit
            if self.daily_pnl <= -self.max_daily_loss * self.account_balance:
                return False, f"Daily loss limit reached ({self.daily_pnl:.2f})"
            
            return True, "Position can be opened"
            
        except Exception as e:
            self.logger.error(f"Error checking if position can be opened: {e}")
            return False, f"Error: {str(e)}"
    
    def open_position(self, signal_data: Dict) -> bool:
        """
        Open a new position based on signal data
        """
        try:
            symbol = signal_data.get('symbol')
            entry_price = signal_data.get('entry_price')
            stop_loss = signal_data.get('stop_loss')
            tp1 = signal_data.get('tp1_price')
            tp2 = signal_data.get('tp2_price')
            tp3 = signal_data.get('tp3_price')
            
            # Validate required data
            if not all([symbol, entry_price, stop_loss, tp1, tp2, tp3]):
                self.logger.error(f"Missing required data for position: {signal_data}")
                return False
            
            # Check if we can open position
            can_open, reason = self.can_open_position(symbol)
            if not can_open:
                self.logger.warning(f"Cannot open position for {symbol}: {reason}")
                return False
            
            # Calculate position size (1% risk per position)
            risk_amount = self.account_balance * 0.01
            price_diff = abs(entry_price - stop_loss)
            if price_diff == 0:
                self.logger.error(f"Invalid stop loss for {symbol}")
                return False
            
            quantity = risk_amount / price_diff
            
            # Create position
            position = Position(
                symbol=symbol,
                side='long',  # Assuming long signals for now
                entry_price=entry_price,
                quantity=quantity,
                stop_loss=stop_loss,
                take_profit_1=tp1,
                take_profit_2=tp2,
                take_profit_3=tp3,
                entry_time=datetime.now(),
                status='open'
            )
            
            self.positions[symbol] = position
            self.logger.info(f"Position opened for {symbol} at {entry_price}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error opening position: {e}")
            return False
    
    def update_position(self, symbol: str, current_price: float) -> Dict:
        """
        Update position with current price and check for TP/SL hits
        """
        try:
            if symbol not in self.positions:
                return {'error': f'No position found for {symbol}'}
            
            position = self.positions[symbol]
            
            # Calculate current P&L
            if position.side == 'long':
                pnl = (current_price - position.entry_price) * position.quantity
            else:
                pnl = (position.entry_price - current_price) * position.quantity
            
            position.pnl = pnl
            position.pnl_percentage = (pnl / (position.entry_price * position.quantity)) * 100
            
            # Check for TP/SL hits
            status_update = self._check_tp_sl_hits(position, current_price)
            
            return {
                'symbol': symbol,
                'current_price': current_price,
                'pnl': pnl,
                'pnl_percentage': position.pnl_percentage,
                'status': position.status,
                'status_update': status_update
            }
            
        except Exception as e:
            self.logger.error(f"Error updating position for {symbol}: {e}")
            return {'error': str(e)}
    
    def _check_tp_sl_hits(self, position: Position, current_price: float) -> Optional[str]:
        """
        Check if current price hits any TP or SL levels
        """
        try:
            if position.status != 'open':
                return None
            
            # Check stop loss
            if position.side == 'long' and current_price <= position.stop_loss:
                position.status = 'sl_hit'
                self._close_position(position, current_price, 'stop_loss')
                return 'Stop loss hit'
            
            # Check take profits
            if position.side == 'long':
                if current_price >= position.take_profit_3:
                    position.status = 'tp3_hit'
                    self._close_position(position, position.take_profit_3, 'tp3')
                    return 'TP3 hit'
                elif current_price >= position.take_profit_2:
                    position.status = 'tp2_hit'
                    self._close_position(position, position.take_profit_2, 'tp2')
                    return 'TP2 hit'
                elif current_price >= position.take_profit_1:
                    position.status = 'tp1_hit'
                    self._close_position(position, position.take_profit_1, 'tp1')
                    return 'TP1 hit'
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error checking TP/SL hits: {e}")
            return None
    
    def _close_position(self, position: Position, exit_price: float, exit_reason: str):
        """
        Close a position and update portfolio
        """
        try:
            # Calculate final P&L
            if position.side == 'long':
                final_pnl = (exit_price - position.entry_price) * position.quantity
            else:
                final_pnl = (position.entry_price - exit_price) * position.quantity
            
            position.pnl = final_pnl
            position.pnl_percentage = (final_pnl / (position.entry_price * position.quantity)) * 100
            
            # Update portfolio totals
            self.total_pnl += final_pnl
            self.daily_pnl += final_pnl
            self.account_balance += final_pnl
            
            # Move to closed positions
            self.closed_positions.append(position)
            del self.positions[position.symbol]
            
            self.logger.info(f"Position closed for {position.symbol}: {exit_reason}, P&L: {final_pnl:.2f}")
            
        except Exception as e:
            self.logger.error(f"Error closing position: {e}")
    
    def get_portfolio_status(self) -> Dict:
        """
        Get comprehensive portfolio status
        """
        try:
            open_positions = len(self.positions)
            total_positions = len(self.closed_positions) + open_positions
            
            # Calculate win rate
            if total_positions > 0:
                winning_trades = sum(1 for pos in self.closed_positions if pos.pnl > 0)
                win_rate = (winning_trades / total_positions) * 100
            else:
                win_rate = 0.0
            
            return {
                'account_balance': self.account_balance,
                'total_pnl': self.total_pnl,
                'daily_pnl': self.daily_pnl,
                'open_positions': open_positions,
                'max_positions': self.max_positions,
                'total_trades': total_positions,
                'win_rate': win_rate,
                'daily_loss_limit': self.max_daily_loss * self.account_balance,
                'can_open_more': open_positions < self.max_positions
            }
            
        except Exception as e:
            self.logger.error(f"Error getting portfolio status: {e}")
            return {'error': str(e)}
    
    def get_position_details(self, symbol: str) -> Optional[Dict]:
        """
        Get detailed information about a specific position
        """
        try:
            if symbol not in self.positions:
                return None
            
            position = self.positions[symbol]
            return {
                'symbol': position.symbol,
                'side': position.side,
                'entry_price': position.entry_price,
                'quantity': position.quantity,
                'stop_loss': position.stop_loss,
                'take_profit_1': position.take_profit_1,
                'take_profit_2': position.take_profit_2,
                'take_profit_3': position.take_profit_3,
                'entry_time': position.entry_time.isoformat(),
                'status': position.status,
                'pnl': position.pnl,
                'pnl_percentage': position.pnl_percentage
            }
            
        except Exception as e:
            self.logger.error(f"Error getting position details for {symbol}: {e}")
            return None
    
    def reset_daily_pnl(self):
        """
        Reset daily P&L (call this at midnight)
        """
        try:
            self.daily_pnl = 0.0
            self.logger.info("Daily P&L reset")
            
        except Exception as e:
            self.logger.error(f"Error resetting daily P&L: {e}")
    
    def get_all_positions(self) -> Dict[str, Dict]:
        """
        Get all open positions
        """
        try:
            positions = {}
            for symbol, position in self.positions.items():
                positions[symbol] = self.get_position_details(symbol)
            
            return positions
            
        except Exception as e:
            self.logger.error(f"Error getting all positions: {e}")
            return {} 