#!/usr/bin/env python3
"""
Dark Pool Filtering for Crypto Signal Bot
Filters out signals that may be affected by dark pool activity
"""

import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

class DarkPoolFilter:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Dark pool detection thresholds
        self.min_volume_ratio = 0.1  # Minimum ratio of visible to total volume
        self.max_price_impact = 0.005  # Maximum price impact (0.5%)
        self.min_trade_size = 1000  # Minimum trade size to consider
        self.suspicious_patterns = [
            'large_orders',
            'price_manipulation',
            'volume_spikes',
            'unusual_spreads'
        ]
        
    def analyze_volume_distribution(self, market_data: Dict) -> Dict:
        """
        Analyze volume distribution to detect potential dark pool activity
        """
        try:
            total_volume = market_data.get('volume', 0)
            visible_volume = market_data.get('visible_volume', 0)
            large_trades = market_data.get('large_trades', [])
            
            # Calculate volume ratio
            if total_volume > 0:
                volume_ratio = visible_volume / total_volume
            else:
                volume_ratio = 0
            
            # Analyze large trades
            large_trade_volume = sum(trade.get('volume', 0) for trade in large_trades)
            large_trade_ratio = large_trade_volume / total_volume if total_volume > 0 else 0
            
            # Check for suspicious patterns
            suspicious_indicators = []
            
            if volume_ratio < self.min_volume_ratio:
                suspicious_indicators.append('low_visible_volume')
            
            if large_trade_ratio > 0.3:  # More than 30% in large trades
                suspicious_indicators.append('high_large_trade_ratio')
            
            # Check for volume spikes
            avg_volume = market_data.get('avg_volume', 0)
            if total_volume > avg_volume * 3:  # 3x average volume
                suspicious_indicators.append('volume_spike')
            
            return {
                'total_volume': total_volume,
                'visible_volume': visible_volume,
                'volume_ratio': volume_ratio,
                'large_trade_volume': large_trade_volume,
                'large_trade_ratio': large_trade_ratio,
                'suspicious_indicators': suspicious_indicators,
                'is_suspicious': len(suspicious_indicators) > 0
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing volume distribution: {e}")
            return {
                'error': str(e),
                'is_suspicious': False
            }
    
    def analyze_price_impact(self, market_data: Dict) -> Dict:
        """
        Analyze price impact to detect manipulation
        """
        try:
            current_price = market_data.get('price', 0)
            vwap = market_data.get('vwap', 0)
            bid = market_data.get('bid', 0)
            ask = market_data.get('ask', 0)
            
            # Calculate price deviations
            price_impact = {}
            
            if vwap > 0:
                vwap_deviation = abs(current_price - vwap) / vwap
                price_impact['vwap_deviation'] = vwap_deviation
            
            if bid > 0 and ask > 0:
                spread = (ask - bid) / bid
                price_impact['spread'] = spread
                
                # Check for unusual spreads
                if spread > 0.01:  # 1% spread
                    price_impact['unusual_spread'] = True
            
            # Check for price manipulation patterns
            manipulation_indicators = []
            
            if vwap_deviation > self.max_price_impact:
                manipulation_indicators.append('high_price_deviation')
            
            if spread > 0.01:
                manipulation_indicators.append('wide_spread')
            
            return {
                'price_impact': price_impact,
                'manipulation_indicators': manipulation_indicators,
                'is_manipulated': len(manipulation_indicators) > 0
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing price impact: {e}")
            return {
                'error': str(e),
                'is_manipulated': False
            }
    
    def detect_large_orders(self, order_book: Dict) -> Dict:
        """
        Detect large orders that might indicate dark pool activity
        """
        try:
            bids = order_book.get('bids', [])
            asks = order_book.get('asks', [])
            
            large_bids = []
            large_asks = []
            
            # Analyze bid orders
            for bid in bids:
                price, volume = bid
                if volume > self.min_trade_size:
                    large_bids.append({
                        'price': price,
                        'volume': volume,
                        'size_category': self._categorize_order_size(volume)
                    })
            
            # Analyze ask orders
            for ask in asks:
                price, volume = ask
                if volume > self.min_trade_size:
                    large_asks.append({
                        'price': price,
                        'volume': volume,
                        'size_category': self._categorize_order_size(volume)
                    })
            
            # Check for order book imbalances
            total_bid_volume = sum(bid['volume'] for bid in large_bids)
            total_ask_volume = sum(ask['volume'] for ask in large_asks)
            
            imbalance_ratio = total_bid_volume / total_ask_volume if total_ask_volume > 0 else 1
            
            return {
                'large_bids': large_bids,
                'large_asks': large_asks,
                'total_bid_volume': total_bid_volume,
                'total_ask_volume': total_ask_volume,
                'imbalance_ratio': imbalance_ratio,
                'has_large_orders': len(large_bids) > 0 or len(large_asks) > 0
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting large orders: {e}")
            return {
                'error': str(e),
                'has_large_orders': False
            }
    
    def _categorize_order_size(self, volume: float) -> str:
        """
        Categorize order size
        """
        if volume < 10000:
            return 'small'
        elif volume < 100000:
            return 'medium'
        elif volume < 1000000:
            return 'large'
        else:
            return 'very_large'
    
    def analyze_trade_patterns(self, trades: List[Dict]) -> Dict:
        """
        Analyze trade patterns for suspicious activity
        """
        try:
            if not trades:
                return {'suspicious_patterns': [], 'is_suspicious': False}
            
            # Analyze trade timing
            trade_times = [trade.get('timestamp', 0) for trade in trades]
            time_gaps = []
            
            for i in range(1, len(trade_times)):
                gap = trade_times[i] - trade_times[i-1]
                time_gaps.append(gap)
            
            # Check for unusual patterns
            suspicious_patterns = []
            
            # Check for rapid-fire trades
            rapid_trades = sum(1 for gap in time_gaps if gap < 1)  # Less than 1 second
            if rapid_trades > len(trades) * 0.5:  # More than 50% rapid trades
                suspicious_patterns.append('rapid_fire_trades')
            
            # Check for large trade clusters
            large_trades = [t for t in trades if t.get('volume', 0) > self.min_trade_size]
            if len(large_trades) > len(trades) * 0.3:  # More than 30% large trades
                suspicious_patterns.append('large_trade_cluster')
            
            # Check for price manipulation
            prices = [t.get('price', 0) for t in trades]
            if len(prices) > 1:
                price_changes = [abs(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
                avg_price_change = sum(price_changes) / len(price_changes)
                
                if avg_price_change > self.max_price_impact:
                    suspicious_patterns.append('excessive_price_volatility')
            
            return {
                'total_trades': len(trades),
                'large_trades': len(large_trades),
                'rapid_trades': rapid_trades,
                'suspicious_patterns': suspicious_patterns,
                'is_suspicious': len(suspicious_patterns) > 0
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing trade patterns: {e}")
            return {
                'error': str(e),
                'is_suspicious': False
            }
    
    def apply_dark_pool_filter(self, market_data: Dict, order_book: Dict = None, trades: List[Dict] = None) -> Dict:
        """
        Apply comprehensive dark pool filtering
        """
        try:
            # Analyze volume distribution
            volume_analysis = self.analyze_volume_distribution(market_data)
            
            # Analyze price impact
            price_analysis = self.analyze_price_impact(market_data)
            
            # Analyze order book if available
            order_analysis = {}
            if order_book:
                order_analysis = self.detect_large_orders(order_book)
            
            # Analyze trade patterns if available
            trade_analysis = {}
            if trades:
                trade_analysis = self.analyze_trade_patterns(trades)
            
            # Combine all indicators
            all_indicators = []
            
            if volume_analysis.get('is_suspicious', False):
                all_indicators.extend(volume_analysis.get('suspicious_indicators', []))
            
            if price_analysis.get('is_manipulated', False):
                all_indicators.extend(price_analysis.get('manipulation_indicators', []))
            
            if trade_analysis.get('is_suspicious', False):
                all_indicators.extend(trade_analysis.get('suspicious_patterns', []))
            
            # Determine if signal should be filtered
            should_filter = len(all_indicators) >= 2  # At least 2 suspicious indicators
            
            return {
                'should_filter': should_filter,
                'suspicious_indicators': all_indicators,
                'volume_analysis': volume_analysis,
                'price_analysis': price_analysis,
                'order_analysis': order_analysis,
                'trade_analysis': trade_analysis,
                'filter_reason': f"Dark pool activity detected: {', '.join(all_indicators)}" if should_filter else None
            }
            
        except Exception as e:
            self.logger.error(f"Error applying dark pool filter: {e}")
            return {
                'should_filter': False,
                'error': str(e)
            } 