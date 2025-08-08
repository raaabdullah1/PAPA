#!/usr/bin/env python3
"""
Three-Stage Filter Mechanism for Crypto Signal Bot
Implements the filtering logic as specified in the original prompt
"""

import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime

# Centralized thresholds
RSI_OVERSOLD = 27
RSI_OVERBOUGHT = 80
EMA_FAST = 9
EMA_SLOW = 21
VOLUME_SPIKE_MULTIPLIER = 1.5

class FilterTree:
    def __init__(self):
        """Initialize the filter tree with thresholds"""
        self.logger = logging.getLogger(__name__)
        
        # Stage 1 thresholds
        self.volume_percentile = 50  # Top 50% by volume
        self.max_funding_rate = 0.005  # ±0.50%
        self.max_spread = 0.005  # 0.50%
        
        # Stage 2 thresholds
        self.min_atr_threshold = 0.0005  # 0.05%
        
        # Stage 3 thresholds
        self.vol_spike_threshold = 1.5  # Volume > vol_sma * 1.5
        self.adx_threshold = 25  # ADX > 25 for regime detection
        self.volatility_rank_threshold = 80  # Volatility rank > 80 for regime detection
        
    def stage_1_filter(self, market_data: Dict, volume_rank: int = None, total_symbols: int = None) -> Tuple[bool, Dict]:
        """
        Stage 1 Filter (Top-30% Volume Percentile + Order-Book)
        • Top 30% by 24h quoteVolume
        • Funding rate in [–0.50%, 0.50%]
        • Bid/ask spread < 0.50%
        Returns: (passed, details)
        """
        try:
            volume = market_data.get('volume', 0)
            funding_rate = market_data.get('funding_rate', 0)
            spread = market_data.get('spread', 0)
            
            # Check if symbol is in top 30% by volume
            if volume_rank is not None and total_symbols is not None:
                top_n_count = int(total_symbols * (self.volume_percentile / 100))
                volume_passed = volume_rank <= top_n_count
            else:
                # Fallback: if rank not provided, assume passed
                volume_passed = True
            
            # Check funding rate in [–0.50%, 0.50%]
            funding_passed = abs(funding_rate) <= self.max_funding_rate  # 0.005 (0.50%)
            
            # Check bid/ask spread < 0.50%
            spread_passed = spread <= self.max_spread  # 0.005 (0.50%)
            
            # Overall stage 1 result
            stage_passed = volume_passed and funding_passed and spread_passed
            
            details = {
                'volume_passed': volume_passed,
                'funding_passed': funding_passed,
                'spread_passed': spread_passed,
                'volume': volume,
                'funding_rate': funding_rate,
                'spread': spread,
                'volume_rank': volume_rank,
                'volume_percentile': self.volume_percentile,
                'funding_threshold': self.max_funding_rate,
                'spread_threshold': self.max_spread,
                'strength_dot': 1 if stage_passed else 0
            }
            
            return stage_passed, details
            
        except Exception as e:
            self.logger.error(f"Error in stage 1 filter: {e}")
            return False, {'error': str(e)}
    
    def stage_1_filter_symbols(self, symbols_data: List[Dict]) -> List[Dict]:
        """
        Stage 1 Filter: Top-30% Volume Percentile + Order-Book Filter
        Args:
            symbols_data: List of dictionaries with market data for each symbol
        Returns:
            List of symbols that passed Stage 1 filter with strength dot 1
        """
        try:
            if not symbols_data:
                return []
            
            # Step 1: Sort symbols by volume descending and assign ranks
            volume_data = []
            for symbol_data in symbols_data:
                symbol = symbol_data.get('symbol', '')
                market_data = symbol_data.get('market_data', {})
                volume = market_data.get('volume', 0)
                
                if volume > 0:  # Only include symbols with valid volume
                    volume_data.append({
                        'symbol': symbol,
                        'market_data': market_data,
                        'volume': volume
                    })
            
            # Sort by volume descending
            volume_data.sort(key=lambda x: x['volume'], reverse=True)
            
            # Assign ranks (1-based)
            for i, data in enumerate(volume_data):
                data['volume_rank'] = i + 1
            
            total_symbols = len(volume_data)
            top_n_count = int(total_symbols * (self.volume_percentile / 100))
            
            self.logger.info(f"Stage 1: Total symbols with volume: {total_symbols}")
            self.logger.info(f"Stage 1: Top {self.volume_percentile}% = {top_n_count} symbols")
            
            # Step 2: Apply Stage 1 filter to ranked symbols
            passing_symbols = []
            
            for data in volume_data:
                symbol = data['symbol']
                market_data = data['market_data']
                volume_rank = data['volume_rank']
                
                # Apply Stage 1 filter with volume rank
                passed, details = self.stage_1_filter(market_data, volume_rank, total_symbols)
                
                if passed:
                    # Add strength dot 1 and symbol info
                    filtered_symbol = {
                        'symbol': symbol,
                        'stage_1_passed': True,
                        'strength_dot': 1,
                        'filter_details': details,
                        'volume': details.get('volume', 0),
                        'funding_rate': details.get('funding_rate', 0),
                        'spread': details.get('spread', 0),
                        'volume_rank': volume_rank
                    }
                    passing_symbols.append(filtered_symbol)
                    
                    self.logger.info(f"Stage 1 PASS: {symbol} - Rank: {volume_rank}/{total_symbols}, "
                                   f"Volume: ${details.get('volume', 0):,.0f}, "
                                   f"Funding: {details.get('funding_rate', 0):.4f}%, "
                                   f"Spread: {details.get('spread', 0):.4f}%")
                else:
                    self.logger.debug(f"Stage 1 FAIL: {symbol} - Rank: {volume_rank}/{total_symbols}, "
                                    f"Volume: ${details.get('volume', 0):,.0f}, "
                                    f"Funding: {details.get('funding_rate', 0):.4f}%, "
                                    f"Spread: {details.get('spread', 0):.4f}%")
            
            self.logger.info(f"Stage 1 Filter: {len(passing_symbols)}/{total_symbols} symbols passed")
            return passing_symbols
            
        except Exception as e:
            self.logger.error(f"Error in stage 1 filter symbols: {e}")
            return []
    
    def stage_1_filter_with_stored_data(self, all_symbols_data: List[Dict]) -> List[Dict]:
        """
        Stage 1: Liquidity filter using stored symbol data (no API calls)
        Args:
            all_symbols_data: List of dictionaries with complete symbol data
        Returns:
            List of symbols that passed Stage 1 filter
        """
        try:
            if not all_symbols_data:
                raise ValueError("No symbol data provided to Stage 1 filter")
            
            # CRITICAL QUALITY CHECK: Validate data structure
            required_fields = ['symbol', 'volume_usd', 'price']
            for i, symbol_data in enumerate(all_symbols_data):
                missing_fields = [field for field in required_fields if field not in symbol_data]
                if missing_fields:
                    raise ValueError(f"Symbol {i} missing critical fields: {missing_fields}")
                
                # Validate data types
                if not isinstance(symbol_data['symbol'], str):
                    raise ValueError(f"Symbol {i}: 'symbol' must be string, got {type(symbol_data['symbol'])}")
                if not isinstance(symbol_data['volume_usd'], (int, float)):
                    raise ValueError(f"Symbol {i}: 'volume_usd' must be numeric, got {type(symbol_data['volume_usd'])}")
                if not isinstance(symbol_data['price'], (int, float)):
                    raise ValueError(f"Symbol {i}: 'price' must be numeric, got {type(symbol_data['price'])}")
            
            # Sort symbols by volume for percentile calculation
            volume_data = []
            for symbol_data in all_symbols_data:
                volume_usd = symbol_data.get('volume_usd', 0)
                if volume_usd > 0:
                    volume_data.append({
                        'symbol': symbol_data['symbol'],
                        'volume_usd': volume_usd,
                        'funding_rate': symbol_data.get('funding_rate'),
                        'spread': symbol_data.get('spread'),
                        'symbol_data': symbol_data
                    })
            
            # CRITICAL QUALITY CHECK: Ensure we have sufficient data
            if len(volume_data) < 10:
                raise ValueError(f"Insufficient symbols with volume data: {len(volume_data)} (minimum 10 required)")
            
            # Sort by volume descending
            volume_data.sort(key=lambda x: x['volume_usd'], reverse=True)
            
            # Calculate volume percentile threshold
            total_symbols = len(volume_data)
            top_n_count = int(total_symbols * (self.volume_percentile / 100))
            
            self.logger.info(f"Stage 1: Total symbols with volume: {total_symbols}")
            self.logger.info(f"Stage 1: Top {self.volume_percentile}% = {top_n_count} symbols")
            
            stage1_passes = []
            
            for i, data in enumerate(volume_data):
                symbol = data['symbol']
                volume_usd = data['volume_usd']
                funding_rate = data['funding_rate']
                spread = data['spread']
                
                # Assertion to prevent NoneType errors in formatting
                assert volume_usd is not None, f"Volume USD cannot be None for {symbol}"
                assert isinstance(volume_usd, (int, float)), f"Volume USD must be numeric for {symbol}, got {type(volume_usd)}"
                
                # Stage 1 criteria - STRICT QUALITY CHECKS
                volume_passed = i < top_n_count  # Top 50% by volume
                
                # CRITICAL: Require funding_rate and spread data for production quality
                if funding_rate is None:
                    self.logger.warning(f"CRITICAL: {symbol} missing funding_rate data - skipping for quality")
                    continue
                
                if spread is None:
                    self.logger.warning(f"CRITICAL: {symbol} missing spread data - skipping for quality")
                    continue
                
                # Validate data types for critical fields
                if not isinstance(funding_rate, (int, float)):
                    self.logger.warning(f"CRITICAL: {symbol} funding_rate must be numeric, got {type(funding_rate)} - skipping")
                    continue
                
                if not isinstance(spread, (int, float)):
                    self.logger.warning(f"CRITICAL: {symbol} spread must be numeric, got {type(spread)} - skipping")
                    continue
                
                # Apply strict criteria
                funding_passed = abs(funding_rate) <= self.max_funding_rate
                spread_passed = spread <= self.max_spread
                
                if volume_passed and funding_passed and spread_passed:
                    stage1_passes.append(data['symbol_data'])
                    
                    # Safe formatting with None handling
                    funding_display = f"{funding_rate:.4f}%" if funding_rate is not None else "N/A"
                    spread_display = f"{spread:.4f}%" if spread is not None else "N/A"
                    
                    self.logger.info(f"Stage 1 PASS: {symbol} - Rank: {i+1}/{total_symbols}, Volume: ${volume_usd:,.0f}, Funding: {funding_display}, Spread: {spread_display}")
            
            # CRITICAL QUALITY CHECK: Ensure minimum pass rate
            pass_rate = len(stage1_passes) / len(volume_data) * 100
            if pass_rate < 1.0:  # Less than 1% pass rate indicates potential issues
                self.logger.warning(f"CRITICAL: Very low Stage 1 pass rate: {pass_rate:.1f}% - may indicate overly strict criteria")
            
            self.logger.info(f"Stage 1 Filter: {len(stage1_passes)}/{total_symbols} symbols passed ({pass_rate:.1f}% pass rate)")
            return stage1_passes
            
        except Exception as e:
            self.logger.error(f"CRITICAL ERROR in stage 1 filter: {e}")
            raise  # Re-raise to fail fast
    
    def stage_2_filter(self, technical_data: Dict, adx_threshold: int = 25) -> Tuple[bool, Dict]:
        """
        Stage 2: EMA, RSI, ATR, MACD criteria with regime-aware thresholds for both LONG and SHORT signals
        • LONG: Fast EMA (9) > slow EMA (21), RSI rising from oversold, MACD line > signal line
        • SHORT: Fast EMA (9) < slow EMA (21), RSI falling from overbought, MACD line < signal line
        • ATR validation (minimum volatility threshold of 0.05%)
        • Volatility rank (ATR/SMA) as optional fourth indicator
        • REGIME-AWARE: High-vol regime (ADX>25 && VolRank>80): require ≥2 of 3 indicators
        • REGIME-AWARE: Normal regime: require ≥1 of 3 indicators
        Args:
            technical_data: Technical indicator data
            adx_threshold: Dynamic ADX threshold (default 25, can be relaxed to 15)
        Returns: (passed, details)
        """
        try:
            # Extract bullish indicators
            ema_crossover = technical_data.get('ema_crossover', False)
            rsi_reversal = technical_data.get('rsi_reversal', False)
            atr_valid = technical_data.get('atr_valid', False)
            macd_bullish = technical_data.get('macd_bullish', False)
            adx_value = technical_data.get('adx_value', 0)
            volatility_rank = technical_data.get('volatility_rank', 0)
            
            # Extract bearish indicators (inverted logic)
            ema_crossunder = technical_data.get('ema_crossunder', False)
            rsi_falling = technical_data.get('rsi_falling', False)
            macd_bearish = technical_data.get('macd_bearish', False)
            
            # Calculate regime based on ADX and Volatility Rank
            high_adx = adx_value > 25  # ADX(14) > 25
            high_volatility = volatility_rank > 80  # VolRank > 80
            regime_high_activity = high_adx and high_volatility
            
            # Determine required indicator count based on regime
            if regime_high_activity:
                required_indicators = 2  # ≥2 of 3 in high activity regime
                regime_type = "HIGH_VOL"
            else:
                required_indicators = 1  # ≥1 of 3 in normal regime
                regime_type = "NORMAL"
            
            # Count passing indicators for LONG signals (EMA, MACD, RSI only - 3 core indicators)
            long_indicator_passes = sum([ema_crossover, macd_bullish, rsi_reversal])
            
            # Count passing indicators for SHORT signals (EMA, MACD, RSI only - 3 core indicators)
            short_indicator_passes = sum([ema_crossunder, macd_bearish, rsi_falling])
            
            # Determine signal direction based on which set passes
            long_passed = atr_valid and long_indicator_passes >= required_indicators
            short_passed = atr_valid and short_indicator_passes >= required_indicators
            
            # Determine signal direction
            if long_passed and not short_passed:
                signal_direction = "LONG"
                indicator_passes = long_indicator_passes
                stage_passed = True
            elif short_passed and not long_passed:
                signal_direction = "SHORT"
                indicator_passes = short_indicator_passes
                stage_passed = True
            elif long_passed and short_passed:
                # If both pass, prefer the one with more indicator passes
                if long_indicator_passes >= short_indicator_passes:
                    signal_direction = "LONG"
                    indicator_passes = long_indicator_passes
                else:
                    signal_direction = "SHORT"
                    indicator_passes = short_indicator_passes
                stage_passed = True
            else:
                signal_direction = "NONE"
                indicator_passes = 0
                stage_passed = False
            
            details = {
                'ema_crossover': ema_crossover,
                'rsi_reversal': rsi_reversal,
                'atr_valid': atr_valid,
                'macd_bullish': macd_bullish,
                'ema_crossunder': ema_crossunder,
                'rsi_falling': rsi_falling,
                'macd_bearish': macd_bearish,
                'signal_direction': signal_direction,
                'long_indicator_passes': long_indicator_passes,
                'short_indicator_passes': short_indicator_passes,
                'atr_value': technical_data.get('atr_value', 0),
                'rsi_value': technical_data.get('rsi_value', 0),
                'ema_fast': technical_data.get('ema_fast', 0),
                'ema_slow': technical_data.get('ema_slow', 0),
                'macd_line': technical_data.get('macd_line', 0),
                'macd_signal': technical_data.get('macd_signal', 0),
                'adx_value': adx_value,
                'volatility_rank': volatility_rank,
                'regime_type': regime_type,
                'required_indicators': required_indicators,
                'indicator_passes': indicator_passes,
                'regime_high_activity': regime_high_activity,
                'adx_threshold_used': adx_threshold,
                'high_adx': high_adx,
                'high_volatility': high_volatility
            }
            
            # Log regime and applied threshold for audit
            self.logger.info(f"Stage 2 Regime: {regime_type} | ADX: {adx_value:.2f} (threshold: 25) | VolRank: {volatility_rank:.2f} (threshold: 80) | Required: {required_indicators} | Long: {long_indicator_passes}/3 | Short: {short_indicator_passes}/3 | Direction: {signal_direction}")
            
            return stage_passed, details
            
        except Exception as e:
            self.logger.error(f"Error in stage 2 filter: {e}")
            return False, {'error': str(e)}
    
    def _calculate_stage_2_indicators(self, ohlcv: List) -> Dict:
        """
        Calculate Stage 2 technical indicators without pandas/numpy dependency
        Args:
            ohlcv: List of [timestamp, open, high, low, close, volume] data
        Returns:
            Dictionary with EMA crossover, RSI reversal, ATR validation, MACD bullish, ADX, and volatility rank
        """
        try:
            if len(ohlcv) < 50:
                return {
                    'ema_crossover': False, 'rsi_reversal': False, 'atr_valid': False,
                    'macd_bullish': False, 'ema_crossunder': False, 'rsi_falling': False,
                    'macd_bearish': False, 'atr_value': 0, 'rsi_value': 0,
                    'ema_fast': 0, 'ema_slow': 0, 'macd_line': 0, 'macd_signal': 0,
                    'adx_value': 0, 'volatility_rank': 0
                }
            
            closes = []
            highs = []
            lows = []
            volumes = []
            
            for candle in ohlcv:
                try:
                    closes.append(float(candle[4]))
                    highs.append(float(candle[2]))
                    lows.append(float(candle[3]))
                    volumes.append(float(candle[5]))
                except (IndexError, ValueError, TypeError):
                    continue
            
            if len(closes) < 50:  # Re-check after filtering bad data
                return {
                    'ema_crossover': False, 'rsi_reversal': False, 'atr_valid': False,
                    'macd_bullish': False, 'ema_crossunder': False, 'rsi_falling': False,
                    'macd_bearish': False, 'atr_value': 0, 'rsi_value': 0,
                    'ema_fast': 0, 'ema_slow': 0, 'macd_line': 0, 'macd_signal': 0,
                    'adx_value': 0, 'volatility_rank': 0
                }
            
            # Calculate basic indicators
            ema_fast = self._calculate_ema(closes, EMA_FAST)
            ema_slow = self._calculate_ema(closes, EMA_SLOW)
            
            # Calculate ATR for volatility-adaptive EMA-distance filter
            atr_values = self._calculate_atr(ohlcv, 14)
            atr_value = atr_values[-1] if atr_values else 0
            
            # Calculate bullish indicators with volatility-adaptive EMA-distance filter
            ema_crossover = False
            ema_distance_passes = False
            
            if len(ema_fast) >= 2 and len(ema_slow) >= 2:
                # Check EMA crossover (existing logic)
                ema_crossover = (ema_fast[-2] <= ema_slow[-2] and ema_fast[-1] > ema_slow[-1])
                
                # Enhanced: Volatility-adaptive EMA-distance filter
                if ema_crossover and atr_value > 0:
                    current_price = closes[-1]
                    ema_gap = abs(ema_fast[-1] - ema_slow[-1]) / current_price
                    required_gap = (atr_value / current_price) * 0.1  # Normalize ATR by price, then multiply by 0.1
                    ema_distance_passes = ema_gap >= required_gap
                    
                    # Only pass if both crossover AND distance requirements are met
                    ema_crossover = ema_crossover and ema_distance_passes
                else:
                    ema_distance_passes = False
            
            rsi_values = self._calculate_rsi(closes, 14)
            rsi_value = rsi_values[-1] if rsi_values else 50
            
            rsi_reversal = False
            if len(rsi_values) >= 2:
                prev_rsi = rsi_values[-2]
                rsi_reversal = prev_rsi <= RSI_OVERSOLD and rsi_value > prev_rsi  # RSI bounce from oversold
            
            atr_valid = atr_value >= self.min_atr_threshold  # Meaningful ATR validation
            
            macd_line, macd_signal = self._calculate_macd(closes, 12, 26, 9)
            macd_bullish = macd_line > macd_signal if macd_line is not None and macd_signal is not None else False
            
            # Calculate bearish indicators (inverted logic) with volatility-adaptive EMA-distance filter
            ema_crossunder = False
            ema_distance_passes_bearish = False
            
            if len(ema_fast) >= 2 and len(ema_slow) >= 2:
                # Check EMA crossunder (existing logic)
                ema_crossunder = (ema_fast[-2] >= ema_slow[-2] and ema_fast[-1] < ema_slow[-1])
                
                # Enhanced: Volatility-adaptive EMA-distance filter for bearish signals
                if ema_crossunder and atr_value > 0:
                    current_price = closes[-1]
                    ema_gap = abs(ema_fast[-1] - ema_slow[-1]) / current_price
                    required_gap = (atr_value / current_price) * 0.1  # Normalize ATR by price, then multiply by 0.1
                    ema_distance_passes_bearish = ema_gap >= required_gap
                    
                    # Only pass if both crossunder AND distance requirements are met
                    ema_crossunder = ema_crossunder and ema_distance_passes_bearish
                else:
                    ema_distance_passes_bearish = False
            
            rsi_falling = False
            if len(rsi_values) >= 2:
                prev_rsi = rsi_values[-2]
                rsi_falling = prev_rsi >= RSI_OVERBOUGHT and rsi_value < prev_rsi  # RSI falling from overbought
            
            macd_bearish = macd_line < macd_signal if macd_line is not None and macd_signal is not None else False
            
            # Calculate ADX for regime detection
            adx_value = self._calculate_adx(highs, lows, closes, 14)
            
            # Calculate volatility rank (ATR/SMA*100)
            volatility_rank = self._calculate_volatility_rank(closes, atr_value)
            
            return {
                'ema_crossover': ema_crossover, 'rsi_reversal': rsi_reversal, 'atr_valid': atr_valid,
                'macd_bullish': macd_bullish, 'ema_crossunder': ema_crossunder, 'rsi_falling': rsi_falling,
                'macd_bearish': macd_bearish, 'atr_value': atr_value, 'rsi_value': rsi_value,
                'ema_fast': ema_fast[-1] if ema_fast else 0,
                'ema_slow': ema_slow[-1] if ema_slow else 0,
                'macd_line': macd_line if macd_line is not None else 0,
                'macd_signal': macd_signal if macd_signal is not None else 0,
                'adx_value': adx_value,
                'volatility_rank': volatility_rank,
                'ema_distance_passes': ema_distance_passes,
                'ema_distance_passes_bearish': ema_distance_passes_bearish
            }
        except Exception as e:
            self.logger.error(f"Error calculating Stage 2 indicators: {e}")
            return {
                'ema_crossover': False, 'rsi_reversal': False, 'atr_valid': False,
                'macd_bullish': False, 'ema_crossunder': False, 'rsi_falling': False,
                'macd_bearish': False, 'atr_value': 0, 'rsi_value': 0,
                'ema_fast': 0, 'ema_slow': 0, 'macd_line': 0, 'macd_signal': 0,
                'adx_value': 0, 'volatility_rank': 0,
                'ema_distance_passes': False, 'ema_distance_passes_bearish': False
            }
    
    def _calculate_ema(self, prices: List[float], period: int) -> List[float]:
        """Calculate Exponential Moving Average"""
        if len(prices) < period or period <= 0:
            return []
        
        ema_values = []
        multiplier = 2.0 / (period + 1)
        
        # First EMA is SMA
        sma = sum(prices[:period]) / period
        ema_values.append(sma)
        
        # Calculate EMA for remaining prices
        for i in range(1, len(prices)):
            ema = (prices[i] * multiplier) + (ema_values[-1] * (1 - multiplier))
            ema_values.append(ema)
        
        return ema_values
    
    def _calculate_rsi(self, prices: List[float], period: int) -> List[float]:
        """Calculate Relative Strength Index"""
        if len(prices) < period + 1 or period <= 0:
            return []
        
        rsi_values = []
        gains = []
        losses = []
        
        # Calculate price changes
        for i in range(1, len(prices)):
            change = prices[i] - prices[i-1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))
        
        # Calculate initial average gain and loss
        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period
        
        # Calculate RSI for first period
        if avg_loss == 0:
            rsi = 100
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        rsi_values.append(rsi)
        
        # Calculate RSI for remaining periods
        for i in range(period, len(gains)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
            
            if avg_loss == 0:
                rsi = 100
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
            rsi_values.append(rsi)
        
        return rsi_values
    
    def _calculate_atr(self, ohlcv: List, period: int) -> List[float]:
        """Calculate Average True Range"""
        if len(ohlcv) < period + 1 or period <= 0:
            return []
        
        atr_values = []
        true_ranges = []
        
        # Calculate True Range for each period
        for i in range(1, len(ohlcv)):
            try:
                high = float(ohlcv[i][2])
                low = float(ohlcv[i][3])
                prev_close = float(ohlcv[i-1][4])
                
                tr1 = high - low
                tr2 = abs(high - prev_close)
                tr3 = abs(low - prev_close)
                true_range = max(tr1, tr2, tr3)
                true_ranges.append(true_range)
            except (IndexError, ValueError, TypeError):
                continue
        
        if len(true_ranges) < period:
            return []
        
        # Calculate initial ATR (SMA of first 'period' true ranges)
        atr = sum(true_ranges[:period]) / period
        atr_values.append(atr)
        
        # Calculate ATR for remaining periods
        for i in range(period, len(true_ranges)):
            atr = (atr * (period - 1) + true_ranges[i]) / period
            atr_values.append(atr)
        
        return atr_values

    def _calculate_macd(self, prices: List[float], fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> Tuple[float, float]:
        """Calculate MACD line and signal line"""
        if len(prices) < slow_period + signal_period:
            return None, None
        
        # Calculate fast and slow EMAs
        ema_fast = self._calculate_ema(prices, fast_period)
        ema_slow = self._calculate_ema(prices, slow_period)
        
        if len(ema_fast) == 0 or len(ema_slow) == 0:
            return None, None
        
        # Calculate MACD line (fast EMA - slow EMA)
        macd_line = ema_fast[-1] - ema_slow[-1]
        
        # Calculate MACD signal line (EMA of MACD line)
        # We need to calculate MACD values for the signal period
        macd_values = []
        for i in range(len(ema_fast)):
            if i < len(ema_slow):
                macd_values.append(ema_fast[i] - ema_slow[i])
        
        if len(macd_values) < signal_period:
            return macd_line, None
        
        # Calculate signal line (EMA of MACD values)
        signal_line_values = self._calculate_ema(macd_values, signal_period)
        signal_line = signal_line_values[-1] if signal_line_values else None
        
        return macd_line, signal_line
    
    def _calculate_adx(self, highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> float:
        """Calculate Average Directional Index (ADX)"""
        try:
            if len(highs) < period + 1 or len(lows) < period + 1 or len(closes) < period + 1:
                return 0.0
            
            # Calculate True Range (TR)
            tr_values = []
            for i in range(1, len(closes)):
                high_low = highs[i] - lows[i]
                high_close = abs(highs[i] - closes[i-1])
                low_close = abs(lows[i] - closes[i-1])
                tr_values.append(max(high_low, high_close, low_close))
            
            if len(tr_values) < period:
                return 0.0
            
            # Calculate Directional Movement (DM)
            dm_plus = []
            dm_minus = []
            
            for i in range(1, len(closes)):
                up_move = highs[i] - highs[i-1]
                down_move = lows[i-1] - lows[i]
                
                if up_move > down_move and up_move > 0:
                    dm_plus.append(up_move)
                else:
                    dm_plus.append(0)
                
                if down_move > up_move and down_move > 0:
                    dm_minus.append(down_move)
                else:
                    dm_minus.append(0)
            
            if len(dm_plus) < period or len(dm_minus) < period:
                return 0.0
            
            # Calculate smoothed values
            atr = sum(tr_values[-period:]) / period
            di_plus = (sum(dm_plus[-period:]) / period) / atr * 100 if atr > 0 else 0
            di_minus = (sum(dm_minus[-period:]) / period) / atr * 100 if atr > 0 else 0
            
            # Calculate ADX
            dx = abs(di_plus - di_minus) / (di_plus + di_minus) * 100 if (di_plus + di_minus) > 0 else 0
            adx = dx  # Simplified ADX calculation
            
            return adx
            
        except Exception as e:
            self.logger.error(f"Error calculating ADX: {e}")
            return 0.0
    
    def _calculate_volatility_rank(self, closes: List[float], atr_value: float) -> float:
        """Calculate volatility rank = ATR(14)/SMA(close,20)*100"""
        try:
            if len(closes) < 20 or atr_value <= 0:
                return 0.0
            
            # Calculate SMA of closes (20 periods)
            sma_close = sum(closes[-20:]) / 20
            
            # Calculate volatility rank
            volatility_rank = (atr_value / sma_close) * 100 if sma_close > 0 else 0
            
            return volatility_rank
            
        except Exception as e:
            self.logger.error(f"Error calculating volatility rank: {e}")
            return 0.0
    
    def trap_trade(self, ohlcv: List, technical_data: Dict) -> bool:
        """
        Trap Trading Strategy
        - Price breaks above resistance with high volume
        - RSI shows momentum but not overbought
        - MACD confirms bullish momentum
        """
        try:
            if len(ohlcv) < 20:
                return False
            
            closes = [float(candle[4]) for candle in ohlcv]
            volumes = [float(candle[5]) for candle in ohlcv]
            
            # Check for resistance break
            recent_highs = max(closes[-5:])
            current_price = closes[-1]
            resistance_break = current_price > recent_highs * 0.995  # Within 0.5% of recent high
            
            # Check volume spike
            avg_volume = sum(volumes[-20:]) / 20
            current_volume = volumes[-1]
            volume_spike = current_volume > avg_volume * 1.2  # 20% above average
            
            # Check RSI momentum (not overbought)
            rsi_value = technical_data.get('rsi_value', 50)
            rsi_momentum = 30 < rsi_value < 70  # Not overbought or oversold
            
            # Check MACD bullish
            macd_bullish = technical_data.get('macd_bullish', False)
            
            # Trap trade conditions
            trap_conditions = [
                resistance_break,
                volume_spike,
                rsi_momentum,
                macd_bullish
            ]
            
            # At least 3 of 4 conditions must be met
            return sum(trap_conditions) >= 3
            
        except Exception as e:
            self.logger.error(f"Error in trap_trade: {e}")
            return False
    
    def smc_trade(self, ohlcv: List, technical_data: Dict) -> bool:
        """
        Smart Money Concept (SMC) Strategy
        - Price action shows institutional accumulation
        - Volume profile indicates smart money activity
        - Support/resistance levels respected
        """
        try:
            if len(ohlcv) < 30:
                return False
            
            closes = [float(candle[4]) for candle in ohlcv]
            volumes = [float(candle[5]) for candle in ohlcv]
            highs = [float(candle[2]) for candle in ohlcv]
            lows = [float(candle[3]) for candle in ohlcv]
            
            # Check for accumulation pattern (higher lows)
            recent_lows = lows[-10:]
            higher_lows = all(recent_lows[i] >= recent_lows[i-1] for i in range(1, len(recent_lows)))
            
            # Check for volume increase on up moves
            up_volume = 0
            down_volume = 0
            for i in range(1, len(closes)):
                if closes[i] > closes[i-1]:
                    up_volume += volumes[i]
                else:
                    down_volume += volumes[i]
            
            volume_preference = up_volume > down_volume * 1.1  # 10% more volume on up moves
            
            # Check for tight range (institutional accumulation)
            recent_range = max(highs[-10:]) - min(lows[-10:])
            avg_price = sum(closes[-10:]) / 10
            tight_range = recent_range < avg_price * 0.05  # Less than 5% range
            
            # Check ADX for trend strength
            adx_value = technical_data.get('adx_value', 0)
            trend_strength = adx_value > 20  # Moderate trend strength
            
            # SMC conditions
            smc_conditions = [
                higher_lows,
                volume_preference,
                tight_range,
                trend_strength
            ]
            
            # At least 3 of 4 conditions must be met
            return sum(smc_conditions) >= 3
            
        except Exception as e:
            self.logger.error(f"Error in smc_trade: {e}")
            return False
    
    def scalp_trade(self, ohlcv: List, technical_data: Dict) -> bool:
        """
        Scalping Strategy
        - High volatility environment
        - Quick momentum shifts
        - Tight spreads and high liquidity
        """
        try:
            if len(ohlcv) < 20:
                return False
            
            closes = [float(candle[4]) for candle in ohlcv]
            volumes = [float(candle[5]) for candle in ohlcv]
            
            # Check for high volatility
            atr_value = technical_data.get('atr_value', 0)
            avg_price = sum(closes[-20:]) / 20
            high_volatility = atr_value > avg_price * 0.02  # 2% ATR minimum
            
            # Check for momentum shifts (price changes direction quickly)
            price_changes = []
            for i in range(1, len(closes)):
                change = (closes[i] - closes[i-1]) / closes[i-1] * 100
                price_changes.append(abs(change))
            
            avg_change = sum(price_changes[-10:]) / 10
            momentum_shifts = avg_change > 1.0  # Average 1% price changes
            
            # Check for high volume relative to recent average
            avg_volume = sum(volumes[-20:]) / 20
            current_volume = volumes[-1]
            high_volume = current_volume > avg_volume * 1.3  # 30% above average
            
            # Check RSI for quick reversals
            rsi_value = technical_data.get('rsi_value', 50)
            rsi_volatility = 20 < rsi_value < 80  # Not extreme levels
            
            # Scalping conditions
            scalp_conditions = [
                high_volatility,
                momentum_shifts,
                high_volume,
                rsi_volatility
            ]
            
            # At least 3 of 4 conditions must be met
            return sum(scalp_conditions) >= 3
            
        except Exception as e:
            self.logger.error(f"Error in scalp_trade: {e}")
            return False
    
    def check_volume_spike(self, ohlcv: List) -> bool:
        """
        Check for volume spike: volume > vol_sma * 1.5
        """
        try:
            if len(ohlcv) < 20:
                return False
            
            volumes = [float(candle[5]) for candle in ohlcv]
            current_volume = volumes[-1]
            
            # Calculate volume SMA (20 periods)
            vol_sma = sum(volumes[-20:]) / 20
            
            # Check for volume spike
            volume_spike = current_volume > vol_sma * VOLUME_SPIKE_MULTIPLIER
            
            return volume_spike
            
        except Exception as e:
            self.logger.error(f"Error checking volume spike: {e}")
            return False
    
    def stage_2_filter_symbols(self, stage1_symbols: List[Dict], scanner, signal_engine) -> List[Dict]:
        """
        Stage 2 Filter: Apply technical analysis to Stage 1 symbols
        Args:
            stage1_symbols: List of symbols that passed Stage 1
            scanner: MarketScanner instance for fetching OHLCV data
            signal_engine: SignalEngine instance for technical calculations
        Returns:
            List of symbols that passed Stage 2 with strength dot 2
        """
        try:
            if not stage1_symbols:
                self.logger.info("Stage 2: No symbols from Stage 1 to process")
                return []
            
            self.logger.info(f"Stage 2: Processing {len(stage1_symbols)} symbols from Stage 1")
            
            passing_symbols = []
            
            for symbol_data in stage1_symbols:
                symbol = symbol_data.get('symbol', '')
                
                try:
                    # Fetch recent OHLCV data
                    ohlcv = scanner.get_ohlcv_data(symbol)
                    if ohlcv is None or len(ohlcv) < 50:
                        self.logger.debug(f"Stage 2: {symbol} - Insufficient OHLCV data")
                        continue
                    
                    # Calculate Stage 2 technical indicators (simplified without pandas/numpy)
                    technical_data = self._calculate_stage_2_indicators(ohlcv)
                    
                    # Apply Stage 2 filter
                    passed, details = self.stage_2_filter(technical_data)
                    
                    if passed:
                        # Update symbol data with Stage 2 results
                        filtered_symbol = {
                            'symbol': symbol,
                            'stage_1_passed': True,
                            'stage_2_passed': True,
                            'strength_dot': 2,  # Now has 2 green dots
                            'stage_1_details': symbol_data.get('filter_details', {}),
                            'stage_2_details': details,
                            'volume': symbol_data.get('volume', 0),
                            'funding_rate': symbol_data.get('funding_rate', 0),
                            'spread': symbol_data.get('spread', 0),
                            'volume_rank': symbol_data.get('volume_rank', 0),
                            'atr_value': details.get('atr_value', 0),
                            'rsi_value': details.get('rsi_value', 0),
                            'ema_fast': details.get('ema_fast', 0),
                            'ema_slow': details.get('ema_slow', 0)
                        }
                        passing_symbols.append(filtered_symbol)
                        
                        self.logger.info(f"Stage 2 PASS: {symbol} - EMA: {details.get('ema_crossover')}, "
                                       f"RSI: {details.get('rsi_reversal')}, ATR: {details.get('atr_valid')}")
                    else:
                        self.logger.debug(f"Stage 2 FAIL: {symbol} - EMA: {details.get('ema_crossover')}, "
                                        f"RSI: {details.get('rsi_reversal')}, ATR: {details.get('atr_valid')}")
                
                except Exception as e:
                    self.logger.error(f"Stage 2: Error processing {symbol}: {e}")
                    continue
            
            self.logger.info(f"Stage 2 Filter: {len(passing_symbols)}/{len(stage1_symbols)} symbols passed")
            return passing_symbols
            
        except Exception as e:
            self.logger.error(f"Error in stage 2 filter symbols: {e}")
            return []
    
    def stage_2_filter_with_stored_data(self, stage1_symbols_data: List[Dict], adx_threshold: int = 25) -> List[Dict]:
        """
        Stage 2: Dual-indicator confluence using stored OHLCV data (no API calls)
        Args:
            stage1_symbols_data: List of symbols that passed Stage 1
            adx_threshold: Dynamic ADX threshold
        Returns:
            List of symbols that passed Stage 2 filter
        """
        try:
            if not stage1_symbols_data:
                raise ValueError("No Stage 1 symbols provided to Stage 2 filter")
            
            # CRITICAL QUALITY CHECK: Validate Stage 1 data
            required_fields = ['symbol', 'ohlcv']
            for i, symbol_data in enumerate(stage1_symbols_data):
                missing_fields = [field for field in required_fields if field not in symbol_data]
                if missing_fields:
                    raise ValueError(f"Stage 1 symbol {i} missing critical fields: {missing_fields}")
                
                # Validate OHLCV data quality
                ohlcv = symbol_data.get('ohlcv')
                if not ohlcv:
                    raise ValueError(f"Stage 1 symbol {i} ({symbol_data['symbol']}) missing OHLCV data")
                
                if len(ohlcv) < 50:
                    raise ValueError(f"Stage 1 symbol {i} ({symbol_data['symbol']}) insufficient OHLCV data: {len(ohlcv)} candles (minimum 50 required)")
                
                # Validate OHLCV structure
                for j, candle in enumerate(ohlcv):
                    if len(candle) != 6:
                        raise ValueError(f"Stage 1 symbol {i} ({symbol_data['symbol']}) invalid OHLCV candle {j}: expected 6 values, got {len(candle)}")
                    
                    # Validate numeric values
                    for k, value in enumerate(candle):
                        if not isinstance(value, (int, float)):
                            raise ValueError(f"Stage 1 symbol {i} ({symbol_data['symbol']}) OHLCV candle {j} value {k} must be numeric, got {type(value)}")
            
            stage2_passes = []
            
            for symbol_data in stage1_symbols_data:
                symbol = symbol_data['symbol']
                ohlcv = symbol_data.get('ohlcv')
                
                try:
                    # Calculate Stage 2 indicators using stored OHLCV data
                    technical_data = self._calculate_stage_2_indicators(ohlcv)
                    
                    # CRITICAL QUALITY CHECK: Validate technical data
                    required_technical_fields = ['ema_crossover', 'rsi_reversal', 'atr_valid', 'macd_bullish', 'adx_value']
                    missing_technical = [field for field in required_technical_fields if field not in technical_data]
                    if missing_technical:
                        self.logger.warning(f"CRITICAL: {symbol} missing technical indicators: {missing_technical} - skipping")
                        continue
                    
                    # Apply Stage 2 filter
                    passed, details = self.stage_2_filter(technical_data, adx_threshold)
                    
                    if passed:
                        # Add technical data and signal direction to symbol data
                        symbol_data['technical_data'] = technical_data
                        symbol_data['signal_direction'] = details.get('signal_direction', 'LONG')
                        stage2_passes.append(symbol_data)
                        
                        signal_direction = details.get('signal_direction', 'LONG')
                        long_passes = details.get('long_indicator_passes', 0)
                        short_passes = details.get('short_indicator_passes', 0)
                        regime_type = details.get('regime_type', 'NORMAL')
                        
                        self.logger.info(f"Stage 2 PASS: {symbol} - {signal_direction} signal - Long: {long_passes}/3, Short: {short_passes}/3, Regime: {regime_type}")
                    
                except Exception as e:
                    self.logger.error(f"CRITICAL: Error processing {symbol} in Stage 2: {e}")
                    continue
            
            # CRITICAL QUALITY CHECK: Ensure minimum pass rate
            pass_rate = len(stage2_passes) / len(stage1_symbols_data) * 100
            if pass_rate < 0.5:  # Less than 0.5% pass rate indicates potential issues
                self.logger.warning(f"CRITICAL: Very low Stage 2 pass rate: {pass_rate:.1f}% - may indicate overly strict criteria")
            
            self.logger.info(f"Stage 2 Filter: {len(stage2_passes)}/{len(stage1_symbols_data)} symbols passed ({pass_rate:.1f}% pass rate)")
            return stage2_passes
            
        except Exception as e:
            self.logger.error(f"CRITICAL ERROR in stage 2 filter: {e}")
            raise  # Re-raise to fail fast
    
    def stage_3_filter(self, signal_data: Dict) -> Tuple[bool, Dict]:
        """
        Stage 3: Order block/SMC confluence and Fibonacci levels
        Returns: (passed, details)
        """
        try:
            order_block = signal_data.get('order_block', False)
            smc_confluence = signal_data.get('smc_confluence', False)
            fibonacci_levels = signal_data.get('fibonacci_levels', False)
            confidence = signal_data.get('confidence', 0)
            risk_reward = signal_data.get('risk_reward', 0)
            
            # Check SMC criteria
            smc_passed = order_block and smc_confluence and fibonacci_levels
            
            # Check confidence threshold
            confidence_passed = confidence >= self.min_confidence
            
            # Check risk-reward ratio
            risk_reward_passed = risk_reward >= self.min_risk_reward
            
            # Overall stage 3 result
            stage_passed = smc_passed and confidence_passed and risk_reward_passed
            
            details = {
                'order_block': order_block,
                'smc_confluence': smc_confluence,
                'fibonacci_levels': fibonacci_levels,
                'confidence_passed': confidence_passed,
                'risk_reward_passed': risk_reward_passed,
                'confidence': confidence,
                'risk_reward': risk_reward
            }
            
            return stage_passed, details
            
        except Exception as e:
            self.logger.error(f"Error in stage 3 filter: {e}")
            return False, {'error': str(e)}
    
    def calculate_strength_score(self, stage_results: List[Tuple[bool, Dict]]) -> int:
        """
        Calculate signal strength based on stages passed (1-3 green dots)
        """
        passed_stages = sum(1 for passed, _ in stage_results if passed)
        return passed_stages
    
    def apply_filters(self, market_data: Dict, technical_data: Dict, signal_data: Dict) -> Dict:
        """
        Apply all three stages of filtering
        Returns: Complete filter results with strength score
        """
        try:
            # Apply each stage
            stage1_result = self.stage_1_filter(market_data)
            stage2_result = self.stage_2_filter(technical_data)
            stage3_result = self.stage_3_filter(signal_data)
            
            # Calculate strength score
            strength_score = self.calculate_strength_score([stage1_result, stage2_result, stage3_result])
            
            # Overall result
            all_stages_passed = all(passed for passed, _ in [stage1_result, stage2_result, stage3_result])
            
            results = {
                'overall_passed': all_stages_passed,
                'strength_score': strength_score,
                'stage_1': {
                    'passed': stage1_result[0],
                    'details': stage1_result[1]
                },
                'stage_2': {
                    'passed': stage2_result[0],
                    'details': stage2_result[1]
                },
                'stage_3': {
                    'passed': stage3_result[0],
                    'details': stage3_result[1]
                },
                'timestamp': datetime.now().isoformat()
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error applying filters: {e}")
            return {
                'overall_passed': False,
                'strength_score': 0,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            } 

    def stage_3_strategy_filter(self, stage2_results: List[Dict], timeframe: str) -> List[Dict]:
        """
        Stage 3: Strategy-specific filters with regime-aware confidence thresholds
        • Trap Trading: Price action near key levels
        • Smart Money Concept (SMC): Institutional order flow patterns
        • Scalping: High-frequency momentum trades
        • Volume Spike: Unusual volume activity
        • REGIME-AWARE: High-vol regime requires ≥5/7 confidence
        • REGIME-AWARE: Normal regime requires ≥2/7 confidence (LOWERED)
        Args:
            stage2_results: Symbols that passed Stage 2
            timeframe: Current timeframe being scanned
        Returns: List of symbols that passed Stage 3
        """
        try:
            if not stage2_results:
                self.logger.info("Stage 3: No Stage 2 results to process")
                return []
            
            self.logger.info(f"Stage 3: Processing {len(stage2_results)} Stage 2 symbols")
            stage3_results = []
            
            for symbol_data in stage2_results:
                try:
                    symbol = symbol_data.get('symbol', 'Unknown')
                    technical_data = symbol_data.get('technical_data', {})
                    signal_direction = symbol_data.get('signal_direction', 'LONG')
                    
                    # Apply strategy filters
                    trap_result = self.trap_trade(symbol_data)
                    smc_result = self.smc_trade(symbol_data)
                    scalp_result = self.scalp_trade(symbol_data)
                    vol_spike = self.check_volume_spike(symbol_data)
                    
                    # Create stage3_data dict for confidence calculation
                    stage3_data = {
                        'trap': trap_result,
                        'smc': smc_result,
                        'scalp': scalp_result,
                        'vol_spike': vol_spike,
                        'signal_direction': signal_direction
                    }
                    
                    # Calculate confidence layers (7 total)
                    confidence_layers = self.calculate_confidence_layers(technical_data, stage3_data)
                    
                    # Determine regime for confidence threshold
                    adx_value = technical_data.get('adx_value', 0)
                    volatility_rank = technical_data.get('volatility_rank', 0)
                    high_adx = adx_value > 25
                    high_volatility = volatility_rank > 80
                    regime_high_activity = high_adx and high_volatility
                    
                    # Apply regime-aware confidence threshold (LOWERED for normal regime)
                    if regime_high_activity:
                        required_confidence = 5  # High-vol regime: ≥5/7
                        regime_type = "HIGH_VOL"
                    else:
                        required_confidence = 2  # Normal regime: ≥2/7 (LOWERED from 3)
                        regime_type = "NORMAL"
                    
                    # Check if confidence meets regime requirement
                    if confidence_layers >= required_confidence:
                        # Add strategy and confidence data
                        symbol_data['stage3_data'] = {
                            'trap': trap_result,
                            'smc': smc_result,
                            'scalp': scalp_result,
                            'vol_spike': vol_spike,
                            'confidence_layers': confidence_layers,
                            'regime_type': regime_type,
                            'required_confidence': required_confidence
                        }
                        symbol_data['confidence'] = confidence_layers
                        symbol_data['strategy'] = self._determine_primary_strategy(trap_result, smc_result, scalp_result, vol_spike)
                        stage3_results.append(symbol_data)
                        
                        self.logger.info(f"Stage 3 PASS: {symbol} - Confidence: {confidence_layers}/7, Regime: {regime_type}, Strategy: {symbol_data['strategy']}")
                    else:
                        self.logger.info(f"Stage 3 REJECT: {symbol} - Confidence: {confidence_layers}/7 < {required_confidence}/7 (Regime: {regime_type})")
                        
                except Exception as e:
                    self.logger.error(f"Error processing {symbol_data.get('symbol', 'Unknown')} in Stage 3: {e}")
                    continue
            
            self.logger.info(f"Stage 3: {len(stage3_results)} symbols passed out of {len(stage2_results)}")
            return stage3_results
            
        except Exception as e:
            self.logger.error(f"Error in Stage 3 strategy filter: {e}")
            return []
    
    def calculate_confidence_layers(self, technical_data: Dict, stage3_data: Dict) -> int:
        """
        Calculate confidence layers (7 total validation layers)
        • EMA crossover/crossunder (1 layer)
        • RSI reversal/falling (1 layer) 
        • MACD bullish/bearish (1 layer)
        • ADX > 20 (1 layer)
        • Volatility rank > 50 (1 layer)
        • EMA distance bonus (1 layer) - volatility-adaptive gap
        • Orderblock alignment bonus (1 layer) - polarity match
        Args:
            technical_data: Technical indicator data
            stage3_data: Stage 3 strategy results
        Returns: Number of confidence layers (0-7)
        """
        try:
            confidence_layers = 0
            signal_direction = stage3_data.get('signal_direction', 'LONG')
            
            # Technical indicator layers (3 layers) - direction-aware
            if signal_direction == 'LONG':
                if technical_data.get('ema_crossover', False):
                    confidence_layers += 1
                if technical_data.get('rsi_reversal', False):
                    confidence_layers += 1
                if technical_data.get('macd_bullish', False):
                    confidence_layers += 1
            else:  # SHORT
                if technical_data.get('ema_crossunder', False):
                    confidence_layers += 1
                if technical_data.get('rsi_falling', False):
                    confidence_layers += 1
                if technical_data.get('macd_bearish', False):
                    confidence_layers += 1
            
            # Market condition layers (2 layers) - direction-agnostic
            if technical_data.get('adx_value', 0) > 20:
                confidence_layers += 1
            if technical_data.get('volatility_rank', 0) > 50:
                confidence_layers += 1
            
            # EMA distance bonus (1 layer) - volatility-adaptive gap
            if signal_direction == 'LONG' and technical_data.get('ema_distance_passes', False):
                confidence_layers += 1
            elif signal_direction == 'SHORT' and technical_data.get('ema_distance_passes_bearish', False):
                confidence_layers += 1
            
            # Orderblock alignment bonus (1 layer) - polarity match
            if stage3_data.get('orderblock_bonus', False):
                orderblock_type = stage3_data.get('orderblock_type', '')
                if (signal_direction == 'LONG' and orderblock_type == 'bullish') or \
                   (signal_direction == 'SHORT' and orderblock_type == 'bearish'):
                    confidence_layers += 1
            
            # Cap confidence at 7 layers
            confidence_layers = min(confidence_layers, 7)
            
            return confidence_layers
            
        except Exception as e:
            self.logger.error(f"Error calculating confidence layers: {e}")
            return 0
    
    def _determine_primary_strategy(self, trap: bool, smc: bool, scalp: bool, vol_spike: bool) -> str:
        """Determine the primary strategy based on which filters passed"""
        try:
            strategies = []
            if trap:
                strategies.append('Trap')
            if smc:
                strategies.append('SMC')
            if scalp:
                strategies.append('Scalp')
            if vol_spike:
                strategies.append('VolSpike')
            
            if strategies:
                return '+'.join(strategies)
            else:
                return 'Technical'
                
        except Exception as e:
            self.logger.error(f"Error determining primary strategy: {e}")
            return 'Unknown' 

    def correlation_throttle(self, stage3_results: List[Dict], scanner) -> List[Dict]:
        """
        Correlation throttle: Remove highly correlated signals and enforce sector limits
        • Fetch 1h returns over past 24h for each symbol
        • Build correlation matrix
        • Drop signals with corr > 0.8 (keep higher confidence)
        • Enforce max 3 signals per sector per day
        Args:
            stage3_results: Symbols that passed Stage 3
            scanner: Market scanner instance
        Returns: Filtered list of symbols
        """
        try:
            if len(stage3_results) <= 1:
                return stage3_results
            
            self.logger.info(f"Correlation throttle: Processing {len(stage3_results)} Stage 3 symbols")
            
            # Get 1h returns for correlation calculation
            returns_data = {}
            for symbol_data in stage3_results:
                symbol = symbol_data.get('symbol', 'Unknown')
                try:
                    # Get 24h of 1h data (24 candles)
                    ohlcv = scanner.get_ohlcv_data(symbol, timeframe='1h', limit=24)
                    if ohlcv and len(ohlcv) >= 24:
                        # Calculate returns
                        returns = []
                        for i in range(1, len(ohlcv)):
                            prev_close = float(ohlcv[i-1][4])
                            curr_close = float(ohlcv[i][4])
                            if prev_close > 0:
                                ret = (curr_close - prev_close) / prev_close
                                returns.append(ret)
                        
                        if len(returns) >= 20:  # Need at least 20 data points
                            returns_data[symbol] = returns
                except Exception as e:
                    self.logger.warning(f"Error getting returns for {symbol}: {e}")
                    continue
            
            if len(returns_data) < 2:
                self.logger.info("Correlation throttle: Insufficient data for correlation analysis")
                return stage3_results
            
            # Build correlation matrix
            symbols = list(returns_data.keys())
            correlation_matrix = {}
            
            for i, symbol1 in enumerate(symbols):
                correlation_matrix[symbol1] = {}
                for j, symbol2 in enumerate(symbols):
                    if i == j:
                        correlation_matrix[symbol1][symbol2] = 1.0
                    else:
                        # Calculate correlation
                        corr = self._calculate_correlation(returns_data[symbol1], returns_data[symbol2])
                        correlation_matrix[symbol1][symbol2] = corr
            
            # Find and remove highly correlated signals
            signals_to_remove = set()
            
            for i, symbol1 in enumerate(symbols):
                if symbol1 in signals_to_remove:
                    continue
                    
                for j, symbol2 in enumerate(symbols[i+1:], i+1):
                    if symbol2 in signals_to_remove:
                        continue
                    
                    corr = correlation_matrix[symbol1][symbol2]
                    if corr > 0.8:  # High correlation threshold
                        # Keep the signal with higher confidence
                        symbol1_data = next((s for s in stage3_results if s.get('symbol') == symbol1), None)
                        symbol2_data = next((s for s in stage3_results if s.get('symbol') == symbol2), None)
                        
                        if symbol1_data and symbol2_data:
                            confidence1 = symbol1_data.get('confidence', 0)
                            confidence2 = symbol2_data.get('confidence', 0)
                            
                            if confidence1 >= confidence2:
                                signals_to_remove.add(symbol2)
                                self.logger.info(f"Correlation throttle: Removed {symbol2} (corr={corr:.3f} with {symbol1}, lower confidence)")
                            else:
                                signals_to_remove.add(symbol1)
                                self.logger.info(f"Correlation throttle: Removed {symbol1} (corr={corr:.3f} with {symbol2}, lower confidence)")
            
            # Apply sector limits (max 3 per sector)
            sector_counts = {}
            sector_signals = {}
            
            for symbol_data in stage3_results:
                if symbol_data.get('symbol') in signals_to_remove:
                    continue
                    
                symbol = symbol_data.get('symbol', 'Unknown')
                sector = self._get_symbol_sector(symbol)
                
                if sector not in sector_counts:
                    sector_counts[sector] = 0
                    sector_signals[sector] = []
                
                sector_counts[sector] += 1
                sector_signals[sector].append(symbol_data)
            
            # Enforce max 3 signals per sector
            final_signals = []
            for sector, signals in sector_signals.items():
                if len(signals) > 3:
                    # Sort by confidence and keep top 3
                    signals.sort(key=lambda x: x.get('confidence', 0), reverse=True)
                    final_signals.extend(signals[:3])
                    self.logger.info(f"Correlation throttle: Limited {sector} sector to 3 signals (had {len(signals)})")
                else:
                    final_signals.extend(signals)
            
            self.logger.info(f"Correlation throttle: {len(stage3_results)} -> {len(final_signals)} signals after filtering")
            return final_signals
            
        except Exception as e:
            self.logger.error(f"Error in correlation throttle: {e}")
            return stage3_results
    
    def _calculate_correlation(self, returns1: List[float], returns2: List[float]) -> float:
        """Calculate correlation coefficient between two return series"""
        try:
            if len(returns1) != len(returns2) or len(returns1) < 2:
                return 0.0
            
            n = len(returns1)
            sum_x = sum(returns1)
            sum_y = sum(returns2)
            sum_xy = sum(returns1[i] * returns2[i] for i in range(n))
            sum_x2 = sum(x * x for x in returns1)
            sum_y2 = sum(y * y for y in returns2)
            
            numerator = n * sum_xy - sum_x * sum_y
            denominator = ((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)) ** 0.5
            
            if denominator == 0:
                return 0.0
            
            return numerator / denominator
            
        except Exception as e:
            self.logger.error(f"Error calculating correlation: {e}")
            return 0.0
    
    def _get_symbol_sector(self, symbol: str) -> str:
        """Get sector classification for a symbol"""
        try:
            # Simple sector classification based on symbol
            if any(coin in symbol.upper() for coin in ['BTC', 'ETH', 'BNB', 'ADA', 'SOL', 'DOT', 'AVAX']):
                return 'Layer1'
            elif any(coin in symbol.upper() for coin in ['LINK', 'UNI', 'AAVE', 'COMP', 'SUSHI']):
                return 'DeFi'
            elif any(coin in symbol.upper() for coin in ['DOGE', 'SHIB', 'PEPE', 'FLOKI']):
                return 'Meme'
            elif any(coin in symbol.upper() for coin in ['MATIC', 'ARB', 'OP', 'IMX']):
                return 'Layer2'
            elif any(coin in symbol.upper() for coin in ['XRP', 'LTC', 'BCH', 'XLM']):
                return 'Payment'
            else:
                return 'Other'
                
        except Exception as e:
            self.logger.error(f"Error getting sector for {symbol}: {e}")
            return 'Unknown' 

    def stage_3_filter_with_stored_data(self, stage2_symbols_data: List[Dict], confidence_threshold: int = 3) -> List[Dict]:
        """
        Stage 3: Strategy filter using stored OHLCV data (no API calls)
        Args:
            stage2_symbols_data: List of symbols that passed Stage 2
            confidence_threshold: Minimum confidence layers required (default: 3)
        Returns:
            List of symbols that passed Stage 3 filter
        """
        try:
            if not stage2_symbols_data:
                raise ValueError("No Stage 2 symbols provided to Stage 3 filter")
            
            # CRITICAL QUALITY CHECK: Validate Stage 2 data
            required_fields = ['symbol', 'ohlcv', 'technical_data', 'signal_direction']
            for i, symbol_data in enumerate(stage2_symbols_data):
                missing_fields = [field for field in required_fields if field not in symbol_data]
                if missing_fields:
                    raise ValueError(f"Stage 2 symbol {i} missing critical fields: {missing_fields}")
                
                # Validate technical data quality
                technical_data = symbol_data.get('technical_data', {})
                if not technical_data:
                    raise ValueError(f"Stage 2 symbol {i} ({symbol_data['symbol']}) missing technical_data")
                
                # Validate signal direction
                signal_direction = symbol_data.get('signal_direction')
                if signal_direction not in ['LONG', 'SHORT']:
                    raise ValueError(f"Stage 2 symbol {i} ({symbol_data['symbol']}) invalid signal_direction: {signal_direction}")
            
            stage3_passes = []
            
            for symbol_data in stage2_symbols_data:
                symbol = symbol_data['symbol']
                ohlcv = symbol_data.get('ohlcv')
                technical_data = symbol_data.get('technical_data', {})
                
                try:
                    # Run strategy functions using stored OHLCV data
                    trap_signal = self.trap_trade(ohlcv, technical_data)
                    smc_signal = self.smc_trade(ohlcv, technical_data)
                    scalp_signal = self.scalp_trade(ohlcv, technical_data)
                    vol_spike = self.check_volume_spike(ohlcv)
                    
                    # CRITICAL QUALITY CHECK: Validate strategy results
                    strategy_results = [trap_signal, smc_signal, scalp_signal, vol_spike]
                    if not all(isinstance(result, bool) for result in strategy_results):
                        self.logger.warning(f"CRITICAL: {symbol} invalid strategy results - skipping")
                        continue
                    
                    # Check if at least one strategy passes
                    strategy_passed = any([trap_signal, smc_signal, scalp_signal, vol_spike])
                    
                    if strategy_passed:
                        # Calculate confidence layers
                        stage3_data = {
                            'trap': trap_signal,
                            'smc': smc_signal,
                            'scalp': scalp_signal,
                            'vol_spike': vol_spike,
                            'signal_direction': symbol_data.get('signal_direction', 'LONG')
                        }
                        confidence_layers = self.calculate_confidence_layers(technical_data, stage3_data)
                        
                        # CRITICAL QUALITY CHECK: Validate confidence calculation
                        if not isinstance(confidence_layers, int) or confidence_layers < 0 or confidence_layers > 7:
                            self.logger.warning(f"CRITICAL: {symbol} invalid confidence_layers: {confidence_layers} - skipping")
                            continue
                        
                        if confidence_layers >= confidence_threshold:  # Use configurable threshold
                            # Build strategies list
                            strategies = []
                            if trap_signal: strategies.append('TRAP')
                            if smc_signal: strategies.append('SMC')
                            if scalp_signal: strategies.append('SCALP')
                            if vol_spike: strategies.append('VOL_SPIKE')
                            
                            # Add confidence_layers and stage3_data to symbol_data
                            symbol_data['confidence_layers'] = confidence_layers
                            symbol_data['stage3_data'] = {
                                'strategies_passed': strategies,
                                'trap': trap_signal,
                                'smc': smc_signal,
                                'scalp': scalp_signal,
                                'vol_spike': vol_spike
                            }
                            stage3_passes.append(symbol_data)
                            
                            self.logger.info(f"Stage 3 PASS: {symbol} - Confidence: {confidence_layers}/7")
                            self.logger.info(f"   Strategies: {', '.join(strategies)}")
                        else:
                            self.logger.debug(f"Stage 3 REJECT: {symbol} - Confidence: {confidence_layers}/7 < {confidence_threshold}/7")
                    else:
                        self.logger.debug(f"Stage 3 FAILED: {symbol} - No strategies passed")
                    
                except Exception as e:
                    self.logger.error(f"CRITICAL: Error processing {symbol} in Stage 3: {e}")
                    continue
            
            # CRITICAL QUALITY CHECK: Ensure minimum pass rate
            pass_rate = len(stage3_passes) / len(stage2_symbols_data) * 100
            if pass_rate < 0.1:  # Less than 0.1% pass rate indicates potential issues
                self.logger.warning(f"CRITICAL: Very low Stage 3 pass rate: {pass_rate:.1f}% - may indicate overly strict criteria")
            
            self.logger.info(f"Stage 3 Filter: {len(stage3_passes)}/{len(stage2_symbols_data)} symbols passed ({pass_rate:.1f}% pass rate)")
            return stage3_passes
            
        except Exception as e:
            self.logger.error(f"CRITICAL ERROR in stage 3 filter: {e}")
            raise  # Re-raise to fail fast 