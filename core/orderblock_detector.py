#!/usr/bin/env python3
"""
Orderblock Detection Module
Identifies bullish and bearish orderblocks across different timeframes
"""

import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

@dataclass
class Orderblock:
    """Represents an identified orderblock"""
    start_candle: int  # Index of start candle
    end_candle: int    # Index of end candle
    start_time: int    # Timestamp of start
    end_time: int      # Timestamp of end
    type: str          # 'bullish' or 'bearish'
    strength: float    # Orderblock strength (0-1)
    volume_ratio: float  # Volume ratio compared to average
    price_range: float   # Price range of the orderblock

class OrderblockDetector:
    """Detects bullish and bearish orderblocks in OHLCV data"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Configuration parameters
        self.min_orderblock_size = 3      # Minimum candles for orderblock
        self.max_orderblock_size = 20     # Maximum candles for orderblock
        self.volume_threshold = 1.5       # Volume must be 1.5x average
        self.price_range_threshold = 0.02 # Minimum 2% price range
        self.lookback_period = 50         # Look back 50 candles for orderblocks
        
    def detect_orderblocks(self, ohlcv_data: List, timeframe: str) -> List[Orderblock]:
        """
        Detect orderblocks in OHLCV data
        
        Args:
            ohlcv_data: List of [timestamp, open, high, low, close, volume]
            timeframe: Timeframe string ('15m', '30m', '1h', '4h')
            
        Returns:
            List of detected Orderblock objects
        """
        if len(ohlcv_data) < self.lookback_period:
            self.logger.warning(f"Insufficient data for orderblock detection: {len(ohlcv_data)} candles")
            return []
        
        orderblocks = []
        
        # Calculate average volume for comparison
        volumes = [candle[5] for candle in ohlcv_data[-self.lookback_period:]]
        avg_volume = sum(volumes) / len(volumes) if volumes else 0
        
        if avg_volume == 0:
            return []
        
        # Look for orderblocks in recent data
        recent_data = ohlcv_data[-self.lookback_period:]
        
        # Detect bullish orderblocks (strong buying pressure)
        bullish_blocks = self._detect_bullish_orderblocks(recent_data, avg_volume)
        orderblocks.extend(bullish_blocks)
        
        # Detect bearish orderblocks (strong selling pressure)
        bearish_blocks = self._detect_bearish_orderblocks(recent_data, avg_volume)
        orderblocks.extend(bearish_blocks)
        
        # Sort by strength (strongest first)
        orderblocks.sort(key=lambda x: x.strength, reverse=True)
        
        self.logger.info(f"Detected {len(orderblocks)} orderblocks on {timeframe}: "
                        f"{len([ob for ob in orderblocks if ob.type == 'bullish'])} bullish, "
                        f"{len([ob for ob in orderblocks if ob.type == 'bearish'])} bearish")
        
        return orderblocks
    
    def _detect_bullish_orderblocks(self, data: List, avg_volume: float) -> List[Orderblock]:
        """Detect bullish orderblocks (strong buying pressure)"""
        bullish_blocks = []
        
        for i in range(len(data) - self.min_orderblock_size):
            # Check for potential bullish orderblock
            if self._is_bullish_orderblock(data, i, avg_volume):
                block = self._create_bullish_orderblock(data, i)
                if block:
                    bullish_blocks.append(block)
        
        return bullish_blocks
    
    def _detect_bearish_orderblocks(self, data: List, avg_volume: float) -> List[Orderblock]:
        """Detect bearish orderblocks (strong selling pressure)"""
        bearish_blocks = []
        
        for i in range(len(data) - self.min_orderblock_size):
            # Check for potential bearish orderblock
            if self._is_bearish_orderblock(data, i, avg_volume):
                block = self._create_bearish_orderblock(data, i)
                if block:
                    bearish_blocks.append(block)
        
        return bearish_blocks
    
    def _is_bullish_orderblock(self, data: List, start_idx: int, avg_volume: float) -> bool:
        """Check if a sequence of candles forms a bullish orderblock"""
        # Look for strong buying pressure pattern
        for size in range(self.min_orderblock_size, min(self.max_orderblock_size, len(data) - start_idx)):
            end_idx = start_idx + size
            
            # Get the block of candles
            block = data[start_idx:end_idx]
            
            # Check volume condition
            total_volume = sum(candle[5] for candle in block)
            avg_block_volume = total_volume / len(block)
            
            if avg_block_volume < avg_volume * self.volume_threshold:
                continue
            
            # Check price action (strong upward movement)
            start_price = block[0][1]  # Open of first candle
            end_price = block[-1][4]   # Close of last candle
            high_price = max(candle[2] for candle in block)
            low_price = min(candle[3] for candle in block)
            
            price_range = (high_price - low_price) / low_price
            
            if price_range < self.price_range_threshold:
                continue
            
            # Check for bullish pattern (higher highs, higher lows)
            closes = [candle[4] for candle in block]
            if len(closes) >= 3:
                # Check if we have an upward trend
                if closes[-1] > closes[0] and high_price > start_price:
                    return True
        
        return False
    
    def _is_bearish_orderblock(self, data: List, start_idx: int, avg_volume: float) -> bool:
        """Check if a sequence of candles forms a bearish orderblock"""
        # Look for strong selling pressure pattern
        for size in range(self.min_orderblock_size, min(self.max_orderblock_size, len(data) - start_idx)):
            end_idx = start_idx + size
            
            # Get the block of candles
            block = data[start_idx:end_idx]
            
            # Check volume condition
            total_volume = sum(candle[5] for candle in block)
            avg_block_volume = total_volume / len(block)
            
            if avg_block_volume < avg_volume * self.volume_threshold:
                continue
            
            # Check price action (strong downward movement)
            start_price = block[0][1]  # Open of first candle
            end_price = block[-1][4]   # Close of last candle
            high_price = max(candle[2] for candle in block)
            low_price = min(candle[3] for candle in block)
            
            price_range = (high_price - low_price) / low_price
            
            if price_range < self.price_range_threshold:
                continue
            
            # Check for bearish pattern (lower highs, lower lows)
            closes = [candle[4] for candle in block]
            if len(closes) >= 3:
                # Check if we have a downward trend
                if closes[-1] < closes[0] and low_price < start_price:
                    return True
        
        return False
    
    def _create_bullish_orderblock(self, data: List, start_idx: int) -> Optional[Orderblock]:
        """Create a bullish orderblock object"""
        try:
            # Find the optimal size for this orderblock
            best_size = self.min_orderblock_size
            best_strength = 0
            
            for size in range(self.min_orderblock_size, min(self.max_orderblock_size, len(data) - start_idx)):
                end_idx = start_idx + size
                block = data[start_idx:end_idx]
                
                # Calculate strength based on volume and price action
                total_volume = sum(candle[5] for candle in block)
                avg_volume = sum(candle[5] for candle in data[-50:]) / 50
                volume_ratio = total_volume / (avg_volume * len(block))
                
                start_price = block[0][1]
                end_price = block[-1][4]
                high_price = max(candle[2] for candle in block)
                low_price = min(candle[3] for candle in block)
                
                price_range = (high_price - low_price) / low_price
                price_movement = (end_price - start_price) / start_price
                
                # Strength calculation
                strength = (volume_ratio * 0.4 + price_range * 10 + max(0, price_movement) * 5) / 3
                
                if strength > best_strength:
                    best_strength = strength
                    best_size = size
            
            if best_strength < 0.3:  # Minimum strength threshold
                return None
            
            end_idx = start_idx + best_size
            block = data[start_idx:end_idx]
            
            total_volume = sum(candle[5] for candle in block)
            avg_volume = sum(candle[5] for candle in data[-50:]) / 50
            volume_ratio = total_volume / (avg_volume * len(block))
            
            high_price = max(candle[2] for candle in block)
            low_price = min(candle[3] for candle in block)
            price_range = (high_price - low_price) / low_price
            
            return Orderblock(
                start_candle=start_idx,
                end_candle=end_idx - 1,
                start_time=block[0][0],
                end_time=block[-1][0],
                type='bullish',
                strength=best_strength,
                volume_ratio=volume_ratio,
                price_range=price_range
            )
            
        except Exception as e:
            self.logger.error(f"Error creating bullish orderblock: {e}")
            return None
    
    def _create_bearish_orderblock(self, data: List, start_idx: int) -> Optional[Orderblock]:
        """Create a bearish orderblock object"""
        try:
            # Find the optimal size for this orderblock
            best_size = self.min_orderblock_size
            best_strength = 0
            
            for size in range(self.min_orderblock_size, min(self.max_orderblock_size, len(data) - start_idx)):
                end_idx = start_idx + size
                block = data[start_idx:end_idx]
                
                # Calculate strength based on volume and price action
                total_volume = sum(candle[5] for candle in block)
                avg_volume = sum(candle[5] for candle in data[-50:]) / 50
                volume_ratio = total_volume / (avg_volume * len(block))
                
                start_price = block[0][1]
                end_price = block[-1][4]
                high_price = max(candle[2] for candle in block)
                low_price = min(candle[3] for candle in block)
                
                price_range = (high_price - low_price) / low_price
                price_movement = (start_price - end_price) / start_price
                
                # Strength calculation
                strength = (volume_ratio * 0.4 + price_range * 10 + max(0, price_movement) * 5) / 3
                
                if strength > best_strength:
                    best_strength = strength
                    best_size = size
            
            if best_strength < 0.3:  # Minimum strength threshold
                return None
            
            end_idx = start_idx + best_size
            block = data[start_idx:end_idx]
            
            total_volume = sum(candle[5] for candle in block)
            avg_volume = sum(candle[5] for candle in data[-50:]) / 50
            volume_ratio = total_volume / (avg_volume * len(block))
            
            high_price = max(candle[2] for candle in block)
            low_price = min(candle[3] for candle in block)
            price_range = (high_price - low_price) / low_price
            
            return Orderblock(
                start_candle=start_idx,
                end_candle=end_idx - 1,
                start_time=block[0][0],
                end_time=block[-1][0],
                type='bearish',
                strength=best_strength,
                volume_ratio=volume_ratio,
                price_range=price_range
            )
            
        except Exception as e:
            self.logger.error(f"Error creating bearish orderblock: {e}")
            return None
    
    def check_orderblock_alignment(self, signal: Dict, orderblocks: List[Orderblock], 
                                 current_price: float, timeframe: str) -> Tuple[bool, Optional[Orderblock]]:
        """
        Check if a signal aligns with an orderblock of the correct polarity
        
        Args:
            signal: Signal dictionary with direction and price info
            orderblocks: List of detected orderblocks
            current_price: Current market price
            timeframe: Timeframe of the signal
            
        Returns:
            Tuple of (is_aligned, best_orderblock)
        """
        if not orderblocks:
            return False, None
        
        signal_direction = signal.get('direction', '').upper()
        signal_price = signal.get('price', current_price)
        
        # Find orderblocks of the correct polarity
        matching_blocks = []
        
        for block in orderblocks:
            # Check if orderblock is recent (within last 20% of data)
            if block.type == 'bullish' and signal_direction == 'LONG':
                matching_blocks.append(block)
            elif block.type == 'bearish' and signal_direction == 'SHORT':
                matching_blocks.append(block)
        
        if not matching_blocks:
            return False, None
        
        # Find the strongest orderblock that's close to the signal price
        best_block = None
        best_score = 0
        
        for block in matching_blocks:
            # Calculate proximity score (closer to signal price = higher score)
            block_mid_price = (block.start_time + block.end_time) / 2  # Simplified
            price_proximity = 1 / (1 + abs(signal_price - block_mid_price) / signal_price)
            
            # Combined score: strength + proximity
            score = block.strength * 0.7 + price_proximity * 0.3
            
            if score > best_score:
                best_score = score
                best_block = block
        
        # Require minimum alignment score
        if best_score >= 0.4:
            self.logger.info(f"Orderblock alignment found for {signal.get('symbol', 'Unknown')} "
                           f"on {timeframe}: {best_block.type} orderblock (strength: {best_block.strength:.2f})")
            return True, best_block
        
        return False, None 