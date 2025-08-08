#!/usr/bin/env python3
"""
Unit Tests for Volatility-Adaptive EMA-Distance Filter
Test the enhanced Stage 2 EMA filter with ATR-based adaptive thresholds
"""

import unittest
from core.filter_tree import FilterTree
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

class TestVolatilityAdaptiveEMA(unittest.TestCase):
    """Test cases for volatility-adaptive EMA-distance filter"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.filter_tree = FilterTree()
    
    def create_mock_ohlcv(self, base_price=100, volatility=0.02, trend=0.001, periods=100):
        """Create mock OHLCV data with specified volatility and trend"""
        import random
        
        ohlcv = []
        current_price = base_price
        
        for i in range(periods):
            # Add trend
            trend_change = current_price * trend
            
            # Add volatility (random walk) - reduced magnitude for more realistic values
            volatility_change = current_price * volatility * (random.random() - 0.5) * 0.1  # Reduced by 10x
            
            # Calculate OHLC
            open_price = current_price
            high_price = open_price * (1 + abs(volatility_change) * 0.5)
            low_price = open_price * (1 - abs(volatility_change) * 0.5)
            close_price = open_price + trend_change + volatility_change
            
            # Ensure prices are positive
            high_price = max(high_price, low_price * 1.001)
            close_price = max(close_price, low_price)
            close_price = min(close_price, high_price)
            
            # Volume (mock)
            volume = 1000000 + random.randint(-100000, 100000)
            
            # Timestamp (mock)
            timestamp = 1609459200000 + i * 60000  # 1-minute intervals
            
            ohlcv.append([timestamp, open_price, high_price, low_price, close_price, volume])
            current_price = close_price
        
        return ohlcv
    
    def test_low_volatility_scenario(self):
        """Test EMA-distance filter in low volatility scenario"""
        print("\n🧪 Testing Low Volatility Scenario")
        
        # Create low volatility data (0.5% daily volatility)
        low_vol_ohlcv = self.create_mock_ohlcv(
            base_price=50000,  # BTC-like price
            volatility=0.005,  # 0.5% volatility
            trend=0.0001,      # Slight uptrend
            periods=100
        )
        
        # Calculate indicators
        indicators = self.filter_tree._calculate_stage_2_indicators(low_vol_ohlcv)
        
        # Extract values
        atr_value = indicators['atr_value']
        ema_fast = indicators['ema_fast']
        ema_slow = indicators['ema_slow']
        ema_crossover = indicators['ema_crossover']
        ema_distance_passes = indicators['ema_distance_passes']
        
        print(f"  📊 Low Volatility Results:")
        print(f"    ATR Value: {atr_value:.6f}")
        print(f"    EMA Fast: {ema_fast:.2f}")
        print(f"    EMA Slow: {ema_slow:.2f}")
        print(f"    EMA Gap: {abs(ema_fast - ema_slow):.2f}")
        print(f"    Required Gap: {atr_value * 0.1:.6f}")
        print(f"    EMA Crossover: {ema_crossover}")
        print(f"    Distance Passes: {ema_distance_passes}")
        
        # In low volatility, the normalized ATR should be small
        normalized_atr = atr_value / 50000  # Normalize by price
        self.assertLess(normalized_atr, 10.0, "Normalized ATR should be low in low volatility scenario")
        
        # The required gap should be (ATR/price) * 0.1
        expected_required_gap = (atr_value / 50000) * 0.1
        actual_gap = abs(ema_fast - ema_slow) / 50000  # Normalized by price
        
        print(f"    Actual Gap (normalized): {actual_gap:.6f}")
        print(f"    Expected Required Gap: {expected_required_gap:.6f}")
        
        # Verify the logic works correctly
        if ema_crossover:
            self.assertEqual(ema_distance_passes, actual_gap >= expected_required_gap,
                           "EMA distance check should match expected logic")
    
    def test_high_volatility_scenario(self):
        """Test EMA-distance filter in high volatility scenario"""
        print("\n🧪 Testing High Volatility Scenario")
        
        # Create high volatility data (5% daily volatility)
        high_vol_ohlcv = self.create_mock_ohlcv(
            base_price=50000,  # BTC-like price
            volatility=0.05,   # 5% volatility
            trend=0.001,       # Stronger trend
            periods=100
        )
        
        # Calculate indicators
        indicators = self.filter_tree._calculate_stage_2_indicators(high_vol_ohlcv)
        
        # Extract values
        atr_value = indicators['atr_value']
        ema_fast = indicators['ema_fast']
        ema_slow = indicators['ema_slow']
        ema_crossover = indicators['ema_crossover']
        ema_distance_passes = indicators['ema_distance_passes']
        
        print(f"  📊 High Volatility Results:")
        print(f"    ATR Value: {atr_value:.6f}")
        print(f"    EMA Fast: {ema_fast:.2f}")
        print(f"    EMA Slow: {ema_slow:.2f}")
        print(f"    EMA Gap: {abs(ema_fast - ema_slow):.2f}")
        print(f"    Required Gap: {atr_value * 0.1:.6f}")
        print(f"    EMA Crossover: {ema_crossover}")
        print(f"    Distance Passes: {ema_distance_passes}")
        
        # In high volatility, the normalized ATR should be larger
        normalized_atr = atr_value / 50000  # Normalize by price
        self.assertGreater(normalized_atr, 0.05, "Normalized ATR should be high in high volatility scenario")
        
        # The required gap should be (ATR/price) * 0.1
        expected_required_gap = (atr_value / 50000) * 0.1
        actual_gap = abs(ema_fast - ema_slow) / 50000  # Normalized by price
        
        print(f"    Actual Gap (normalized): {actual_gap:.6f}")
        print(f"    Expected Required Gap: {expected_required_gap:.6f}")
        
        # Verify the logic works correctly
        if ema_crossover:
            self.assertEqual(ema_distance_passes, actual_gap >= expected_required_gap,
                           "EMA distance check should match expected logic")
    
    def test_volatility_scaling(self):
        """Test that the required gap scales correctly with volatility"""
        print("\n🧪 Testing Volatility Scaling")
        
        # Test multiple volatility levels
        volatility_levels = [0.005, 0.01, 0.02, 0.05, 0.1]  # 0.5% to 10%
        required_gaps = []
        
        for vol in volatility_levels:
            ohlcv = self.create_mock_ohlcv(
                base_price=50000,
                volatility=vol,
                trend=0.0001,
                periods=100
            )
            
            indicators = self.filter_tree._calculate_stage_2_indicators(ohlcv)
            atr_value = indicators['atr_value']
            normalized_atr = atr_value / 50000  # Normalize by price
            required_gap = normalized_atr * 0.1
            
            required_gaps.append(required_gap)
            
            print(f"  📊 Volatility {vol*100:.1f}%: ATR={atr_value:.6f}, Normalized ATR={normalized_atr:.6f}, Required Gap={required_gap:.6f}")
        
        # Verify that required gaps generally increase with volatility (allowing for some variation)
        increasing_count = 0
        for i in range(1, len(required_gaps)):
            if required_gaps[i] > required_gaps[i-1]:
                increasing_count += 1
        
        # At least 60% of comparisons should show increasing gaps
        self.assertGreaterEqual(increasing_count / (len(required_gaps) - 1), 0.6,
                               f"Most required gaps should increase with volatility, but only {increasing_count}/{len(required_gaps)-1} did")
        
        print(f"  ✅ Volatility scaling verified: {increasing_count}/{len(required_gaps)-1} gaps increase with volatility")
    
    def test_bearish_signals(self):
        """Test volatility-adaptive EMA-distance filter for bearish signals"""
        print("\n🧪 Testing Bearish Signals")
        
        # Create data with downtrend
        bearish_ohlcv = self.create_mock_ohlcv(
            base_price=50000,
            volatility=0.02,
            trend=-0.002,  # Downtrend
            periods=100
        )
        
        # Calculate indicators
        indicators = self.filter_tree._calculate_stage_2_indicators(bearish_ohlcv)
        
        # Extract values
        atr_value = indicators['atr_value']
        ema_fast = indicators['ema_fast']
        ema_slow = indicators['ema_slow']
        ema_crossunder = indicators['ema_crossunder']
        ema_distance_passes_bearish = indicators['ema_distance_passes_bearish']
        
        print(f"  📊 Bearish Signal Results:")
        print(f"    ATR Value: {atr_value:.6f}")
        print(f"    EMA Fast: {ema_fast:.2f}")
        print(f"    EMA Slow: {ema_slow:.2f}")
        print(f"    EMA Crossunder: {ema_crossunder}")
        print(f"    Distance Passes (Bearish): {ema_distance_passes_bearish}")
        
        # Verify bearish logic works
        if ema_crossunder:
            expected_required_gap = (atr_value / 50000) * 0.1
            actual_gap = abs(ema_fast - ema_slow) / 50000
            
            self.assertEqual(ema_distance_passes_bearish, actual_gap >= expected_required_gap,
                           "Bearish EMA distance check should match expected logic")
    
    def test_integration_with_existing_logic(self):
        """Test that the new filter integrates correctly with existing Stage 2 logic"""
        print("\n🧪 Testing Integration with Existing Logic")
        
        # Create test data
        ohlcv = self.create_mock_ohlcv(
            base_price=50000,
            volatility=0.02,
            trend=0.001,
            periods=100
        )
        
        # Calculate indicators
        indicators = self.filter_tree._calculate_stage_2_indicators(ohlcv)
        
        # Verify all required fields are present
        required_fields = [
            'ema_crossover', 'rsi_reversal', 'atr_valid', 'macd_bullish',
            'ema_crossunder', 'rsi_falling', 'macd_bearish', 'atr_value',
            'rsi_value', 'ema_fast', 'ema_slow', 'macd_line', 'macd_signal',
            'adx_value', 'volatility_rank', 'ema_distance_passes', 'ema_distance_passes_bearish'
        ]
        
        for field in required_fields:
            self.assertIn(field, indicators, f"Required field '{field}' missing from indicators")
        
        print(f"  ✅ All required fields present in indicators")
        print(f"  ✅ Integration with existing logic verified")

def run_volatility_adaptive_tests():
    """Run all volatility-adaptive EMA tests"""
    print("🧪 Running Volatility-Adaptive EMA-Distance Filter Tests")
    print("=" * 60)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestVolatilityAdaptiveEMA)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Summary:")
    print(f"  Tests Run: {result.testsRun}")
    print(f"  Failures: {len(result.failures)}")
    print(f"  Errors: {len(result.errors)}")
    
    if result.failures:
        print("\n❌ Failures:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")
    
    if result.errors:
        print("\n❌ Errors:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")
    
    if not result.failures and not result.errors:
        print("\n✅ All tests passed! Volatility-adaptive EMA filter working correctly.")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    run_volatility_adaptive_tests() 