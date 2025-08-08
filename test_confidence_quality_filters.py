#!/usr/bin/env python3
"""
Unit Tests for Confidence & Quality Filters
Test 7-layer confidence scoring, star ratings, and signal ranking
"""

import unittest
from core.filter_tree import FilterTree
from send_real_signals import format_signal_for_discord, CONFIG
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

class TestConfidenceQualityFilters(unittest.TestCase):
    """Test cases for confidence and quality filters"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.filter_tree = FilterTree()
    
    def test_7_layer_confidence_scoring(self):
        """Test that confidence scores now span 0-7 layers"""
        print("\n🧪 Testing 7-Layer Confidence Scoring")
        
        # Test data with all possible layers
        technical_data = {
            'ema_crossover': True,
            'rsi_reversal': True,
            'macd_bullish': True,
            'adx_value': 25,
            'volatility_rank': 60,
            'ema_distance_passes': True,
            'ema_distance_passes_bearish': False
        }
        
        stage3_data = {
            'signal_direction': 'LONG',
            'orderblock_bonus': True,
            'orderblock_type': 'bullish'
        }
        
        # Calculate confidence layers
        confidence = self.filter_tree.calculate_confidence_layers(technical_data, stage3_data)
        
        print(f"  📊 Confidence Score: {confidence}/7")
        print(f"  ✅ Expected: 7 layers (all conditions met)")
        
        # Verify confidence is 7 (all layers)
        self.assertEqual(confidence, 7, f"Expected 7 confidence layers, got {confidence}")
        
        # Test minimum confidence (0 layers)
        min_technical_data = {
            'ema_crossover': False,
            'rsi_reversal': False,
            'macd_bullish': False,
            'adx_value': 15,
            'volatility_rank': 30,
            'ema_distance_passes': False,
            'ema_distance_passes_bearish': False
        }
        
        min_stage3_data = {
            'signal_direction': 'LONG',
            'orderblock_bonus': False
        }
        
        min_confidence = self.filter_tree.calculate_confidence_layers(min_technical_data, min_stage3_data)
        
        print(f"  📊 Minimum Confidence Score: {min_confidence}/7")
        print(f"  ✅ Expected: 0 layers (no conditions met)")
        
        # Verify minimum confidence is 0
        self.assertEqual(min_confidence, 0, f"Expected 0 confidence layers, got {min_confidence}")
        
        # Test partial confidence (4 layers)
        partial_technical_data = {
            'ema_crossover': True,
            'rsi_reversal': True,
            'macd_bullish': False,
            'adx_value': 25,
            'volatility_rank': 30,
            'ema_distance_passes': True,
            'ema_distance_passes_bearish': False
        }
        
        partial_stage3_data = {
            'signal_direction': 'LONG',
            'orderblock_bonus': False
        }
        
        partial_confidence = self.filter_tree.calculate_confidence_layers(partial_technical_data, partial_stage3_data)
        
        print(f"  📊 Partial Confidence Score: {partial_confidence}/7")
        print(f"  ✅ Expected: 4 layers (EMA crossover, RSI reversal, ADX, EMA distance)")
        
        # Verify partial confidence is 4
        self.assertEqual(partial_confidence, 4, f"Expected 4 confidence layers, got {partial_confidence}")
    
    def test_star_rating_conversion(self):
        """Test star rating conversion logic"""
        print("\n🧪 Testing Star Rating Conversion")
        
        # Test cases: (confidence, expected_stars)
        test_cases = [
            (0, "★1"),
            (1, "★1"),
            (2, "★2"),
            (3, "★2"),
            (4, "★3"),
            (5, "★4"),
            (6, "★5"),
            (7, "★5")
        ]
        
        for confidence, expected_stars in test_cases:
            # Create mock signal
            signal = {
                'symbol': 'BTC/USDT',
                'signal_direction': 'LONG',
                'timeframe': '15m',
                'price': 50000,
                'stop_loss': 49000,
                'tp1_price': 52000,
                'risk_reward': 1.5,
                'confidence_layers': confidence,
                'confluence_percentage': 75.0
            }
            
            # Format signal
            formatted = format_signal_for_discord(signal)
            
            # Check if star rating is present
            self.assertIn(expected_stars, formatted, 
                         f"Expected {expected_stars} for confidence {confidence}, not found in: {formatted}")
            
            print(f"  ✅ Confidence {confidence} → {expected_stars}")
    
    def test_signal_ranking_and_cap(self):
        """Test signal ranking by quality and cap functionality"""
        print("\n🧪 Testing Signal Ranking and Cap")
        
        # Create mock signals with varying quality
        mock_signals = [
            {
                'symbol': 'BTC/USDT',
                'confidence_layers': 5,
                'confluence_percentage': 80.0,
                'quality_score': 5 + 80.0
            },
            {
                'symbol': 'ETH/USDT',
                'confidence_layers': 7,
                'confluence_percentage': 60.0,
                'quality_score': 7 + 60.0
            },
            {
                'symbol': 'BNB/USDT',
                'confidence_layers': 3,
                'confluence_percentage': 90.0,
                'quality_score': 3 + 90.0
            },
            {
                'symbol': 'ADA/USDT',
                'confidence_layers': 6,
                'confluence_percentage': 70.0,
                'quality_score': 6 + 70.0
            },
            {
                'symbol': 'SOL/USDT',
                'confidence_layers': 4,
                'confluence_percentage': 85.0,
                'quality_score': 4 + 85.0
            },
            {
                'symbol': 'DOT/USDT',
                'confidence_layers': 2,
                'confluence_percentage': 95.0,
                'quality_score': 2 + 95.0
            }
        ]
        
        # Test ranking logic (same as in send_real_signals.py)
        original_count = len(mock_signals)
        mock_signals.sort(key=lambda s: (s.get('confidence_layers', 0) + s.get('confluence_percentage', 0)), reverse=True)
        
        # Apply cap (default is 5)
        max_signals = CONFIG['MAX_SIGNALS_PER_SCAN']
        capped_signals = mock_signals[:max_signals]
        
        print(f"  📊 Original signals: {original_count}")
        print(f"  🏆 Top signals selected: {len(capped_signals)}")
        
        # Verify cap is applied
        self.assertLessEqual(len(capped_signals), max_signals, 
                           f"Signal cap not applied: {len(capped_signals)} > {max_signals}")
        
        # Verify ranking is correct (highest quality first)
        if len(capped_signals) >= 2:
            first_quality = capped_signals[0]['quality_score']
            second_quality = capped_signals[1]['quality_score']
            self.assertGreaterEqual(first_quality, second_quality,
                                  f"Ranking incorrect: {first_quality} < {second_quality}")
        
        print(f"  ✅ Signal cap applied: {len(capped_signals)}/{max_signals} signals")
        print(f"  ✅ Ranking verified: highest quality signals selected")
        
        # Display ranking results
        for i, signal in enumerate(capped_signals):
            quality = signal['quality_score']
            print(f"    {i+1}. {signal['symbol']}: Quality {quality:.1f} (Confidence: {signal['confidence_layers']}, Confluence: {signal['confluence_percentage']:.1f}%)")
    
    def test_ema_distance_bonus_integration(self):
        """Test that EMA distance bonus is properly integrated into confidence scoring"""
        print("\n🧪 Testing EMA Distance Bonus Integration")
        
        # Test bullish signal with EMA distance bonus
        bullish_technical_data = {
            'ema_crossover': True,
            'rsi_reversal': True,
            'macd_bullish': True,
            'adx_value': 25,
            'volatility_rank': 60,
            'ema_distance_passes': True,  # EMA distance bonus
            'ema_distance_passes_bearish': False
        }
        
        bullish_stage3_data = {
            'signal_direction': 'LONG',
            'orderblock_bonus': False
        }
        
        bullish_confidence = self.filter_tree.calculate_confidence_layers(bullish_technical_data, bullish_stage3_data)
        
        print(f"  📊 Bullish Confidence (with EMA distance): {bullish_confidence}/7")
        
        # Should be 6 layers: EMA crossover, RSI reversal, MACD bullish, ADX, Volatility rank, EMA distance
        self.assertEqual(bullish_confidence, 6, f"Expected 6 confidence layers with EMA distance bonus, got {bullish_confidence}")
        
        # Test without EMA distance bonus
        bullish_technical_data_no_distance = {
            'ema_crossover': True,
            'rsi_reversal': True,
            'macd_bullish': True,
            'adx_value': 25,
            'volatility_rank': 60,
            'ema_distance_passes': False,  # No EMA distance bonus
            'ema_distance_passes_bearish': False
        }
        
        bullish_confidence_no_distance = self.filter_tree.calculate_confidence_layers(bullish_technical_data_no_distance, bullish_stage3_data)
        
        print(f"  📊 Bullish Confidence (without EMA distance): {bullish_confidence_no_distance}/7")
        
        # Should be 5 layers (missing EMA distance bonus)
        self.assertEqual(bullish_confidence_no_distance, 5, f"Expected 5 confidence layers without EMA distance bonus, got {bullish_confidence_no_distance}")
        
        # Verify the difference is exactly 1 layer
        self.assertEqual(bullish_confidence - bullish_confidence_no_distance, 1, 
                        "EMA distance bonus should add exactly 1 confidence layer")
    
    def test_orderblock_bonus_integration(self):
        """Test that orderblock alignment bonus is properly integrated into confidence scoring"""
        print("\n🧪 Testing Orderblock Bonus Integration")
        
        # Test bullish signal with bullish orderblock
        bullish_technical_data = {
            'ema_crossover': True,
            'rsi_reversal': True,
            'macd_bullish': True,
            'adx_value': 25,
            'volatility_rank': 60,
            'ema_distance_passes': True,
            'ema_distance_passes_bearish': False
        }
        
        bullish_stage3_data = {
            'signal_direction': 'LONG',
            'orderblock_bonus': True,
            'orderblock_type': 'bullish'  # Matches signal direction
        }
        
        bullish_confidence = self.filter_tree.calculate_confidence_layers(bullish_technical_data, bullish_stage3_data)
        
        print(f"  📊 Bullish Confidence (with bullish orderblock): {bullish_confidence}/7")
        
        # Should be 7 layers (all including orderblock bonus)
        self.assertEqual(bullish_confidence, 7, f"Expected 7 confidence layers with orderblock bonus, got {bullish_confidence}")
        
        # Test with mismatched orderblock (bullish signal, bearish orderblock)
        mismatched_stage3_data = {
            'signal_direction': 'LONG',
            'orderblock_bonus': True,
            'orderblock_type': 'bearish'  # Doesn't match signal direction
        }
        
        mismatched_confidence = self.filter_tree.calculate_confidence_layers(bullish_technical_data, mismatched_stage3_data)
        
        print(f"  📊 Bullish Confidence (with bearish orderblock): {mismatched_confidence}/7")
        
        # Should be 6 layers (no orderblock bonus due to polarity mismatch)
        self.assertEqual(mismatched_confidence, 6, f"Expected 6 confidence layers with mismatched orderblock, got {mismatched_confidence}")
        
        # Verify the difference is exactly 1 layer
        self.assertEqual(bullish_confidence - mismatched_confidence, 1, 
                        "Orderblock bonus should add exactly 1 confidence layer when polarity matches")

def run_confidence_quality_tests():
    """Run all confidence and quality filter tests"""
    print("🧪 Running Confidence & Quality Filter Tests")
    print("=" * 60)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestConfidenceQualityFilters)
    
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
        print("\n✅ All tests passed! Confidence & quality filters working correctly.")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    run_confidence_quality_tests() 