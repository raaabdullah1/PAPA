#!/usr/bin/env python3
"""
Test Orderblock Detection
Verify orderblock detection functionality with sample data
"""

from core.orderblock_detector import OrderblockDetector
from core.scanner import MarketScanner
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

def test_orderblock_detection():
    """Test orderblock detection with real market data"""
    print("🧪 Testing Orderblock Detection")
    print("=" * 50)
    
    # Initialize components
    scanner = MarketScanner()
    orderblock_detector = OrderblockDetector()
    
    # Test symbols
    test_symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
    timeframes = ['15m', '30m', '1h', '4h']
    
    total_orderblocks = 0
    
    for symbol in test_symbols:
        print(f"\n📊 Testing {symbol}:")
        
        for timeframe in timeframes:
            try:
                # Fetch data
                ohlcv_data = scanner.fetch_ohlcv_paginated(symbol, timeframe=timeframe, max_bars=1000)
                
                if not ohlcv_data or len(ohlcv_data) < 50:
                    print(f"  ⚠️ {timeframe}: Insufficient data")
                    continue
                
                # Detect orderblocks
                orderblocks = orderblock_detector.detect_orderblocks(ohlcv_data, timeframe)
                
                if orderblocks:
                    bullish_count = len([ob for ob in orderblocks if ob.type == 'bullish'])
                    bearish_count = len([ob for ob in orderblocks if ob.type == 'bearish'])
                    
                    print(f"  ✅ {timeframe}: {len(orderblocks)} orderblocks "
                          f"({bullish_count} bullish, {bearish_count} bearish)")
                    
                    # Show strongest orderblock
                    strongest = max(orderblocks, key=lambda x: x.strength)
                    print(f"    🏆 Strongest: {strongest.type} (strength: {strongest.strength:.2f}, "
                          f"volume ratio: {strongest.volume_ratio:.2f})")
                    
                    total_orderblocks += len(orderblocks)
                else:
                    print(f"  ❌ {timeframe}: No orderblocks detected")
                    
            except Exception as e:
                print(f"  ❌ {timeframe}: Error - {e}")
    
    print(f"\n📈 Summary: Detected {total_orderblocks} total orderblocks")
    
    # Test orderblock alignment
    print(f"\n🔍 Testing Orderblock Alignment:")
    
    # Simulate a signal
    test_signal = {
        'symbol': 'BTC/USDT',
        'direction': 'LONG',
        'price': 50000,
        'confidence_layers': 3
    }
    
    # Get orderblocks for BTC/USDT 15m
    try:
        btc_data = scanner.fetch_ohlcv_paginated('BTC/USDT', timeframe='15m', max_bars=1000)
        if btc_data:
            orderblocks = orderblock_detector.detect_orderblocks(btc_data, '15m')
            
            is_aligned, best_block = orderblock_detector.check_orderblock_alignment(
                test_signal, orderblocks, 50000, '15m'
            )
            
            if is_aligned:
                print(f"✅ Signal aligned with {best_block.type} orderblock "
                      f"(strength: {best_block.strength:.2f})")
            else:
                print("❌ No orderblock alignment found")
                
    except Exception as e:
        print(f"❌ Alignment test error: {e}")
    
    print("\n✅ Orderblock detection test completed!")

if __name__ == "__main__":
    test_orderblock_detection() 