#!/usr/bin/env python3
"""
Test Production Pipeline
Run production pipeline on a smaller batch to verify everything works
"""

from core.scanner import MarketScanner
from core.filter_tree import FilterTree
from core.notifier import DiscordNotifier
import time
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# Test configuration (smaller batch)
TEST_CONFIG = {
    'stage1_timeframe': '15m',
    'TOP_N': 0,
    'TF_WEIGHTS': {
        '4h': 3,
        '1h': 2, 
        '30m': 1,
        '15m': 1
    },
    'CONFLUENCE_BONUSES': {
        'high': 2,
        'medium': 1,
        'low': 0
    },
    'FAST_FAIL_4H': True,
    'MAX_WORKERS': 4,
    'PRODUCTION_MODE': False,  # Test mode
    'TEST_SYMBOLS_LIMIT': 100,  # Test with 100 symbols
    'STAGE3_THRESHOLD': 2,
    'MIN_RISK_REWARD': 1.15,
    'MIN_CONFLUENCE': 50
}

def test_production_pipeline():
    """Test production pipeline on smaller batch"""
    print("🧪 Testing Production Pipeline (100 symbols)")
    print("=" * 60)
    
    # Initialize components
    scanner = MarketScanner()
    filter_tree = FilterTree()
    discord = DiscordNotifier()
    
    # Display configuration
    print(f"📋 Test Configuration:")
    print(f"  Stage 1 Timeframe: {TEST_CONFIG['stage1_timeframe']}")
    print(f"  Test Symbols: {TEST_CONFIG['TEST_SYMBOLS_LIMIT']}")
    print(f"  Stage 3 Threshold: {TEST_CONFIG['STAGE3_THRESHOLD']}/7")
    print(f"  Min Risk-Reward: {TEST_CONFIG['MIN_RISK_REWARD']}")
    print(f"  Min Confluence: {TEST_CONFIG['MIN_CONFLUENCE']}%")
    
    try:
        # Get symbols
        all_symbols = scanner.get_futures_symbols()
        print(f"📊 Total symbols available: {len(all_symbols)}")
        
        # Use test batch
        symbols_to_process = all_symbols[:TEST_CONFIG['TEST_SYMBOLS_LIMIT']]
        print(f"🧪 TEST MODE: Processing first {len(symbols_to_process)} symbols")
        
        # Stage 1: Single timeframe filtering
        print(f"\n📊 Step 1: Stage 1 filtering on {TEST_CONFIG['stage1_timeframe']} timeframe...")
        
        # Import the function from send_real_signals.py
        from send_real_signals import run_stage1_single_timeframe, run_multi_timeframe_confluence, send_signals_to_discord, print_comprehensive_summary
        
        stage1_winners = run_stage1_single_timeframe(scanner, filter_tree, symbols_to_process, TEST_CONFIG['stage1_timeframe'])
        
        if not stage1_winners:
            print("❌ No symbols passed Stage 1 - aborting")
            return False
        
        print(f"✅ Stage 1 completed: {len(stage1_winners)} symbols passed")
        
        # Stage 2 & 3: Multi-timeframe confluence
        print(f"\n📊 Step 2: Multi-timeframe Stage 2 & 3 for {len(stage1_winners)} symbols...")
        
        final_signals = run_multi_timeframe_confluence(scanner, filter_tree, stage1_winners)
        
        if not final_signals:
            print("❌ No signals generated - aborting")
            return False
        
        # Send signals to Discord
        print(f"\n📤 Sending {len(final_signals)} signals to Discord...")
        signals_sent = send_signals_to_discord(discord, final_signals)
        
        # Final summary
        print_comprehensive_summary(scanner, stage1_winners, final_signals, signals_sent)
        
        return signals_sent > 0
        
    except Exception as e:
        print(f"❌ Critical error in test pipeline: {e}")
        return False

if __name__ == "__main__":
    test_production_pipeline() 