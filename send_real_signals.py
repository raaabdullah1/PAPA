#!/usr/bin/env python3
"""
Send Real Signals to Discord
Generate signals from recent scan results and send to Discord
"""

from core.scanner import MarketScanner
from core.filter_tree import FilterTree
from core.notifier import DiscordNotifier
from core.orderblock_detector import OrderblockDetector
import logging
import time
from typing import Dict, List, Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Configurable settings for the confluence pipeline
CONFIG = {
    'stage1_timeframe': '15m',  # Single timeframe for Stage 1 filtering
    'TOP_N': 0,                 # 0 = scan all winners; >0 = shortlist top N by quality
    'TF_WEIGHTS': {
        '4h': 3,
        '1h': 2, 
        '30m': 1,
        '15m': 1
    },
    'CONFLUENCE_BONUSES': {
        'high': 2,    # ≥75% confluence (3+ timeframes)
        'medium': 1,  # ≥50% confluence (2+ timeframes)
        'low': 0      # <50% confluence
    },
    'FAST_FAIL_4H': True,  # Drop symbols that fail Stage 2 on 4h timeframe
    'MAX_WORKERS': 4,      # Parallel processing workers
    'PRODUCTION_MODE': True,  # Set to True for full production (all symbols)
    'TEST_SYMBOLS_LIMIT': 50,   # Number of symbols to test in non-production mode
    'STAGE3_THRESHOLD': 2,      # Minimum strategies passed for Stage 3 (was 3, now 2 for better signal capture)
    'MIN_RISK_REWARD': 1.15,    # Minimum risk-reward ratio for signal validation
    'MIN_CONFLUENCE': 50,       # Minimum confluence percentage for signal generation
    'MAX_SIGNALS_PER_SCAN': 5   # Maximum number of signals to send per scan (top N by quality)
}

def generate_real_signals():
    """Generate real signals using intelligent two-stage confluence pipeline"""
    print("🚀 Generating Real Signals with Confluence Pipeline")
    print("=" * 60)
    
    # Initialize components
    scanner = MarketScanner()
    filter_tree = FilterTree()
    discord = DiscordNotifier()
    
    # Display configuration
    print(f"📋 Configuration:")
    print(f"  Stage 1 Timeframe: {CONFIG['stage1_timeframe']}")
    print(f"  Top N Ranking: {CONFIG['TOP_N']} (0 = all winners)")
    print(f"  Fast-fail 4H: {CONFIG['FAST_FAIL_4H']}")
    print(f"  Max Workers: {CONFIG['MAX_WORKERS']}")
    print(f"  Production Mode: {CONFIG['PRODUCTION_MODE']}")
    print(f"  Stage 3 Threshold: {CONFIG['STAGE3_THRESHOLD']}/7")
    print(f"  Min Risk-Reward: {CONFIG['MIN_RISK_REWARD']}")
    print(f"  Min Confluence: {CONFIG['MIN_CONFLUENCE']}%")
    print(f"  Max Signals Per Scan: {CONFIG['MAX_SIGNALS_PER_SCAN']}")
    if not CONFIG['PRODUCTION_MODE']:
        print(f"  Test Symbols Limit: {CONFIG['TEST_SYMBOLS_LIMIT']}")
    
    # Step 1: Stage 1 filtering on single timeframe
    print(f"\n📊 Step 1: Stage 1 filtering on {CONFIG['stage1_timeframe']} timeframe...")
    
    try:
        # Get all symbols
        all_symbols = scanner.get_futures_symbols()
        print(f"📊 Total symbols available: {len(all_symbols)}")
        
        # Apply production/test mode filtering
        if not CONFIG['PRODUCTION_MODE']:
            symbols_to_process = all_symbols[:CONFIG['TEST_SYMBOLS_LIMIT']]
            print(f"🧪 TEST MODE: Processing first {len(symbols_to_process)} symbols")
        else:
            symbols_to_process = all_symbols
            print(f"🚀 PRODUCTION MODE: Processing all {len(symbols_to_process)} symbols")
        
        # Stage 1: Single timeframe filtering
        stage1_winners = run_stage1_single_timeframe(scanner, filter_tree, symbols_to_process, CONFIG['stage1_timeframe'])
        
        if not stage1_winners:
            print("❌ No symbols passed Stage 1 - aborting")
            return False
        
        print(f"✅ Stage 1 completed: {len(stage1_winners)} symbols passed")
        
        # Step 2: Optional quality ranking
        if CONFIG['TOP_N'] > 0:
            stage1_winners = rank_by_quality(stage1_winners, CONFIG['TOP_N'])
            print(f"📊 Quality ranking applied: Top {len(stage1_winners)} symbols selected")
        
        # Step 3: Multi-timeframe Stage 2 & 3 with confluence scoring
        print(f"\n📊 Step 2: Multi-timeframe Stage 2 & 3 for {len(stage1_winners)} symbols...")
        
        final_signals = run_multi_timeframe_confluence(
            scanner, filter_tree, stage1_winners
        )
        
        if not final_signals:
            print("❌ No signals generated - aborting")
            return False
        
        # Step 4: Rank signals by quality and apply cap
        print(f"\n📊 Ranking {len(final_signals)} signals by quality...")
        
        # Sort signals by (confidence + confluence strength) descending
        final_signals.sort(key=lambda s: (s.get('confidence_layers', 0) + s.get('confluence_percentage', 0)), reverse=True)
        
        # Apply signal cap
        original_count = len(final_signals)
        final_signals = final_signals[:CONFIG['MAX_SIGNALS_PER_SCAN']]
        
        print(f"📈 Signal ranking applied:")
        print(f"  🎯 Original signals: {original_count}")
        print(f"  🏆 Top signals selected: {len(final_signals)}")
        print(f"  📊 Quality range: {final_signals[0].get('confidence_layers', 0) + final_signals[0].get('confluence_percentage', 0):.1f} - {final_signals[-1].get('confidence_layers', 0) + final_signals[-1].get('confluence_percentage', 0):.1f}")
        
        # Step 5: Send signals to Discord
        print(f"\n📤 Sending {len(final_signals)} signals to Discord...")
        signals_sent = send_signals_to_discord(discord, final_signals)
        
        # Final summary
        print_comprehensive_summary(scanner, stage1_winners, final_signals, signals_sent)
        
        return signals_sent > 0
        
    except Exception as e:
        print(f"❌ Critical error in signal generation: {e}")
        return False 

def run_stage1_single_timeframe(scanner: MarketScanner, filter_tree: FilterTree, symbols: List[str], timeframe: str) -> List[Dict]:
    """Run Stage 1 filtering on all symbols using single timeframe"""
    print(f"📊 Running Stage 1 on {len(symbols)} symbols using {timeframe} timeframe...")
    
    # Fetch data for all symbols on single timeframe
    symbols_data = []
    successful_fetches = 0
    failed_fetches = 0
    invalid_symbols = 0
    
    for i, symbol in enumerate(symbols):
        try:
            if (i + 1) % 25 == 0:  # Progress update every 25 symbols
                print(f"  📈 Progress: {i+1}/{len(symbols)} symbols processed ({successful_fetches} successful, {failed_fetches} failed, {invalid_symbols} invalid)")
            
            # Validate symbol exists in exchange markets
            if symbol not in scanner.exchange.markets:
                invalid_symbols += 1
                continue
            
            # Fetch OHLCV data for single timeframe
            ohlcv_data = scanner.fetch_ohlcv_paginated(symbol, timeframe=timeframe, max_bars=2400)
            
            if not ohlcv_data or len(ohlcv_data) < 50:
                failed_fetches += 1
                continue
            
            # Fetch additional data with error handling
            try:
                ticker_data = scanner.get_ticker_data(symbol)
                if not ticker_data or not ticker_data.get('last'):
                    failed_fetches += 1
                    continue
            except Exception as e:
                failed_fetches += 1
                continue
            
            try:
                funding_rate = scanner.get_funding_rate(symbol)
                spread = scanner.calculate_spread(symbol)
            except:
                funding_rate = 0.0
                spread = 0.0
            
            # Prepare symbol data
            symbol_data = {
                'symbol': symbol,
                'ohlcv': ohlcv_data,
                'price': ticker_data.get('last', 0),
                'volume_usd': ticker_data.get('quoteVolume', 0),
                'change_24h': ticker_data.get('percentage', 0),
                'high_24h': ticker_data.get('high', 0),
                'low_24h': ticker_data.get('low', 0),
                'funding_rate': funding_rate or 0.0,
                'spread': spread or 0.0
            }
            
            symbols_data.append(symbol_data)
            successful_fetches += 1
            
            # Rate limiting for production
            if CONFIG['PRODUCTION_MODE'] and (i + 1) % 10 == 0:
                time.sleep(0.5)  # Brief pause every 10 symbols
            
        except Exception as e:
            failed_fetches += 1
            if failed_fetches % 10 == 0:  # Log every 10th failure
                print(f"    ⚠️ Batch failures: {failed_fetches} symbols failed")
            continue
    
    print(f"📊 Stage 1 data fetch: {successful_fetches} successful, {failed_fetches} failed, {invalid_symbols} invalid symbols")
    
    if len(symbols_data) < 10:
        print(f"❌ Insufficient symbols with valid data: {len(symbols_data)} (need at least 10)")
        return []
    
    # Run Stage 1 filter
    try:
        stage1_result = filter_tree.stage_1_filter_with_stored_data(symbols_data)
        print(f"✅ Stage 1 completed: {len(stage1_result)} symbols passed")
        return stage1_result
    except Exception as e:
        print(f"❌ Stage 1 filter error: {e}")
        return []

def rank_by_quality(symbols_data: List[Dict], top_n: int) -> List[Dict]:
    """Rank symbols by quality metric (24h volume) and return top N"""
    print(f"📊 Ranking {len(symbols_data)} symbols by quality...")
    
    # Sort by 24h volume (descending)
    ranked_symbols = sorted(symbols_data, key=lambda x: x.get('volume_usd', 0), reverse=True)
    
    # Return top N
    top_symbols = ranked_symbols[:top_n]
    print(f"✅ Quality ranking: Selected top {len(top_symbols)} symbols by volume")
    
    return top_symbols

def run_multi_timeframe_confluence(scanner: MarketScanner, filter_tree: FilterTree, stage1_winners: List[Dict]) -> List[Dict]:
    """Run Stage 2 & 3 on multiple timeframes with confluence scoring"""
    print(f"📊 Multi-timeframe confluence analysis for {len(stage1_winners)} symbols...")
    
    timeframes = ['15m', '30m', '1h', '4h']
    final_signals = []
    
    # Process symbols in parallel for better performance
    with ThreadPoolExecutor(max_workers=CONFIG['MAX_WORKERS']) as executor:
        # Submit all symbol processing tasks
        future_to_symbol = {
            executor.submit(process_symbol_confluence, scanner, filter_tree, symbol_data, timeframes): symbol_data
            for symbol_data in stage1_winners
        }
        
        # Collect results with progress tracking
        completed = 0
        for future in as_completed(future_to_symbol):
            symbol_data = future_to_symbol[future]
            completed += 1
            
            # Progress update
            if completed % 5 == 0 or completed == len(stage1_winners):
                print(f"  📈 Confluence progress: {completed}/{len(stage1_winners)} symbols processed")
            
            try:
                result = future.result()
                if result:
                    final_signals.append(result)
                    print(f"  ✅ {symbol_data['symbol']}: Confluence {result['confluence_percentage']:.1f}%, Confidence {result['confidence']}")
                else:
                    print(f"  ⚠️ {symbol_data['symbol']}: No signal generated")
            except Exception as e:
                print(f"  ❌ {symbol_data['symbol']}: Error - {e}")
                continue
    
    print(f"✅ Confluence analysis completed: {len(final_signals)} signals generated")
    return final_signals

def process_symbol_confluence(scanner: MarketScanner, filter_tree: FilterTree, symbol_data: Dict, timeframes: List[str]) -> Optional[Dict]:
    """Process a single symbol across multiple timeframes with orderblock detection"""
    symbol = symbol_data['symbol']
    
    # Initialize orderblock detector
    orderblock_detector = OrderblockDetector()
    
    # Fetch 1m data for resampling
    try:
        raw_1m_data = scanner.fetch_ohlcv_paginated(symbol, timeframe='1m', max_bars=12000)
        if not raw_1m_data or len(raw_1m_data) < 50:
            return None
    except Exception as e:
        return None
    
    timeframe_results = {}
    
    # Process each timeframe
    for timeframe in timeframes:
        try:
            # Resample 1m data to target timeframe
            resampled_data = scanner.resample_ohlcv(raw_1m_data, timeframe)
            if not resampled_data or len(resampled_data) < 50:
                continue
            
            # Run Stage 2 filter
            stage2_result = run_stage2_on_timeframe(filter_tree, symbol_data, resampled_data, timeframe)
            if not stage2_result:
                continue
            
            # Run Stage 3 filter
            stage3_result = run_stage3_on_timeframe(filter_tree, stage2_result, timeframe)
            if not stage3_result:
                continue
            
            # Detect orderblocks for this timeframe
            orderblocks = orderblock_detector.detect_orderblocks(resampled_data, timeframe)
            
            # Check for orderblock alignment
            current_price = symbol_data.get('price', 0)
            is_aligned, best_orderblock = orderblock_detector.check_orderblock_alignment(
                stage3_result, orderblocks, current_price, timeframe
            )
            
            # Add orderblock bonus to confidence if aligned
            if is_aligned and best_orderblock:
                stage3_result['confidence_layers'] += 1
                stage3_result['orderblock_bonus'] = True
                stage3_result['orderblock_type'] = best_orderblock.type
                stage3_result['orderblock_strength'] = best_orderblock.strength
            else:
                stage3_result['orderblock_bonus'] = False
            
            timeframe_results[timeframe] = stage3_result
            
        except Exception as e:
            continue
    
    if not timeframe_results:
        return None
    
    # Calculate confluence signal
    final_signal = calculate_confluence_signal(symbol, timeframe_results)
    
    # Check minimum confluence threshold
    if final_signal['confluence_percentage'] < CONFIG['MIN_CONFLUENCE']:
        return None
    
    return final_signal

def run_stage2_on_timeframe(filter_tree: FilterTree, symbol_data: Dict, ohlcv: List, timeframe: str) -> Optional[Dict]:
    """Run Stage 2 filter on specific timeframe"""
    try:
        # Prepare data for Stage 2
        stage2_data = {
            'symbol': symbol_data['symbol'],
            'ohlcv': ohlcv,
            'price': symbol_data['price'],
            'volume_usd': symbol_data['volume_usd'],
            'change_24h': symbol_data['change_24h'],
            'high_24h': symbol_data['high_24h'],
            'low_24h': symbol_data['low_24h'],
            'funding_rate': symbol_data['funding_rate'],
            'spread': symbol_data['spread']
        }
        
        # Run Stage 2
        stage2_result = filter_tree.stage_2_filter_with_stored_data([stage2_data])
        return stage2_result[0] if stage2_result else None
        
    except Exception as e:
        return None

def run_stage3_on_timeframe(filter_tree: FilterTree, stage2_data: Dict, timeframe: str) -> Optional[Dict]:
    """Run Stage 3 filter on specific timeframe"""
    try:
        # Run Stage 3 with configurable threshold
        stage3_result = filter_tree.stage_3_filter_with_stored_data([stage2_data], confidence_threshold=CONFIG['STAGE3_THRESHOLD'])
        if stage3_result:
            result = stage3_result[0]
            result['timeframe'] = timeframe
            return result
        return None
        
    except Exception as e:
        return None

def calculate_confluence_signal(symbol: str, timeframe_results: Dict[str, Dict]) -> Dict:
    """Calculate confluence signal with orderblock information"""
    # Calculate weighted confluence percentage
    total_weight = 0
    weighted_sum = 0
    
    for timeframe, result in timeframe_results.items():
        weight = CONFIG['TF_WEIGHTS'].get(timeframe, 1)
        total_weight += weight
        weighted_sum += weight
    
    confluence_percentage = (weighted_sum / total_weight) * 100 if total_weight > 0 else 0
    
    # Select best timeframe result based on confidence
    best_timeframe = max(timeframe_results.keys(), 
                        key=lambda tf: timeframe_results[tf].get('confidence_layers', 0))
    best_result = timeframe_results[best_timeframe]
    
    # Apply confluence bonuses
    confidence_bonus = 0
    if confluence_percentage >= 75:
        confidence_bonus = CONFIG['CONFLUENCE_BONUSES']['high']
    elif confluence_percentage >= 50:
        confidence_bonus = CONFIG['CONFLUENCE_BONUSES']['medium']
    
    # Check for orderblock bonus across timeframes
    orderblock_bonus = False
    orderblock_type = None
    orderblock_strength = 0
    
    for timeframe, result in timeframe_results.items():
        if result.get('orderblock_bonus', False):
            orderblock_bonus = True
            orderblock_type = result.get('orderblock_type', 'unknown')
            orderblock_strength = max(orderblock_strength, result.get('orderblock_strength', 0))
    
    # Calculate final confidence
    base_confidence = best_result.get('confidence_layers', 0)
    final_confidence = base_confidence + confidence_bonus
    
    # Create final signal
    final_signal = {
        'symbol': symbol,
        'direction': best_result.get('direction', 'UNKNOWN'),
        'timeframe': best_timeframe,
        'price': best_result.get('price', 0),
        'stop_loss': best_result.get('stop_loss', 0),
        'take_profit': best_result.get('take_profit', 0),
        'risk_reward': best_result.get('risk_reward', 0),
        'confidence': final_confidence,
        'confluence_percentage': confluence_percentage,
        'strategies': best_result.get('strategies', []),
        'orderblock_bonus': orderblock_bonus,
        'orderblock_type': orderblock_type,
        'orderblock_strength': orderblock_strength
    }
    
    return final_signal

def send_signals_to_discord(discord: DiscordNotifier, signals: List[Dict]) -> int:
    """Send signals to Discord with proper formatting"""
    signals_sent = 0
    
    for i, signal in enumerate(signals):
        try:
            # Format signal for Discord
            discord_signal = format_signal_for_discord(signal)
            
            # Send to Discord
            success = discord.send_signal_notification(discord_signal)
            
            if success:
                signals_sent += 1
                print(f"✅ Signal {i+1}/{len(signals)} sent: {signal['symbol']} {signal['signal_direction']} ({signal['timeframe']}) - Confluence: {signal['confluence_percentage']:.1f}%")
            else:
                print(f"❌ Failed to send signal {i+1}: {signal['symbol']}")
            
            # Rate limiting
            time.sleep(1)
            
        except Exception as e:
            print(f"❌ Error processing signal {i+1}: {e}")
            continue
    
    return signals_sent

def format_signal_for_discord(signal: Dict) -> str:
    """Format signal for Discord notification with star rating and orderblock information"""
    symbol = signal['symbol']
    direction = signal['signal_direction']
    timeframe = signal['timeframe']
    price = signal['price']
    stop_loss = signal['stop_loss']
    take_profit = signal['tp1_price'] # Assuming tp1_price is the primary take profit
    risk_reward = signal['risk_reward']
    confidence = signal['confidence_layers']
    confluence = signal['confluence_percentage']
    
    # Convert confidence score to star rating (0-7 → ★1-★5)
    if confidence <= 1:
        star_rating = "★1"
    elif confidence <= 3:
        star_rating = "★2"
    elif confidence <= 4:
        star_rating = "★3"
    elif confidence <= 5:
        star_rating = "★4"
    else:  # 6-7
        star_rating = "★5"
    
    # Check for orderblock bonus
    orderblock_info = ""
    if signal.get('orderblock_bonus', False):
        orderblock_type = signal.get('orderblock_type', 'unknown')
        orderblock_strength = signal.get('orderblock_strength', 0)
        orderblock_info = f" | Orderblock ✔ ({orderblock_type}, strength: {orderblock_strength:.2f})"
    
    # Format price with proper decimal places
    price_str = f"{price:.8f}".rstrip('0').rstrip('.')
    sl_str = f"{stop_loss:.8f}".rstrip('0').rstrip('.')
    tp_str = f"{take_profit:.8f}".rstrip('0').rstrip('.')
    
    # Risk-reward validation
    if risk_reward < CONFIG['MIN_RISK_REWARD']:
        return f"❌ {symbol}: Error - Risk-reward {risk_reward:.2f} < {CONFIG['MIN_RISK_REWARD']}"
    
    # Create signal message
    emoji = "🟢" if direction == "LONG" else "🔴"
    
    message = (
        f"{emoji} **{symbol} {direction}** ({timeframe}) {star_rating}\n"
        f"💰 **Price:** ${price_str}\n"
        f"🛑 **Stop Loss:** ${sl_str}\n"
        f"🎯 **Take Profit:** ${tp_str}\n"
        f"⚖️ **Risk/Reward:** {risk_reward:.2f}\n"
        f"🎯 **Confidence:** {confidence}/7 | **Confluence:** {confluence:.1f}%{orderblock_info}\n"
        f"⏰ **Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} PST"
    )
    
    return message

def print_comprehensive_summary(scanner: MarketScanner, stage1_winners: List[Dict], final_signals: List[Dict], signals_sent: int):
    """Print comprehensive summary of the confluence pipeline"""
    print("\n" + "=" * 60)
    print("🎯 CONFLUENCE PIPELINE SUMMARY:")
    print("=" * 60)
    
    print("📊 PIPELINE PERFORMANCE:")
    print(f"  📈 Stage 1 Winners: {len(stage1_winners)}")
    print(f"  🎯 Final Signals: {len(final_signals)}")
    print(f"  📤 Signals Sent: {signals_sent}")
    print(f"  ⚡ Success Rate: {(len(final_signals)/len(stage1_winners)*100):.1f}%" if stage1_winners else "N/A")
    
    print("\n📈 CONFLUENCE STATISTICS:")
    if final_signals:
        confluence_scores = [s['confluence_percentage'] for s in final_signals]
        avg_confluence = sum(confluence_scores) / len(confluence_scores)
        max_confluence = max(confluence_scores)
        min_confluence = min(confluence_scores)
        
        print(f"  📊 Average Confluence: {avg_confluence:.1f}%")
        print(f"  🏆 Highest Confluence: {max_confluence:.1f}%")
        print(f"  📉 Lowest Confluence: {min_confluence:.1f}%")
        
        # Count by confluence levels
        high_confluence = len([s for s in final_signals if s['confluence_percentage'] >= 75])
        medium_confluence = len([s for s in final_signals if 50 <= s['confluence_percentage'] < 75])
        low_confluence = len([s for s in final_signals if s['confluence_percentage'] < 50])
        
        print(f"  🟢 High Confluence (≥75%): {high_confluence} signals")
        print(f"  🟡 Medium Confluence (50-74%): {medium_confluence} signals")
        print(f"  🔴 Low Confluence (<50%): {low_confluence} signals")
    
    print("\n⚡ PERFORMANCE METRICS:")
    print(f"  🔄 Scanner API Calls: {scanner.metrics['total_fetches']}")
    print(f"  ✅ Successful Fetches: {scanner.metrics['successful_fetches']}")
    print(f"  ❌ Failed Fetches: {scanner.metrics['failed_fetches']}")
    print(f"  ⚠️ Rate Limit Hits: {scanner.metrics['rate_limit_hits']}")
    
    print(f"\n🎉 FINAL RESULT:")
    if signals_sent > 0:
        print("🚀 SUCCESS: Confluence signals sent to Discord!")
        print(f"✅ Pipeline completed successfully with {signals_sent} high-quality signals!")
    else:
        print("⚠️ No signals met the confluence criteria for sending")

def main():
    """Main function"""
    print("🤖 Crypto Signal Bot - Confluence Pipeline")
    print("=" * 60)
    
    success = generate_real_signals()
    
    if success:
        print("\n✅ Confluence pipeline completed successfully!")
    else:
        print("\n❌ Confluence pipeline failed!")
        exit(1)

if __name__ == "__main__":
    main() 