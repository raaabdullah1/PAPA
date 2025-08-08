import ccxt
import time
from typing import Dict, List, Optional, Tuple
import logging
from secrets import BINANCE_API_KEY, BINANCE_SECRET_KEY
from datetime import datetime, timedelta
import math
import random

# Constants for multi-timeframe scanning
MAX_BARS_NEEDED = 50 * 240  # 50 candles × highest timeframe (4h = 240m)
MIN_BARS_NEEDED = 50  # Minimum bars required for Stage 2
BATCH_SIZE = 1000  # Binance limit per request
RETRY_ATTEMPTS = 3  # Number of retry attempts for API calls
BASE_DELAY = 0.1  # Base delay for rate limiting
MAX_DELAY = 2.0  # Maximum delay for exponential backoff

class MarketScanner:
    def __init__(self):
        # Initialize logger first
        self.logger = logging.getLogger(__name__)
        
        # Try multiple API configurations
        self.exchange = None
        self.setup_exchange()
        
        # Metrics tracking
        self.metrics = {
            'total_fetches': 0,
            'successful_fetches': 0,
            'failed_fetches': 0,
            'retry_attempts': 0,
            'rate_limit_hits': 0,
            'insufficient_data_skips': 0,
            'invalid_symbol_skips': 0
        }
    
    def setup_exchange(self):
        """Setup exchange with multiple fallback options"""
        configs = [
            # Config 1: Futures trading with API keys (for funding rates)
            {
                'apiKey': BINANCE_API_KEY,
                'secret': BINANCE_SECRET_KEY,
                'sandbox': False,
                'enableRateLimit': True,
                'options': {'defaultType': 'future'}
            },
            # Config 2: Futures trading without API keys (public data only)
            {
                'sandbox': False,
                'enableRateLimit': True,
                'options': {'defaultType': 'future'}
            },
            # Config 3: Spot trading fallback (if futures fails)
            {
                'apiKey': BINANCE_API_KEY,
                'secret': BINANCE_SECRET_KEY,
                'sandbox': False,
                'enableRateLimit': True,
                'options': {'defaultType': 'spot'}
            },
            # Config 4: Testnet (if main API fails)
            {
                'apiKey': BINANCE_API_KEY,
                'secret': BINANCE_SECRET_KEY,
                'sandbox': True,
                'enableRateLimit': True,
                'options': {'defaultType': 'future'}
            }
        ]
        
        for i, config in enumerate(configs):
            try:
                self.logger.info(f"Trying API config {i+1}...")
                test_exchange = ccxt.binance(config)
                
                # Test the connection
                test_exchange.load_markets()
                ticker = test_exchange.fetch_ticker('BTC/USDT')
                
                if ticker and ticker.get('last'):
                    self.exchange = test_exchange
                    self.logger.info(f"✅ API config {i+1} working: BTC price = ${ticker['last']}")
                    return
                    
            except Exception as e:
                self.logger.warning(f"❌ API config {i+1} failed: {e}")
                continue
        
        # If all configs fail, create a basic exchange for fallback
        self.logger.error("All API configs failed, using fallback mode")
        self.exchange = ccxt.binance({
            'sandbox': False,
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        })
    
    def _exponential_backoff(self, attempt: int) -> float:
        """Calculate exponential backoff delay with jitter"""
        delay = min(BASE_DELAY * (2 ** attempt), MAX_DELAY)
        jitter = random.uniform(0, 0.1 * delay)
        return delay + jitter
    
    def _retry_api_call(self, func, *args, **kwargs):
        """Retry API call with exponential backoff"""
        for attempt in range(RETRY_ATTEMPTS):
            try:
                result = func(*args, **kwargs)
                if attempt > 0:
                    self.logger.debug(f"API call succeeded on attempt {attempt + 1}")
                return result
            except ccxt.RateLimitExceeded as e:
                self.metrics['rate_limit_hits'] += 1
                if attempt < RETRY_ATTEMPTS - 1:
                    delay = self._exponential_backoff(attempt)
                    self.logger.warning(f"Rate limit hit, retrying in {delay:.2f}s (attempt {attempt + 1}/{RETRY_ATTEMPTS})")
                    time.sleep(delay)
                else:
                    self.logger.error(f"Rate limit exceeded after {RETRY_ATTEMPTS} attempts: {e}")
                    raise
            except (ccxt.NetworkError, ccxt.ExchangeError) as e:
                if attempt < RETRY_ATTEMPTS - 1:
                    delay = self._exponential_backoff(attempt)
                    self.logger.warning(f"Network/Exchange error, retrying in {delay:.2f}s (attempt {attempt + 1}/{RETRY_ATTEMPTS}): {e}")
                    time.sleep(delay)
                else:
                    self.logger.error(f"API call failed after {RETRY_ATTEMPTS} attempts: {e}")
                    raise
            except Exception as e:
                self.logger.error(f"Unexpected error in API call: {e}")
                raise
        
        return None
    
    def get_futures_symbols(self) -> List[str]:
        """Get all USDT futures symbols (400+ pairs) with validation"""
        try:
            # Force reload markets to ensure we get the latest data
            self.exchange.load_markets(True)
            markets = self.exchange.markets
            
            # Get all USDT pairs
            usdt_symbols = []
            for symbol in markets.keys():
                if symbol.endswith('/USDT'):
                    usdt_symbols.append(symbol)
            
            self.logger.info(f"Found {len(usdt_symbols)} USDT symbols from Binance API")
            
            # Return all symbols for production use
            if len(usdt_symbols) >= 300:
                self.logger.info(f"✅ Production mode: Using {len(usdt_symbols)} symbols")
                return usdt_symbols
            else:
                self.logger.warning(f"Only found {len(usdt_symbols)} symbols, using fallback")
                # Fallback to common symbols if API returns too few
                fallback_symbols = [
                    'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'SOL/USDT',
                    'XRP/USDT', 'DOT/USDT', 'DOGE/USDT', 'AVAX/USDT', 'MATIC/USDT',
                    'LINK/USDT', 'UNI/USDT', 'LTC/USDT', 'BCH/USDT', 'XLM/USDT',
                    'VET/USDT', 'FIL/USDT', 'TRX/USDT', 'EOS/USDT', 'AAVE/USDT'
                ]
                return fallback_symbols
            
        except Exception as e:
            self.logger.error(f"Error fetching futures symbols: {e}")
            # Fallback to common symbols
            fallback_symbols = [
                'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'SOL/USDT',
                'XRP/USDT', 'DOT/USDT', 'DOGE/USDT', 'AVAX/USDT', 'MATIC/USDT',
                'LINK/USDT', 'UNI/USDT', 'LTC/USDT', 'BCH/USDT', 'XLM/USDT',
                'VET/USDT', 'FIL/USDT', 'TRX/USDT', 'EOS/USDT', 'AAVE/USDT'
            ]
            self.logger.info(f"Using fallback symbols: {len(fallback_symbols)}")
            return fallback_symbols
    
    def get_ticker_data(self, symbol: str) -> Optional[Dict]:
        """Get ticker data for a symbol with robust error handling and fallbacks"""
        try:
            # Try multiple methods to get ticker data
            methods = [
                lambda: self.exchange.fetch_ticker(symbol),
                lambda: self.exchange.fetch_ticker(symbol.replace('/USDT', 'USDT')),
                lambda: self.exchange.fetch_ticker(symbol.replace('USDT', '/USDT'))
            ]
            
            for method in methods:
                try:
                    ticker = method()
                    if ticker and ticker.get('last') and ticker.get('quoteVolume') is not None:
                        return ticker
                except Exception:
                    continue
            
            # If all methods fail, return None
            return None
            
        except Exception as e:
            self.logger.debug(f"Error fetching ticker for {symbol}: {e}")
            return None
    
    def get_ohlcv_data(self, symbol: str, timeframe: str = '15m', limit: int = 100) -> Optional[List]:
        """Get OHLCV data with multiple fallback methods"""
        try:
            # Try multiple methods
            methods = [
                lambda: self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit),
                lambda: self.exchange.fetch_ohlcv(symbol.replace('/USDT', 'USDT'), timeframe, limit=limit),
                lambda: self.exchange.fetch_ohlcv(symbol.replace('USDT', '/USDT'), timeframe, limit=limit)
            ]
            
            for method in methods:
                try:
                    ohlcv = method()
                    if ohlcv and len(ohlcv) >= 50:
                        return ohlcv
                except Exception:
                    continue
            
            return None
            
        except Exception as e:
            self.logger.debug(f"Error fetching OHLCV for {symbol}: {e}")
            return None
    
    def get_funding_rate(self, symbol: str) -> Optional[float]:
        """Get current funding rate with robust error handling"""
        try:
            funding_info = self.exchange.fetch_funding_rate(symbol)
            return funding_info['fundingRate'] if funding_info else None
        except Exception as e:
            # Log error and return None - don't abort entire scan
            self.logger.debug(f"Error fetching funding rate for {symbol}: {e}")
            return None
    
    def calculate_atr(self, ohlcv: List, period: int = 14) -> float:
        """Calculate Average True Range (without pandas)"""
        try:
            if len(ohlcv) < period + 1:
                return 0.0
            
            # Extract high, low, close from OHLCV data
            highs = [candle[2] for candle in ohlcv]
            lows = [candle[3] for candle in ohlcv]
            closes = [candle[4] for candle in ohlcv]
            
            # Calculate True Range
            tr_values = []
            for i in range(1, len(ohlcv)):
                high = highs[i]
                low = lows[i]
                prev_close = closes[i-1]
                
                tr1 = high - low
                tr2 = abs(high - prev_close)
                tr3 = abs(low - prev_close)
                
                tr = max(tr1, tr2, tr3)
                tr_values.append(tr)
            
            # Calculate ATR as simple moving average
            if len(tr_values) >= period:
                atr = sum(tr_values[-period:]) / period
                return atr
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating ATR: {e}")
            return 0.0
    
    def calculate_volume_rank(self, ohlcv: List, period: int = 20) -> float:
        """Calculate volume rank based on recent volume vs average (without pandas)"""
        try:
            if len(ohlcv) < period:
                return 0.0
            
            # Extract volumes from OHLCV data
            volumes = [candle[5] for candle in ohlcv]
            
            current_volume = volumes[-1]
            avg_volume = sum(volumes[-period:]) / period
            
            return (current_volume / avg_volume) * 100 if avg_volume > 0 else 0
            
        except Exception as e:
            self.logger.error(f"Error calculating volume rank: {e}")
            return 0.0
    
    def calculate_spread(self, symbol: str) -> Optional[float]:
        """Calculate bid/ask spread percentage with robust error handling"""
        try:
            # Use limit=5 instead of limit=1 (Binance requirement)
            orderbook = self.exchange.fetch_order_book(symbol, limit=5)
            best_bid = orderbook['bids'][0][0] if orderbook['bids'] else 0
            best_ask = orderbook['asks'][0][0] if orderbook['asks'] else 0
            
            if best_bid > 0 and best_ask > 0:
                spread = ((best_ask - best_bid) / best_bid) * 100
                return spread
            return None
        except Exception as e:
            # Log error and return None - don't abort entire scan
            self.logger.debug(f"Error calculating spread for {symbol}: {e}")
            return None
    
    def get_all_symbol_data(self, symbol: str, timeframe: str = '15m') -> Optional[Dict]:
        """
        Get comprehensive data for a single symbol with timeout protection
        Args:
            symbol: Symbol to fetch data for
            timeframe: Timeframe for OHLCV data (default: '15m')
        Returns:
            Dictionary with all symbol data or None if failed
        """
        try:
            # Set a timeout for the entire operation
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError(f"Timeout fetching data for {symbol}")
            
            # Set 10 second timeout
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(10)
            
            try:
                # Fetch all data
                ticker = self.get_ticker_data(symbol)
                if not ticker:
                    return None
                
                ohlcv = self.get_ohlcv_data(symbol, timeframe=timeframe)
                funding_rate = self.get_funding_rate(symbol)
                spread = self.calculate_spread(symbol)
                
                # Cancel timeout
                signal.alarm(0)
                
                return {
                    'symbol': symbol,
                    'price': ticker.get('last', 0),
                    'volume_usd': ticker.get('quoteVolume', 0),
                    'change_24h': ticker.get('percentage', 0),
                    'high_24h': ticker.get('high', 0),
                    'low_24h': ticker.get('low', 0),
                    'ohlcv': ohlcv,
                    'funding_rate': funding_rate,
                    'spread': spread,
                    'timeframe': timeframe
                }
                
            except TimeoutError:
                self.logger.debug(f"Timeout fetching data for {symbol}")
                return None
            finally:
                signal.alarm(0)
                
        except Exception as e:
            self.logger.debug(f"Error fetching all data for {symbol}: {e}")
            return None
    
    def scan_all_symbols_data(self, timeframe: str = '15m') -> List[Dict]:
        """
        Scan all futures symbols and fetch all required data in one pass
        Args:
            timeframe: Timeframe for OHLCV data (default: '15m')
        Returns:
            List of dictionaries with complete symbol data
        """
        symbols = self.get_futures_symbols()
        all_symbols_data = []
        failed_symbols = 0
        
        self.logger.info(f"Fetching all data for {len(symbols)} futures symbols on {timeframe} timeframe...")
        
        # Process symbols in smaller batches for better progress tracking
        batch_size = 20
        total_batches = (len(symbols) + batch_size - 1) // batch_size
        
        for batch_num in range(total_batches):
            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, len(symbols))
            batch_symbols = symbols[start_idx:end_idx]
            
            self.logger.info(f"Processing batch {batch_num + 1}/{total_batches} (symbols {start_idx + 1}-{end_idx})")
            
            for symbol in batch_symbols:
                try:
                    symbol_data = self.get_all_symbol_data(symbol, timeframe=timeframe)
                    if symbol_data:
                        all_symbols_data.append(symbol_data)
                    else:
                        failed_symbols += 1
                        
                except Exception as e:
                    failed_symbols += 1
                    self.logger.debug(f"Error processing {symbol}: {e}")
                    continue
            
            # Progress update after each batch
            processed = len(all_symbols_data) + failed_symbols
            self.logger.info(f"Batch {batch_num + 1} complete: {processed}/{len(symbols)} symbols processed")
        
        # Log scan results
        successful_scans = len(all_symbols_data)
        total_symbols = len(symbols)
        success_rate = (successful_scans / total_symbols * 100) if total_symbols > 0 else 0
        
        self.logger.info(f"Data fetch completed: {successful_scans}/{total_symbols} symbols processed successfully ({success_rate:.1f}% success rate)")
        if failed_symbols > 0:
            self.logger.info(f"Failed symbols: {failed_symbols} (gracefully skipped)")
        
        return all_symbols_data
    
    def scan_markets(self, min_volume_usd: float = 0) -> List[Dict]:
        """Scan ALL futures markets (400+ pairs) and rank them with optimized data fetching"""
        all_symbols_data = self.scan_all_symbols_data()
        
        # Convert to the expected format for backward compatibility
        market_data = []
        for symbol_data in all_symbols_data:
            try:
                # Calculate additional metrics using stored OHLCV data
                ohlcv = symbol_data.get('ohlcv')
                atr = self.calculate_atr(ohlcv) if ohlcv else 0
                volume_rank = self.calculate_volume_rank(ohlcv) if ohlcv else 0
                
                # Calculate technical score
                technical_score = 0
                if volume_rank > 100:
                    technical_score += 1
                if symbol_data.get('funding_rate') and abs(symbol_data['funding_rate']) < 0.0005:
                    technical_score += 1
                if symbol_data.get('spread') and symbol_data['spread'] < 0.1:
                    technical_score += 1
                
                market_data.append({
                    'symbol': symbol_data['symbol'],
                    'price': symbol_data['price'],
                    'volume_usd': symbol_data['volume_usd'],
                    'volume_rank': volume_rank,
                    'atr': atr,
                    'funding_rate': symbol_data['funding_rate'],
                    'spread': symbol_data['spread'],
                    'technical_score': technical_score,
                    'change_24h': symbol_data['change_24h'],
                    'high_24h': symbol_data['high_24h'],
                    'low_24h': symbol_data['low_24h']
                })
                
            except Exception as e:
                self.logger.debug(f"Error processing market data for {symbol_data['symbol']}: {e}")
                continue
        
        # Sort by technical score and volume
        market_data.sort(key=lambda x: (x['technical_score'], x['volume_usd']), reverse=True)
        
        return market_data
    
    def get_top_markets(self, top_n: int = 20) -> List[Dict]:
        """Get top N markets based on ranking"""
        markets = self.scan_markets()
        return markets[:top_n] 

    def fetch_ohlcv_all_symbols(self, timeframe: str = '1m', limit: int = MAX_BARS_NEEDED) -> Dict[str, List]:
        """
        Fetch raw OHLCV data for all symbols in a single pass using pagination
        Args:
            timeframe: Timeframe for OHLCV data (default: '1m')
            limit: Number of candles to fetch (default: MAX_BARS_NEEDED)
        Returns:
            Dict {symbol: List[(timestamp, open, high, low, close, volume)]}
        """
        start_time = time.time()
        symbols = self.get_futures_symbols()
        raw_ohlcv = {}
        
        # Reset metrics for this scan
        self.metrics = {
            'total_fetches': 0,
            'successful_fetches': 0,
            'failed_fetches': 0,
            'retry_attempts': 0,
            'rate_limit_hits': 0,
            'insufficient_data_skips': 0,
            'invalid_symbol_skips': 0
        }
        
        self.logger.info(f"🚀 Starting {timeframe} OHLCV fetch for {len(symbols)} symbols (target: {limit} bars per symbol)")
        
        # Process symbols in optimized batches
        batch_size = 25  # Increased batch size for better performance
        total_batches = (len(symbols) + batch_size - 1) // batch_size
        
        for batch_num in range(total_batches):
            batch_start_time = time.time()
            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, len(symbols))
            batch_symbols = symbols[start_idx:end_idx]
            
            self.logger.info(f"📊 Processing batch {batch_num + 1}/{total_batches} (symbols {start_idx + 1}-{end_idx})")
            
            batch_success = 0
            batch_failed = 0
            
            for symbol in batch_symbols:
                try:
                    # Use paginated fetch to get sufficient data
                    ohlcv_data = self.fetch_ohlcv_paginated(symbol, timeframe=timeframe, max_bars=limit)
                    if ohlcv_data and len(ohlcv_data) >= MIN_BARS_NEEDED:
                        raw_ohlcv[symbol] = ohlcv_data
                        batch_success += 1
                    else:
                        batch_failed += 1
                        
                except Exception as e:
                    batch_failed += 1
                    self.logger.debug(f"Error fetching {timeframe} data for {symbol}: {e}")
                    continue
            
            # Batch completion metrics
            batch_time = time.time() - batch_start_time
            self.logger.info(f"✅ Batch {batch_num + 1} complete: {batch_success} success, {batch_failed} failed ({batch_time:.1f}s)")
            
            # Adaptive rate limiting between batches
            if batch_num < total_batches - 1:  # Don't sleep after last batch
                time.sleep(0.5)  # Brief pause between batches
        
        # Final scan results and metrics
        scan_time = time.time() - start_time
        successful_fetches = len(raw_ohlcv)
        total_symbols = len(symbols)
        success_rate = (successful_fetches / total_symbols * 100) if total_symbols > 0 else 0
        
        self.logger.info(f"📊 {timeframe} scan completed in {scan_time:.1f}s:")
        self.logger.info(f"  ✅ Successful: {successful_fetches}/{total_symbols} symbols ({success_rate:.1f}%)")
        self.logger.info(f"  ❌ Failed: {self.metrics['failed_fetches']} symbols")
        self.logger.info(f"  ⚠️ Insufficient data: {self.metrics['insufficient_data_skips']} symbols")
        self.logger.info(f"  🔄 Rate limit hits: {self.metrics['rate_limit_hits']}")
        self.logger.info(f"  📈 Total API calls: {self.metrics['total_fetches']}")
        
        # Performance warning if scan took too long
        if scan_time > 300:  # 5 minutes
            self.logger.warning(f"⚠️ Scan took {scan_time:.1f}s - consider optimizing batch size or reducing symbols")
        
        return raw_ohlcv
    
    def fetch_ohlcv_paginated(self, symbol: str, timeframe: str = '1m', max_bars: int = MAX_BARS_NEEDED):
        """Fetch OHLCV data with pagination and robust error handling"""
        self.metrics['total_fetches'] += 1
        
        # Validate symbol exists in exchange markets
        if symbol not in self.exchange.markets:
            self.logger.warning(f"Symbol {symbol} not found in exchange markets - skipping")
            self.metrics['invalid_symbol_skips'] += 1
            return None
        
        try:
            # Use simple fetch without since parameter for now
            all_bars = self._retry_api_call(
                self.exchange.fetch_ohlcv,
                symbol, timeframe, limit=max_bars
            )
            
            if not all_bars:
                self.logger.warning(f"No data returned for {symbol}")
                self.metrics['insufficient_data_skips'] += 1
                return None
            
            # Validate minimum bars requirement
            if len(all_bars) < MIN_BARS_NEEDED:
                self.logger.warning(f"Insufficient data: {len(all_bars)} candles (need {MIN_BARS_NEEDED})")
                self.metrics['insufficient_data_skips'] += 1
                return None
            
            self.metrics['successful_fetches'] += 1
            self.logger.info(f"[{symbol}] fetched {len(all_bars)} {timeframe} bars")
            return all_bars
            
        except Exception as e:
            self.logger.error(f"Error fetching {timeframe} data for {symbol}: {e}")
            self.metrics['failed_fetches'] += 1
            return None
    
    def resample_ohlcv(self, raw_ohlcv: List, target_timeframe: str) -> List:
        """
        Resample 1m OHLCV data to target timeframe using pure Python with validation
        Args:
            raw_ohlcv: List of (timestamp, open, high, low, close, volume) tuples
            target_timeframe: Target timeframe ('15m', '30m', '1h', '4h')
        Returns:
            List of resampled OHLCV data
        """
        if not raw_ohlcv or len(raw_ohlcv) < 10:
            self.logger.debug(f"Resampling {target_timeframe}: insufficient input data ({len(raw_ohlcv) if raw_ohlcv else 0} bars)")
            return []
        
        self.logger.debug(f"Resampling {target_timeframe}: input {len(raw_ohlcv)} 1m bars")
        
        try:
            # Validate input data
            for i, candle in enumerate(raw_ohlcv):
                if len(candle) != 6:
                    self.logger.warning(f"Invalid candle format at index {i}: {candle}")
                    return []
                if not all(isinstance(x, (int, float)) for x in candle[1:6]):
                    self.logger.warning(f"Invalid candle data types at index {i}: {candle}")
                    return []
            
            # Convert timestamps to datetime objects for easier manipulation
            ohlcv_with_dt = []
            for candle in raw_ohlcv:
                timestamp_ms = candle[0]
                dt = datetime.fromtimestamp(timestamp_ms / 1000)
                ohlcv_with_dt.append((dt, candle[1], candle[2], candle[3], candle[4], candle[5]))
            
            # Sort by timestamp
            ohlcv_with_dt.sort(key=lambda x: x[0])
            
            # Determine interval in minutes
            if target_timeframe == '15m':
                interval_minutes = 15
            elif target_timeframe == '30m':
                interval_minutes = 30
            elif target_timeframe == '1h':
                interval_minutes = 60
            elif target_timeframe == '4h':
                interval_minutes = 240
            else:
                self.logger.warning(f"Unsupported timeframe for resampling: {target_timeframe}")
                return []
            
            # Calculate expected number of resampled bars
            time_span_minutes = (ohlcv_with_dt[-1][0] - ohlcv_with_dt[0][0]).total_seconds() / 60
            expected_bars = math.ceil(time_span_minutes / interval_minutes)
            
            # Group candles by interval - use a more robust approach
            resampled_candles = []
            candles_by_interval = {}
            
            for dt, open_price, high, low, close, volume in ohlcv_with_dt:
                # Calculate interval start time (round down to nearest interval)
                interval_start = dt.replace(
                    minute=(dt.minute // interval_minutes) * interval_minutes,
                    second=0,
                    microsecond=0
                )
                
                # Use interval start as key
                if interval_start not in candles_by_interval:
                    candles_by_interval[interval_start] = []
                
                candles_by_interval[interval_start].append((dt, open_price, high, low, close, volume))
            
            # Process each interval
            for interval_start in sorted(candles_by_interval.keys()):
                interval_candles = candles_by_interval[interval_start]
                if interval_candles:
                    resampled_candle = self._aggregate_candles(interval_candles)
                    resampled_candles.append(resampled_candle)
            
            # Interval coverage validation
            actual_bars = len(resampled_candles)
            coverage_ratio = actual_bars / expected_bars if expected_bars > 0 else 0
            
            if coverage_ratio < 0.8:  # Less than 80% coverage
                self.logger.warning(f"Resampling {target_timeframe}: low coverage {coverage_ratio:.2f} ({actual_bars}/{expected_bars} bars)")
            elif coverage_ratio > 1.2:  # More than 120% coverage
                self.logger.warning(f"Resampling {target_timeframe}: high coverage {coverage_ratio:.2f} ({actual_bars}/{expected_bars} bars)")
            
            # Ensure minimum bars for Stage 2
            if actual_bars < MIN_BARS_NEEDED:
                self.logger.warning(f"Resampling {target_timeframe}: insufficient output {actual_bars} bars (need {MIN_BARS_NEEDED})")
                return []
            
            self.logger.debug(f"Resampling {target_timeframe}: output {actual_bars} bars (expected ~{expected_bars}, coverage: {coverage_ratio:.2f})")
            return resampled_candles
            
        except Exception as e:
            self.logger.error(f"Error resampling OHLCV data to {target_timeframe}: {e}")
            return []
    
    def _aggregate_candles(self, candles: List[Tuple]) -> List:
        """
        Aggregate multiple candles into a single OHLCV candle
        Args:
            candles: List of (datetime, open, high, low, close, volume) tuples
        Returns:
            Single OHLCV candle as [timestamp, open, high, low, close, volume]
        """
        if not candles:
            return []
        
        # Sort by timestamp
        candles.sort(key=lambda x: x[0])
        
        # Extract OHLCV values
        opens = [c[1] for c in candles]
        highs = [c[2] for c in candles]
        lows = [c[3] for c in candles]
        closes = [c[4] for c in candles]
        volumes = [c[5] for c in candles]
        
        # Aggregate OHLCV
        open_price = opens[0]  # First open
        high_price = max(highs)  # Highest high
        low_price = min(lows)   # Lowest low
        close_price = closes[-1]  # Last close
        total_volume = sum(volumes)  # Sum of volumes
        
        # Use timestamp of the first candle
        timestamp_ms = int(candles[0][0].timestamp() * 1000)
        
        return [timestamp_ms, open_price, high_price, low_price, close_price, total_volume] 

    def _get_interval_ms(self, timeframe: str) -> int:
        """Convert timeframe string to milliseconds"""
        timeframe_map = {
            '1m': 60 * 1000,
            '5m': 5 * 60 * 1000,
            '15m': 15 * 60 * 1000,
            '30m': 30 * 60 * 1000,
            '1h': 60 * 60 * 1000,
            '4h': 4 * 60 * 60 * 1000,
            '1d': 24 * 60 * 60 * 1000
        }
        return timeframe_map.get(timeframe, 60 * 1000)  # Default to 1m 