import requests
import time
from typing import Dict, List, Optional
import logging
# API key not configured in secrets.py - using placeholder
COINGLASS_API_KEY = ""

class LiquidationTracker:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.coinglass_api_key = COINGLASS_API_KEY
        self.base_url = "https://open-api.coinglass.com/api/pro/v1"
        
    def get_liquidation_data(self, symbol: str = None) -> Optional[Dict]:
        """Get liquidation data from Coinglass API"""
        try:
            if not self.coinglass_api_key:
                self.logger.warning("Coinglass API key not configured")
                return self._get_mock_liquidation_data(symbol)
            
            headers = {
                'accept': 'application/json',
                'CG-API-KEY': self.coinglass_api_key
            }
            
            # Get liquidation data
            url = f"{self.base_url}/futures/liquidation"
            params = {}
            if symbol:
                params['symbol'] = symbol.replace('/USDT', '')
            
            response = requests.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get('success') and data.get('data'):
                return self._parse_liquidation_data(data['data'], symbol)
            else:
                self.logger.warning(f"Coinglass API returned error: {data}")
                return self._get_mock_liquidation_data(symbol)
                
        except Exception as e:
            self.logger.error(f"Error fetching liquidation data: {e}")
            return self._get_mock_liquidation_data(symbol)
    
    def _parse_liquidation_data(self, data: Dict, symbol: str) -> Dict:
        """Parse liquidation data from Coinglass API response"""
        try:
            # Extract relevant liquidation information
            liquidation_data = {
                'symbol': symbol,
                'timestamp': time.time(),
                'total_liquidations': 0,
                'long_liquidations': 0,
                'short_liquidations': 0,
                'liquidation_value_usd': 0,
                'liquidation_density': 0,
                'liquidation_clusters': []
            }
            
            # Parse the data structure (adjust based on actual API response)
            if isinstance(data, list):
                for item in data:
                    if item.get('symbol', '').replace('USDT', '') == symbol.replace('/USDT', ''):
                        liquidation_data['total_liquidations'] += item.get('total', 0)
                        liquidation_data['long_liquidations'] += item.get('long', 0)
                        liquidation_data['short_liquidations'] += item.get('short', 0)
                        liquidation_data['liquidation_value_usd'] += item.get('value', 0)
            
            # Calculate liquidation density (liquidations per hour)
            liquidation_data['liquidation_density'] = liquidation_data['total_liquidations'] / 24  # per hour
            
            return liquidation_data
            
        except Exception as e:
            self.logger.error(f"Error parsing liquidation data: {e}")
            return self._get_mock_liquidation_data(symbol)
    
    def _get_mock_liquidation_data(self, symbol: str) -> Dict:
        """Get mock liquidation data for testing"""
        return {
            'symbol': symbol,
            'timestamp': time.time(),
            'total_liquidations': 150,
            'long_liquidations': 80,
            'short_liquidations': 70,
            'liquidation_value_usd': 2500000,
            'liquidation_density': 6.25,  # liquidations per hour
            'liquidation_clusters': [
                {'price_level': 45000, 'liquidations': 25, 'side': 'long'},
                {'price_level': 44000, 'liquidations': 30, 'side': 'short'},
                {'price_level': 46000, 'liquidations': 20, 'side': 'long'}
            ]
        }
    
    def get_liquidation_heatmap(self, symbols: List[str] = None) -> Dict:
        """Get liquidation heatmap for multiple symbols"""
        try:
            if not symbols:
                symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'SOL/USDT']
            
            heatmap_data = {
                'timestamp': time.time(),
                'symbols': {}
            }
            
            for symbol in symbols:
                liquidation_data = self.get_liquidation_data(symbol)
                if liquidation_data:
                    heatmap_data['symbols'][symbol] = liquidation_data
            
            return heatmap_data
            
        except Exception as e:
            self.logger.error(f"Error generating liquidation heatmap: {e}")
            return {'timestamp': time.time(), 'symbols': {}}
    
    def detect_liquidation_clusters(self, symbol: str) -> List[Dict]:
        """Detect significant liquidation clusters"""
        try:
            liquidation_data = self.get_liquidation_data(symbol)
            if not liquidation_data:
                return []
            
            clusters = liquidation_data.get('liquidation_clusters', [])
            
            # Filter for significant clusters (more than 10 liquidations)
            significant_clusters = [
                cluster for cluster in clusters 
                if cluster.get('liquidations', 0) > 10
            ]
            
            return significant_clusters
            
        except Exception as e:
            self.logger.error(f"Error detecting liquidation clusters: {e}")
            return []
    
    def get_liquidation_signal(self, symbol: str) -> Optional[Dict]:
        """Generate liquidation-based trading signal"""
        try:
            liquidation_data = self.get_liquidation_data(symbol)
            if not liquidation_data:
                return None
            
            # Analyze liquidation patterns
            long_liquidations = liquidation_data.get('long_liquidations', 0)
            short_liquidations = liquidation_data.get('short_liquidations', 0)
            total_liquidations = liquidation_data.get('total_liquidations', 0)
            
            if total_liquidations == 0:
                return None
            
            # Calculate liquidation ratios
            long_ratio = long_liquidations / total_liquidations
            short_ratio = short_liquidations / total_liquidations
            
            # Generate signal based on liquidation imbalance
            signal = None
            confidence = 0.0
            
            if long_ratio > 0.7:  # Heavy long liquidations
                signal = 'LONG'  # Potential reversal to upside
                confidence = min(long_ratio * 1.5, 1.0)
            elif short_ratio > 0.7:  # Heavy short liquidations
                signal = 'SHORT'  # Potential reversal to downside
                confidence = min(short_ratio * 1.5, 1.0)
            
            if signal and confidence > 0.6:
                return {
                    'symbol': symbol,
                    'signal': signal,
                    'confidence': confidence,
                    'long_liquidations': long_liquidations,
                    'short_liquidations': short_liquidations,
                    'total_liquidations': total_liquidations,
                    'liquidation_value_usd': liquidation_data.get('liquidation_value_usd', 0),
                    'timestamp': time.time()
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error generating liquidation signal: {e}")
            return None
    
    def get_market_liquidation_summary(self) -> Dict:
        """Get overall market liquidation summary"""
        try:
            # Get liquidation data for major symbols
            major_symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'SOL/USDT']
            
            total_market_liquidations = 0
            total_market_value = 0
            symbol_summaries = {}
            
            for symbol in major_symbols:
                liquidation_data = self.get_liquidation_data(symbol)
                if liquidation_data:
                    total_market_liquidations += liquidation_data.get('total_liquidations', 0)
                    total_market_value += liquidation_data.get('liquidation_value_usd', 0)
                    symbol_summaries[symbol] = liquidation_data
            
            return {
                'timestamp': time.time(),
                'total_market_liquidations': total_market_liquidations,
                'total_market_value_usd': total_market_value,
                'symbol_summaries': symbol_summaries,
                'market_liquidation_density': total_market_liquidations / 24  # per hour
            }
            
        except Exception as e:
            self.logger.error(f"Error getting market liquidation summary: {e}")
            return {
                'timestamp': time.time(),
                'total_market_liquidations': 0,
                'total_market_value_usd': 0,
                'symbol_summaries': {},
                'market_liquidation_density': 0
            } 