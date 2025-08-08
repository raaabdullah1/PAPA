import requests
import json
import time
from typing import Dict, List, Optional
import logging
# API keys not configured in secrets.py - using placeholders
CRYPTOPANIC_API_KEY = ""
TRADINGVIEW_WEBHOOK_URL = ""

class SentimentAnalyzer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.cryptopanic_api_key = CRYPTOPANIC_API_KEY
        self.tradingview_webhook_url = TRADINGVIEW_WEBHOOK_URL
        
        # Sentiment thresholds
        self.bullish_threshold = 0.6
        self.bearish_threshold = 0.4
        self.neutral_threshold = 0.5
        
    def get_cryptopanic_sentiment(self, symbol: str = None) -> Optional[Dict]:
        """Get sentiment data from CryptoPanic API"""
        try:
            if not self.cryptopanic_api_key:
                self.logger.warning("CryptoPanic API key not configured")
                return None
            
            # Build API URL
            base_url = "https://cryptopanic.com/api/v1/posts/"
            params = {
                'auth_token': self.cryptopanic_api_key,
                'currencies': symbol.replace('/USDT', '') if symbol else 'BTC',
                'filter': 'hot',
                'public': 'true'
            }
            
            response = requests.get(base_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Analyze sentiment from news
            bullish_count = 0
            bearish_count = 0
            neutral_count = 0
            total_count = 0
            
            for post in data.get('results', []):
                vote = post.get('vote', 'neutral')
                if vote == 'positive':
                    bullish_count += 1
                elif vote == 'negative':
                    bearish_count += 1
                else:
                    neutral_count += 1
                total_count += 1
            
            if total_count == 0:
                return None
            
            # Calculate sentiment scores
            bullish_score = bullish_count / total_count
            bearish_score = bearish_count / total_count
            neutral_score = neutral_count / total_count
            
            # Determine overall sentiment
            if bullish_score > self.bullish_threshold:
                overall_sentiment = 'bullish'
            elif bearish_score > self.bearish_threshold:
                overall_sentiment = 'bearish'
            else:
                overall_sentiment = 'neutral'
            
            return {
                'symbol': symbol,
                'bullish_score': bullish_score,
                'bearish_score': bearish_score,
                'neutral_score': neutral_score,
                'overall_sentiment': overall_sentiment,
                'total_news_count': total_count,
                'timestamp': time.time()
            }
            
        except Exception as e:
            self.logger.error(f"Error fetching CryptoPanic sentiment: {e}")
            return None
    
    def get_tradingview_sentiment(self, symbol: str) -> Optional[Dict]:
        """Get sentiment data from TradingView (placeholder for webhook integration)"""
        try:
            if not self.tradingview_webhook_url:
                self.logger.warning("TradingView webhook URL not configured")
                return None
            
            # This is a placeholder - in a real implementation, you would:
            # 1. Set up a webhook endpoint to receive TradingView alerts
            # 2. Parse the webhook data for sentiment indicators
            # 3. Store and analyze the sentiment data
            
            # For now, return a mock sentiment based on symbol
            mock_sentiment = {
                'symbol': symbol,
                'technical_sentiment': 'bullish',  # Based on technical indicators
                'social_sentiment': 'neutral',     # Based on social media
                'overall_sentiment': 'bullish',
                'confidence': 0.75,
                'timestamp': time.time()
            }
            
            return mock_sentiment
            
        except Exception as e:
            self.logger.error(f"Error fetching TradingView sentiment: {e}")
            return None
    
    def analyze_market_sentiment(self, symbol: str) -> Dict:
        """Analyze overall market sentiment combining multiple sources"""
        try:
            sentiment_data = {
                'symbol': symbol,
                'cryptopanic': None,
                'tradingview': None,
                'overall_sentiment': 'neutral',
                'confidence': 0.5,
                'timestamp': time.time()
            }
            
            # Get CryptoPanic sentiment
            cryptopanic_sentiment = self.get_cryptopanic_sentiment(symbol)
            if cryptopanic_sentiment:
                sentiment_data['cryptopanic'] = cryptopanic_sentiment
            
            # Get TradingView sentiment
            tradingview_sentiment = self.get_tradingview_sentiment(symbol)
            if tradingview_sentiment:
                sentiment_data['tradingview'] = tradingview_sentiment
            
            # Combine sentiment scores
            sentiment_scores = []
            weights = []
            
            if cryptopanic_sentiment:
                if cryptopanic_sentiment['overall_sentiment'] == 'bullish':
                    sentiment_scores.append(0.8)
                elif cryptopanic_sentiment['overall_sentiment'] == 'bearish':
                    sentiment_scores.append(0.2)
                else:
                    sentiment_scores.append(0.5)
                weights.append(0.6)  # CryptoPanic weight
            
            if tradingview_sentiment:
                if tradingview_sentiment['overall_sentiment'] == 'bullish':
                    sentiment_scores.append(0.8)
                elif tradingview_sentiment['overall_sentiment'] == 'bearish':
                    sentiment_scores.append(0.2)
                else:
                    sentiment_scores.append(0.5)
                weights.append(0.4)  # TradingView weight
            
            # Calculate weighted average
            if sentiment_scores and weights:
                total_weight = sum(weights)
                weighted_score = sum(score * weight for score, weight in zip(sentiment_scores, weights)) / total_weight
                
                if weighted_score > self.bullish_threshold:
                    sentiment_data['overall_sentiment'] = 'bullish'
                elif weighted_score < self.bearish_threshold:
                    sentiment_data['overall_sentiment'] = 'bearish'
                else:
                    sentiment_data['overall_sentiment'] = 'neutral'
                
                sentiment_data['confidence'] = weighted_score
            
            return sentiment_data
            
        except Exception as e:
            self.logger.error(f"Error analyzing market sentiment: {e}")
            return {
                'symbol': symbol,
                'overall_sentiment': 'neutral',
                'confidence': 0.5,
                'error': str(e),
                'timestamp': time.time()
            }
    
    def get_sentiment_signal(self, symbol: str) -> Optional[str]:
        """Get sentiment-based trading signal"""
        try:
            sentiment_data = self.analyze_market_sentiment(symbol)
            
            # Only generate signals for strong sentiment
            if sentiment_data['confidence'] > 0.7:
                if sentiment_data['overall_sentiment'] == 'bullish':
                    return 'LONG'
                elif sentiment_data['overall_sentiment'] == 'bearish':
                    return 'SHORT'
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting sentiment signal: {e}")
            return None
    
    def get_market_fear_greed_index(self) -> Optional[Dict]:
        """Get market fear and greed index (placeholder)"""
        try:
            # This would typically fetch from alternative.me API or similar
            # For now, return a mock value
            return {
                'value': 65,  # 0-100 scale
                'classification': 'Greed',
                'timestamp': time.time()
            }
        except Exception as e:
            self.logger.error(f"Error fetching fear/greed index: {e}")
            return None
    
    def get_social_sentiment(self, symbol: str) -> Optional[Dict]:
        """Get social media sentiment (placeholder for Twitter/Reddit integration)"""
        try:
            # This would integrate with Twitter API, Reddit API, etc.
            # For now, return mock data
            return {
                'symbol': symbol,
                'twitter_sentiment': 'bullish',
                'reddit_sentiment': 'neutral',
                'telegram_sentiment': 'bullish',
                'overall_social_sentiment': 'bullish',
                'social_volume': 1500,
                'timestamp': time.time()
            }
        except Exception as e:
            self.logger.error(f"Error fetching social sentiment: {e}")
            return None 