#!/usr/bin/env python3
"""
Discord Webhook Notification System for Crypto Signal Bot
Works in Pakistan (Discord is not banned)
"""

import requests
import logging
from datetime import datetime
from typing import Dict, Optional

class DiscordNotifier:
    def __init__(self, webhook_url: str = None):
        self.logger = logging.getLogger(__name__)
        self.webhook_url = webhook_url or "https://discord.com/api/webhooks/1400399801962467388/j3bH1b_EsVuro3uOGnCE3difBSuyVnjkvJ2K3VnnnRkqvr_AtD90pgDNJDxzaH6zOomc"
        
    def send_signal_notification(self, signal_data: Dict) -> bool:
        """Send signal notification to Discord"""
        try:
            if not self.webhook_url:
                self.logger.warning("Discord webhook URL not configured")
                return False
            
            # Format the signal message using the exact template
            message = self.format_signal_message(signal_data)
            
            # Create Discord embed
            embed = {
                "title": "🚀 Crypto Signal Alert",
                "description": message,
                "color": 0x00ff00,  # Green color
                "timestamp": datetime.now().isoformat(),
                "footer": {
                    "text": "Multi-Timeframe Crypto Signal Bot"
                }
            }
            
            # Prepare Discord payload
            payload = {
                "embeds": [embed]
            }
            
            # Send to Discord
            response = requests.post(self.webhook_url, json=payload, timeout=10)
            
            if response.status_code == 204:
                self.logger.info(f"Discord notification sent successfully for {signal_data.get('symbol', 'Unknown')}")
                return True
            else:
                self.logger.error(f"Discord notification failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error sending Discord notification: {e}")
            return False
    
    def send_daily_summary(self, summary_data):
        """Send daily summary to Discord"""
        try:
            embed = {
                "title": "📊 Daily Crypto Bot Summary",
                "description": f"**Date:** {datetime.now().strftime('%Y-%m-%d')}",
                "color": 0x0099ff,  # Blue color
                "fields": [
                    {
                        "name": "📈 Total Signals",
                        "value": str(summary_data.get('total_signals', 0)),
                        "inline": True
                    },
                    {
                        "name": "✅ Successful Trades",
                        "value": str(summary_data.get('successful_trades', 0)),
                        "inline": True
                    },
                    {
                        "name": "💰 Total P&L",
                        "value": f"{summary_data.get('total_pnl', 0):.2f}%",
                        "inline": True
                    },
                    {
                        "name": "🎯 Win Rate",
                        "value": f"{summary_data.get('win_rate', 0):.1f}%",
                        "inline": True
                    },
                    {
                        "name": "🤖 Bot Status",
                        "value": "✅ Running",
                        "inline": True
                    },
                    {
                        "name": "📱 Notifications",
                        "value": "Discord (Working in Pakistan)",
                        "inline": True
                    }
                ],
                "footer": {
                    "text": "🤖 Crypto Signal Bot | Pakistan | Discord notifications working"
                },
                "timestamp": datetime.now().isoformat()
            }
            
            discord_data = {
                "embeds": [embed],
                "username": "Crypto Signal Bot"
            }
            
            response = requests.post(
                self.webhook_url,
                json=discord_data,
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            
            if response.status_code == 204:
                print("✅ Daily summary sent to Discord")
                return True
            else:
                print(f"❌ Daily summary failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Daily summary error: {e}")
            return False

    def format_signal_message(self, signal_data: Dict) -> str:
        """Format signal data into Discord message using exact template"""
        try:
            # Extract signal data
            symbol = signal_data.get('symbol', 'UNKNOWN')
            side = signal_data.get('signal_type', 'LONG')
            entry_price = signal_data.get('entry_price', 0)
            stop_loss = signal_data.get('stop_loss', 0)
            tp1 = signal_data.get('tp1_price', 0)
            tp2 = signal_data.get('tp2_price', 0)
            tp3 = signal_data.get('tp3_price', 0)
            risk_reward = signal_data.get('risk_reward', 0)
            strength_dot = signal_data.get('strength_dot', 3)
            confidence_layers = signal_data.get('confidence_layers', 5)
            timeframe = signal_data.get('timeframe', '15m')
            
            # Get technical data for indicator status
            technical_data = signal_data.get('technical_data', {})
            stage3_data = signal_data.get('stage3_data', {})
            
            # Determine indicator status
            ema_status = "✅" if technical_data.get('ema_crossover', False) else "❌"
            rsi_status = "✅" if technical_data.get('rsi_reversal', False) else "❌"
            macd_status = "✅" if technical_data.get('macd_bullish', False) else "❌"
            volume_status = "✅" if stage3_data.get('vol_spike', False) else "❌"
            adx_status = "✅" if technical_data.get('adx_value', 0) > 20 else "❌"
            vol_rank_status = "✅" if technical_data.get('volatility_rank', 0) > 50 else "❌"
            
            # Format strength and confidence as numeric labels
            strength_label = f"{strength_dot}/7"
            confidence_label = f"{confidence_layers}/7"
            
            # Format prices with decimal-precision formatting
            def fmt(x, symbol):
                # use tick-size or default to:
                decimals = 2 if symbol.endswith('USDT') else 4
                return f"${x:.{decimals}f}"
            
            # Format the message using exact template
            message = f"""📊 {side} Signal Alert

**Coin:** {symbol}
**Entry:** {fmt(entry_price, symbol)}
**Stop-Loss:** {fmt(stop_loss, symbol)}
**TP1:** {fmt(tp1, symbol)}
**TP2:** {fmt(tp2, symbol)}
**TP3:** {fmt(tp3, symbol)}
**Risk-Reward:** {risk_reward:.2f}
**Strength:** {strength_label}
**Confidence:** {confidence_label}
**Market Regime:** {signal_data.get('market_regime', 'NORMAL')}
**Indicators:** EMA {ema_status} | RSI {rsi_status} | MACD {macd_status} | Volume {volume_status} | ADX {adx_status} | VolRank {vol_rank_status}
**Sentiment:** Coinglass ✅ | TradingView ✅
**Timeframe:** {timeframe}
**Market:** Binance Futures"""
            
            return message
            
        except Exception as e:
            self.logger.error(f"Error formatting signal message: {e}")
            return "Error formatting signal message"

class TelegramNotifier:
    def __init__(self, bot_token: str = None, main_chat_id: str = None, profit_chat_id: str = None):
        """
        Initialize Telegram Notifier
        Args:
            bot_token: Telegram bot token
            main_chat_id: Main chat ID for signals
            profit_chat_id: Profit chat ID for TP/SL hits
        """
        self.logger = logging.getLogger(__name__)
        self.bot_token = bot_token or "YOUR_BOT_TOKEN"
        self.main_chat_id = main_chat_id or "MAIN_CHAT_ID"
        self.profit_chat_id = profit_chat_id or "PROFIT_CHAT_ID"
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"
        
        self.logger.info("Telegram Notifier initialized")
    
    def send_signal_notification(self, signal_data: Dict) -> bool:
        """Send signal notification to MAIN_CHAT_ID"""
        try:
            if not self.bot_token or self.bot_token == "YOUR_BOT_TOKEN":
                self.logger.warning("Telegram bot token not configured")
                return False
            
            # Format the signal message
            message = self.format_signal_message(signal_data)
            
            # Send to main chat
            url = f"{self.base_url}/sendMessage"
            payload = {
                "chat_id": self.main_chat_id,
                "text": message,
                "parse_mode": "HTML"
            }
            
            response = requests.post(url, json=payload, timeout=10)
            
            if response.status_code == 200:
                self.logger.info(f"Telegram signal notification sent successfully for {signal_data.get('symbol', 'Unknown')}")
                return True
            else:
                self.logger.error(f"Telegram notification failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error sending Telegram notification: {e}")
            return False
    
    def send_tp_hit_notification(self, symbol: str, entry_price: float, tp_level: str, tp_price: float) -> bool:
        """Send TP hit notification to PROFIT_CHAT_ID"""
        try:
            if not self.bot_token or self.bot_token == "YOUR_BOT_TOKEN":
                self.logger.warning("Telegram bot token not configured")
                return False
            
            # Calculate unrealized gain
            gain_pct = ((tp_price - entry_price) / entry_price) * 100
            
            # Format TP hit message
            message = f"""🎯 {tp_level} HIT – {symbol}

Entry: ${entry_price:,.2f} → {tp_level}: ${tp_price:,.2f}
Unrealized Gain: +{gain_pct:.2f}%"""
            
            # Send to profit chat
            url = f"{self.base_url}/sendMessage"
            payload = {
                "chat_id": self.profit_chat_id,
                "text": message,
                "parse_mode": "HTML"
            }
            
            response = requests.post(url, json=payload, timeout=10)
            
            if response.status_code == 200:
                self.logger.info(f"Telegram TP hit notification sent for {symbol} - {tp_level}")
                return True
            else:
                self.logger.error(f"Telegram TP notification failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error sending Telegram TP notification: {e}")
            return False
    
    def send_sl_hit_notification(self, symbol: str, entry_price: float, sl_price: float) -> bool:
        """Send SL hit notification to PROFIT_CHAT_ID"""
        try:
            if not self.bot_token or self.bot_token == "YOUR_BOT_TOKEN":
                self.logger.warning("Telegram bot token not configured")
                return False
            
            # Calculate unrealized loss
            loss_pct = ((sl_price - entry_price) / entry_price) * 100
            
            # Format SL hit message
            message = f"""🛑 STOP LOSS HIT – {symbol}

Entry: ${entry_price:,.2f} → SL: ${sl_price:,.2f}
Unrealized Loss: {loss_pct:.2f}%"""
            
            # Send to profit chat
            url = f"{self.base_url}/sendMessage"
            payload = {
                "chat_id": self.profit_chat_id,
                "text": message,
                "parse_mode": "HTML"
            }
            
            response = requests.post(url, json=payload, timeout=10)
            
            if response.status_code == 200:
                self.logger.info(f"Telegram SL hit notification sent for {symbol}")
                return True
            else:
                self.logger.error(f"Telegram SL notification failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error sending Telegram SL notification: {e}")
            return False
    
    def format_signal_message(self, signal_data: Dict) -> str:
        """Format signal data into Telegram message"""
        try:
            # Extract signal data
            symbol = signal_data.get('symbol', 'UNKNOWN')
            side = signal_data.get('signal_type', 'LONG')
            entry_price = signal_data.get('entry_price', 0)
            stop_loss = signal_data.get('stop_loss', 0)
            tp1 = signal_data.get('tp1_price', 0)
            tp2 = signal_data.get('tp2_price', 0)
            tp3 = signal_data.get('tp3_price', 0)
            risk_reward = signal_data.get('risk_reward', 0)
            strength_dot = signal_data.get('strength_dot', 3)
            confidence_layers = signal_data.get('confidence_layers', 5)
            timeframe = signal_data.get('timeframe', '15m')
            
            # Get technical data for indicator status
            technical_data = signal_data.get('technical_data', {})
            stage3_data = signal_data.get('stage3_data', {})
            
            # Determine indicator status
            ema_status = "✅" if technical_data.get('ema_crossover', False) else "❌"
            rsi_status = "✅" if technical_data.get('rsi_reversal', False) else "❌"
            macd_status = "✅" if technical_data.get('macd_bullish', False) else "❌"
            volume_status = "✅" if stage3_data.get('vol_spike', False) else "❌"
            adx_status = "✅" if technical_data.get('adx_value', 0) > 20 else "❌"
            vol_rank_status = "✅" if technical_data.get('volatility_rank', 0) > 50 else "❌"
            
            # Format strength and confidence as numeric labels
            strength_label = f"{strength_dot}/7"
            confidence_label = f"{confidence_layers}/7"
            
            # Format prices with decimal-precision formatting
            def fmt(x, symbol):
                # use tick-size or default to:
                decimals = 2 if symbol.endswith('USDT') else 4
                return f"${x:.{decimals}f}"
            
            # Format the message for Telegram
            message = f"""📊 <b>{side} Signal Alert</b>

<b>Coin:</b> {symbol}
<b>Entry:</b> {fmt(entry_price, symbol)}
<b>Stop-Loss:</b> {fmt(stop_loss, symbol)}
<b>TP1:</b> {fmt(tp1, symbol)}
<b>TP2:</b> {fmt(tp2, symbol)}
<b>TP3:</b> {fmt(tp3, symbol)}
<b>Risk-Reward:</b> {risk_reward:.2f}
<b>Strength:</b> {strength_label}
<b>Confidence:</b> {confidence_label}
<b>Market Regime:</b> {signal_data.get('market_regime', 'NORMAL')}
<b>Indicators:</b> EMA {ema_status} | RSI {rsi_status} | MACD {macd_status} | Volume {volume_status} | ADX {adx_status} | VolRank {vol_rank_status}
<b>Sentiment:</b> Coinglass ✅ | TradingView ✅
<b>Timeframe:</b> {timeframe}
<b>Market:</b> Binance Futures"""
            
            return message
            
        except Exception as e:
            self.logger.error(f"Error formatting Telegram signal message: {e}")
            return "Error formatting signal message"

def test_discord_connection():
    """Test Discord webhook connection"""
    print("🎮 Testing Discord Webhook Connection")
    print("=" * 40)
    
    # Test signal data
    test_signal = {
        'symbol': 'BTC/USDT',
        'strategy': 'Test Strategy',
        'entry_price': 45000.0,
        'stop_loss': 44000.0,
        'tp1_price': 46000.0,
        'tp2_price': 47000.0,
        'tp3_price': 48000.0,
        'risk_reward': 1.5,
        'confidence': 4.5
    }
    
    notifier = DiscordNotifier()
    
    print("📤 Sending test Discord notification...")
    if notifier.send_signal_notification(test_signal):
        print("✅ Discord test successful!")
    else:
        print("❌ Discord test failed")
        print("\n💡 To set up Discord webhooks:")
        print("1. Create a Discord server")
        print("2. Go to Server Settings > Integrations > Webhooks")
        print("3. Create a new webhook")
        print("4. Copy the webhook URL")
        print("5. Update the webhook_url in discord_notifier.py")

if __name__ == "__main__":
    test_discord_connection() 