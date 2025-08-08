# PAPA - Enhanced Crypto Signal Bot

[![GitHub](https://img.shields.io/badge/GitHub-PAPA-blue.svg)](https://github.com/raaabdullah1/PAPA)
[![Python](https://img.shields.io/badge/Python-3.8+-green.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🚀 PAPA - Professional Advanced Predictive Algorithm

**PAPA** is an advanced cryptocurrency signal bot that combines multiple technical analysis techniques with intelligent filtering to generate high-quality trading signals. Built with orderblock detection, volatility-adaptive filters, and 7-layer confidence scoring.

## ✨ Key Features

### 🔍 **Orderblock Detection**
- Identifies bullish and bearish orderblocks across multiple timeframes
- Polarity matching for signal alignment
- Strength scoring and confidence bonuses

### 📊 **Volatility-Adaptive EMA Filters**
- Dynamic EMA distance requirements based on ATR(14)
- Scales automatically with market volatility
- Prevents false signals in different market conditions

### ⭐ **7-Layer Confidence Scoring**
- Extended confidence system (0-7 layers)
- Star rating system (★1-★5) for visual quality assessment
- EMA distance and orderblock alignment bonuses

### 🎯 **Intelligent Signal Ranking**
- Quality-based signal selection
- Configurable signal caps per scan
- Confluence scoring across multiple timeframes

### 🔄 **Confluence Pipeline**
- Multi-timeframe analysis (15m, 30m, 1h, 4h)
- Weighted confluence scoring
- Two-stage filtering for efficiency

## 🏗️ Architecture

```
PAPA/
├── send_real_signals.py          # Main confluence pipeline
├── core/                         # Core modules
│   ├── filter_tree.py           # Enhanced 7-layer confidence
│   ├── orderblock_detector.py   # Orderblock detection
│   ├── scanner.py               # Market data fetching
│   ├── notifier.py              # Discord notifications
│   └── ...                      # Other core modules
├── ui/                          # Dashboard components
├── test_*.py                    # Comprehensive test suite
└── requirements.txt             # Dependencies
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Binance API credentials
- Discord webhook URL

### Installation
```bash
git clone https://github.com/raaabdullah1/PAPA.git
cd PAPA
pip install -r requirements.txt
```

### Configuration
1. Copy `secrets.py.example` to `secrets.py`
2. Add your API credentials:
```python
BINANCE_API_KEY = "your_api_key"
BINANCE_SECRET_KEY = "your_secret_key"
DISCORD_WEBHOOK_URL = "your_webhook_url"
```

### Running PAPA
```bash
# Production mode (all symbols)
python3 send_real_signals.py

# Test mode (limited symbols)
python3 test_production.py
```

## 📈 Signal Quality

### Star Rating System
- **★1**: Low confidence (0-1 layers)
- **★2**: Below average (2-3 layers)
- **★3**: Average (4 layers)
- **★4**: Above average (5 layers)
- **★5**: High confidence (6-7 layers)

### Example Signal
```
🟢 **BTC/USDT LONG** (15m) ★4
💰 **Price:** $116840.4
🛑 **Stop Loss:** $115000.0
🎯 **Take Profit:** $120000.0
⚖️ **Risk/Reward:** 1.25
🎯 **Confidence:** 5/7 | **Confluence:** 100.0% | Orderblock ✔ (bullish, strength: 0.52)
⏰ **Time:** 2025-08-08 12:37:15 PST
```

## 🧪 Testing

Run the comprehensive test suite:
```bash
# Test confidence and quality filters
python3 test_confidence_quality_filters.py

# Test volatility-adaptive EMA filters
python3 test_volatility_adaptive_ema.py

# Test orderblock detection
python3 test_orderblock_detection.py

# Test production pipeline
python3 test_production.py
```

## ⚙️ Configuration

### Main Settings (`send_real_signals.py`)
```python
CONFIG = {
    'stage1_timeframe': '15m',      # Stage 1 filtering timeframe
    'MAX_SIGNALS_PER_SCAN': 5,      # Maximum signals per scan
    'MIN_RISK_REWARD': 1.15,        # Minimum risk-reward ratio
    'MIN_CONFLUENCE': 50,           # Minimum confluence percentage
    'PRODUCTION_MODE': True,        # Full production mode
}
```

## 🔧 Core Components

### Filter Tree (`core/filter_tree.py`)
- **Stage 1**: Liquidity and volume filtering
- **Stage 2**: Technical indicators with volatility-adaptive EMA
- **Stage 3**: Strategy validation with 7-layer confidence

### Orderblock Detector (`core/orderblock_detector.py`)
- Identifies market structure orderblocks
- Polarity matching for signal alignment
- Strength scoring and confidence bonuses

### Scanner (`core/scanner.py`)
- Efficient market data fetching
- OHLCV resampling for multiple timeframes
- Robust error handling and rate limiting

## 📊 Performance Metrics

- **Signal Quality**: 7-layer confidence scoring
- **Confluence Analysis**: Multi-timeframe validation
- **Orderblock Alignment**: Market structure confirmation
- **Volatility Adaptation**: Dynamic filter requirements

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new features
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Orderblock Detection**: Advanced market structure analysis
- **Volatility-Adaptive Filters**: Dynamic technical analysis
- **Confluence Pipeline**: Multi-timeframe signal validation
- **7-Layer Confidence**: Enhanced signal quality assessment

---

**PAPA** - Professional Advanced Predictive Algorithm  
*Building the future of algorithmic trading* 🚀 