import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import requests
import json
import time
from typing import Dict, List

# Page configuration
st.set_page_config(
    page_title="Crypto Signal Bot Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .signal-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        margin-bottom: 1rem;
    }
    .positive { color: #28a745; }
    .negative { color: #dc3545; }
    .neutral { color: #6c757d; }
</style>
""", unsafe_allow_html=True)

class Dashboard:
    def __init__(self):
        self.api_base_url = "http://localhost:5000"
        
    def get_api_data(self, endpoint: str) -> Dict:
        """Get data from Flask API"""
        try:
            response = requests.get(f"{self.api_base_url}/{endpoint}", timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"Error fetching data from API: {e}")
            return {}
    
    def get_bot_status(self) -> Dict:
        """Get bot status"""
        return self.get_api_data("status")
    
    def get_health(self) -> Dict:
        """Get health status"""
        return self.get_api_data("health")
    
    def get_debug_info(self) -> Dict:
        """Get debug information"""
        return self.get_api_data("debug")
    
    def get_signals(self) -> List[Dict]:
        """Get recent signals"""
        data = self.get_api_data("signals")
        return data.get('signals', [])
    
    def create_price_chart(self, symbol: str, timeframe: str = "1h") -> go.Figure:
        """Create price chart for a symbol"""
        # This would integrate with your data source
        # For now, create a mock chart
        dates = pd.date_range(start=datetime.now() - timedelta(days=7), periods=168, freq='H')
        prices = [100 + i * 0.1 + (i % 10) * 0.5 for i in range(168)]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates,
            y=prices,
            mode='lines',
            name=symbol,
            line=dict(color='#1f77b4', width=2)
        ))
        
        fig.update_layout(
            title=f"{symbol} Price Chart ({timeframe})",
            xaxis_title="Time",
            yaxis_title="Price (USDT)",
            height=400,
            showlegend=False
        )
        
        return fig
    
    def create_liquidation_heatmap(self) -> go.Figure:
        """Create liquidation heatmap"""
        # Mock data for liquidation heatmap
        symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'SOL/USDT']
        hours = list(range(24))
        
        # Mock liquidation data
        liquidation_data = []
        for symbol in symbols:
            for hour in hours:
                liquidation_data.append({
                    'symbol': symbol,
                    'hour': hour,
                    'liquidations': abs(hash(f"{symbol}{hour}") % 100)
                })
        
        df = pd.DataFrame(liquidation_data)
        pivot_df = df.pivot(index='symbol', columns='hour', values='liquidations')
        
        fig = go.Figure(data=go.Heatmap(
            z=pivot_df.values,
            x=pivot_df.columns,
            y=pivot_df.index,
            colorscale='Reds',
            showscale=True
        ))
        
        fig.update_layout(
            title="Liquidation Heatmap (24h)",
            xaxis_title="Hour",
            yaxis_title="Symbol",
            height=400
        )
        
        return fig
    
    def create_portfolio_exposure_chart(self, portfolio_data: Dict) -> go.Figure:
        """Create portfolio exposure chart"""
        # Mock portfolio data
        symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'SOL/USDT']
        exposures = [30, 25, 20, 15, 10]
        
        fig = go.Figure(data=[go.Pie(
            labels=symbols,
            values=exposures,
            hole=0.3
        )])
        
        fig.update_layout(
            title="Portfolio Exposure by Symbol",
            height=400
        )
        
        return fig
    
    def create_strategy_performance_chart(self, signals: List[Dict]) -> go.Figure:
        """Create strategy performance chart"""
        if not signals:
            # Return empty chart
            fig = go.Figure()
            fig.update_layout(
                title="Strategy Performance",
                xaxis_title="Strategy",
                yaxis_title="Success Rate (%)",
                height=400
            )
            return fig
        
        # Process signals to get strategy performance
        strategy_data = {}
        for signal in signals:
            strategy = signal.get('signal_data', {}).get('strategy', 'Unknown')
            if strategy not in strategy_data:
                strategy_data[strategy] = {'total': 0, 'successful': 0}
            
            strategy_data[strategy]['total'] += 1
            # Mock success rate
            if hash(str(signal)) % 3 == 0:  # 33% success rate
                strategy_data[strategy]['successful'] += 1
        
        strategies = list(strategy_data.keys())
        success_rates = [
            (strategy_data[s]['successful'] / strategy_data[s]['total'] * 100) 
            if strategy_data[s]['total'] > 0 else 0 
            for s in strategies
        ]
        
        fig = go.Figure(data=[go.Bar(
            x=strategies,
            y=success_rates,
            marker_color=['#28a745', '#ffc107', '#dc3545']
        )])
        
        fig.update_layout(
            title="Strategy Performance (Success Rate)",
            xaxis_title="Strategy",
            yaxis_title="Success Rate (%)",
            height=400
        )
        
        return fig
    
    def run(self):
        """Run the dashboard"""
        st.markdown('<h1 class="main-header">🤖 Crypto Signal Bot Dashboard</h1>', unsafe_allow_html=True)
        
        # Sidebar
        st.sidebar.title("📊 Dashboard Controls")
        
        # Auto-refresh
        auto_refresh = st.sidebar.checkbox("Auto-refresh", value=True)
        refresh_interval = st.sidebar.slider("Refresh interval (seconds)", 5, 60, 30)
        
        if auto_refresh:
            time.sleep(refresh_interval)
            st.experimental_rerun()
        
        # Manual refresh button
        if st.sidebar.button("🔄 Refresh Now"):
            st.experimental_rerun()
        
        # Get data
        health_data = self.get_health()
        status_data = self.get_bot_status()
        debug_data = self.get_debug_info()
        signals = self.get_signals()
        
        # Health status
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            health_status = health_data.get('status', 'unknown')
            status_color = '🟢' if health_status == 'healthy' else '🔴'
            st.metric(
                label="System Health",
                value=f"{status_color} {health_status.title()}",
                delta=None
            )
        
        with col2:
            bot_running = status_data.get('bot_status', {}).get('running', False)
            running_status = '🟢 Running' if bot_running else '🔴 Stopped'
            st.metric(
                label="Bot Status",
                value=running_status,
                delta=None
            )
        
        with col3:
            total_signals = status_data.get('bot_status', {}).get('total_signals', 0)
            st.metric(
                label="Total Signals",
                value=total_signals,
                delta=None
            )
        
        with col4:
            daily_signals = status_data.get('bot_status', {}).get('daily_signals', 0)
            max_daily = status_data.get('bot_status', {}).get('max_daily_signals', 20)
            st.metric(
                label="Daily Signals",
                value=f"{daily_signals}/{max_daily}",
                delta=None
            )
        
        # Main content tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Live Signals", 
            "📊 Charts & Analysis", 
            "💼 Portfolio", 
            "📋 Logs", 
            "⚙️ System"
        ])
        
        with tab1:
            st.header("📈 Live Signals Feed")
            
            if signals:
                for signal in signals[-10:]:  # Show last 10 signals
                    signal_data = signal.get('signal_data', {})
                    
                    with st.container():
                        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
                        
                        with col1:
                            symbol = signal_data.get('symbol', 'Unknown')
                            strategy = signal_data.get('strategy', 'Unknown')
                            confidence = signal_data.get('confidence', 0)
                            
                            st.markdown(f"""
                            <div class="signal-card">
                                <h4>{symbol} - {strategy}</h4>
                                <p><strong>Confidence:</strong> {confidence:.1f}/5</p>
                                <p><strong>Entry:</strong> ${signal_data.get('entry_price', 0):.4f}</p>
                                <p><strong>Stop Loss:</strong> ${signal_data.get('stop_loss', 0):.4f}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            tp1 = signal_data.get('tp1_price', 0)
                            st.metric("TP1", f"${tp1:.4f}")
                        
                        with col3:
                            tp2 = signal_data.get('tp2_price', 0)
                            st.metric("TP2", f"${tp2:.4f}")
                        
                        with col4:
                            tp3 = signal_data.get('tp3_price', 0)
                            st.metric("TP3", f"${tp3:.4f}")
            else:
                st.info("No signals available yet. The bot will generate signals when market conditions are met.")
        
        with tab2:
            st.header("📊 Charts & Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Price chart
                symbol_selector = st.selectbox(
                    "Select Symbol",
                    ["BTC/USDT", "ETH/USDT", "BNB/USDT", "ADA/USDT", "SOL/USDT"]
                )
                price_chart = self.create_price_chart(symbol_selector)
                st.plotly_chart(price_chart, use_container_width=True)
                
                # Strategy performance
                strategy_chart = self.create_strategy_performance_chart(signals)
                st.plotly_chart(strategy_chart, use_container_width=True)
            
            with col2:
                # Liquidation heatmap
                liquidation_chart = self.create_liquidation_heatmap()
                st.plotly_chart(liquidation_chart, use_container_width=True)
                
                # Portfolio exposure
                portfolio_chart = self.create_portfolio_exposure_chart({})
                st.plotly_chart(portfolio_chart, use_container_width=True)
        
        with tab3:
            st.header("💼 Portfolio Management")
            
            portfolio_status = status_data.get('portfolio_status', {})
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Portfolio Summary")
                
                account_balance = portfolio_status.get('account_balance', 0)
                daily_pnl = portfolio_status.get('daily_pnl', 0)
                open_positions = portfolio_status.get('open_positions_count', 0)
                closed_positions = portfolio_status.get('closed_positions_count', 0)
                
                st.metric("Account Balance", f"${account_balance:,.2f}")
                st.metric("Daily P&L", f"${daily_pnl:,.2f}", 
                         delta=f"{daily_pnl/account_balance*100:.2f}%" if account_balance > 0 else None)
                st.metric("Open Positions", open_positions)
                st.metric("Closed Positions", closed_positions)
            
            with col2:
                st.subheader("Risk Management")
                
                max_daily_loss = portfolio_status.get('max_daily_loss_reached', False)
                max_positions = portfolio_status.get('max_positions_reached', False)
                
                risk_status = "🟢 Normal" if not max_daily_loss else "🔴 Daily Loss Limit Reached"
                position_status = "🟢 Normal" if not max_positions else "🔴 Max Positions Reached"
                
                st.metric("Daily Loss Status", risk_status)
                st.metric("Position Limit Status", position_status)
        
        with tab4:
            st.header("📋 System Logs")
            
            # Log viewer
            log_file = st.selectbox(
                "Select Log File",
                ["bot.log", "signals.json", "Daily Summary"]
            )
            
            if log_file == "bot.log":
                try:
                    with open("logs/bot.log", "r") as f:
                        logs = f.read()
                    st.text_area("Bot Logs", logs, height=400)
                except FileNotFoundError:
                    st.warning("Log file not found")
            
            elif log_file == "signals.json":
                try:
                    with open("logs/signals.json", "r") as f:
                        signals_log = json.load(f)
                    st.json(signals_log)
                except FileNotFoundError:
                    st.warning("Signals log file not found")
            
            else:
                # Daily summary
                if signals:
                    df = pd.DataFrame([
                        {
                            'Date': signal.get('timestamp', ''),
                            'Symbol': signal.get('signal_data', {}).get('symbol', ''),
                            'Strategy': signal.get('signal_data', {}).get('strategy', ''),
                            'Confidence': signal.get('signal_data', {}).get('confidence', 0)
                        }
                        for signal in signals
                    ])
                    st.dataframe(df)
        
        with tab5:
            st.header("⚙️ System Information")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Bot Status")
                
                uptime = status_data.get('bot_status', {}).get('uptime', 'N/A')
                total_scans = status_data.get('bot_status', {}).get('total_scans', 0)
                start_time = status_data.get('bot_status', {}).get('start_time', 'N/A')
                
                st.metric("Uptime", uptime)
                st.metric("Total Scans", total_scans)
                st.metric("Start Time", start_time)
            
            with col2:
                st.subheader("API Connections")
                
                api_connections = debug_data.get('api_connections', {})
                binance_status = "🟢 Connected" if api_connections.get('binance', False) else "🔴 Disconnected"
                telegram_status = "🟢 Connected" if api_connections.get('telegram', False) else "🔴 Disconnected"
                
                st.metric("Binance API", binance_status)
                st.metric("Telegram Bot", telegram_status)
            
            # System info
            st.subheader("System Information")
            system_info = debug_data.get('system_info', {})
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Python Version", system_info.get('python_version', 'N/A'))
            with col2:
                st.metric("Memory Usage", system_info.get('memory_usage', 'N/A'))
            with col3:
                st.metric("Disk Usage", system_info.get('disk_usage', 'N/A'))

def main():
    """Main function"""
    dashboard = Dashboard()
    dashboard.run()

if __name__ == "__main__":
    main() 