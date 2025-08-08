import json
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any

class SimpleLogger:
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        # Use Pakistan Standard Time (UTC+5)
        self.pst_timezone = timezone(timedelta(hours=5))
        self.log_file = os.path.join(log_dir, "signals.json")
        self.daily_log_file = os.path.join(log_dir, f"signals_{datetime.now(self.pst_timezone).strftime('%Y%m%d')}.json")
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self._setup_logging()
        
        # Initialize log files
        self._initialize_log_files()
    
    def _setup_logging(self):
        """Setup logging configuration with Pakistan Standard Time"""
        # Custom formatter with Pakistan timezone
        class PakistanTimeFormatter(logging.Formatter):
            def formatTime(self, record, datefmt=None):
                dt = datetime.fromtimestamp(record.created, self.pst_timezone)
                if datefmt:
                    return dt.strftime(datefmt)
                else:
                    return dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Create formatter with Pakistan timezone
        formatter = PakistanTimeFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Setup handlers
        file_handler = logging.FileHandler(os.path.join(self.log_dir, 'bot.log'))
        file_handler.setFormatter(formatter)
        
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        
        # Configure root logger
        logging.basicConfig(
            level=logging.INFO,
            handlers=[file_handler, stream_handler]
        )
    
    def _initialize_log_files(self):
        """Initialize log files if they don't exist"""
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w') as f:
                json.dump([], f)
        
        if not os.path.exists(self.daily_log_file):
            with open(self.daily_log_file, 'w') as f:
                json.dump([], f)
    
    def log_signal(self, signal: Dict, additional_metadata: Dict = None) -> bool:
        """Log a trading signal with all metadata"""
        try:
            # Create log entry
            log_entry = {
                'timestamp': datetime.now(self.pst_timezone).isoformat(),
                'signal_id': self._generate_signal_id(),
                'signal_data': signal.copy(),
                'metadata': additional_metadata or {},
                'log_version': '1.0'
            }
            
            # Add signal to main log file
            self._append_to_log_file(self.log_file, log_entry)
            
            # Add signal to daily log file
            self._append_to_log_file(self.daily_log_file, log_entry)
            
            # Log to console
            self.logger.info(f"Signal logged: {signal.get('symbol', 'Unknown')} - {signal.get('strategy', 'Unknown')} - Confidence: {signal.get('confidence', 0):.2f}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error logging signal: {e}")
            return False
    
    def log_trade_execution(self, signal_id: str, execution_data: Dict) -> bool:
        """Log trade execution details"""
        try:
            log_entry = {
                'timestamp': datetime.now(self.pst_timezone).isoformat(),
                'signal_id': signal_id,
                'execution_data': execution_data,
                'type': 'trade_execution'
            }
            
            self._append_to_log_file(self.log_file, log_entry)
            self._append_to_log_file(self.daily_log_file, log_entry)
            
            self.logger.info(f"Trade execution logged for signal {signal_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error logging trade execution: {e}")
            return False
    
    def log_trade_exit(self, signal_id: str, exit_data: Dict) -> bool:
        """Log trade exit details"""
        try:
            log_entry = {
                'timestamp': datetime.now(self.pst_timezone).isoformat(),
                'signal_id': signal_id,
                'exit_data': exit_data,
                'type': 'trade_exit'
            }
            
            self._append_to_log_file(self.log_file, log_entry)
            self._append_to_log_file(self.daily_log_file, log_entry)
            
            self.logger.info(f"Trade exit logged for signal {signal_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error logging trade exit: {e}")
            return False
    
    def log_error(self, error_data: Dict) -> bool:
        """Log error information"""
        try:
            log_entry = {
                'timestamp': datetime.now(self.pst_timezone).isoformat(),
                'error_data': error_data,
                'type': 'error'
            }
            
            self._append_to_log_file(self.log_file, log_entry)
            self._append_to_log_file(self.daily_log_file, log_entry)
            
            self.logger.error(f"Error logged: {error_data.get('message', 'Unknown error')}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error logging error: {e}")
            return False
    
    def log_system_status(self, status_data: Dict) -> bool:
        """Log system status information"""
        try:
            log_entry = {
                'timestamp': datetime.now(self.pst_timezone).isoformat(),
                'status_data': status_data,
                'type': 'system_status'
            }
            
            self._append_to_log_file(self.log_file, log_entry)
            self._append_to_log_file(self.daily_log_file, log_entry)
            
            self.logger.info(f"System status logged: {status_data.get('status', 'Unknown')}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error logging system status: {e}")
            return False
    
    def _append_to_log_file(self, file_path: str, log_entry: Dict):
        """Append log entry to file atomically"""
        try:
            # Read existing logs
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    logs = json.load(f)
            else:
                logs = []
            
            # Add new log entry
            logs.append(log_entry)
            
            # Write back to file
            with open(file_path, 'w') as f:
                json.dump(logs, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error appending to log file: {e}")
    
    def _generate_signal_id(self) -> str:
        """Generate unique signal ID"""
        return f"signal_{datetime.now(self.pst_timezone).strftime('%Y%m%d_%H%M%S_%f')}"
    
    def get_signals_by_date(self, date: str = None) -> List[Dict]:
        """Get signals for a specific date"""
        try:
            if date is None:
                date = datetime.now(self.pst_timezone).strftime('%Y%m%d')
            
            daily_file = os.path.join(self.log_dir, f"signals_{date}.json")
            
            if os.path.exists(daily_file):
                with open(daily_file, 'r') as f:
                    logs = json.load(f)
                
                # Filter signals only
                signals = [log for log in logs if log.get('signal_data')]
                return signals
            else:
                return []
                
        except Exception as e:
            self.logger.error(f"Error getting signals by date: {e}")
            return []
    
    def get_signals_by_symbol(self, symbol: str, days: int = 7) -> List[Dict]:
        """Get signals for a specific symbol in the last N days"""
        try:
            signals = []
            for i in range(days):
                date = (datetime.now(self.pst_timezone) - timedelta(days=i)).strftime('%Y%m%d')
                daily_signals = self.get_signals_by_date(date)
                
                for signal in daily_signals:
                    if signal.get('signal_data', {}).get('symbol') == symbol:
                        signals.append(signal)
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error getting signals by symbol: {e}")
            return []
    
    def get_signals_by_strategy(self, strategy: str, days: int = 7) -> List[Dict]:
        """Get signals for a specific strategy in the last N days"""
        try:
            signals = []
            for i in range(days):
                date = (datetime.now(self.pst_timezone) - timedelta(days=i)).strftime('%Y%m%d')
                daily_signals = self.get_signals_by_date(date)
                
                for signal in daily_signals:
                    if signal.get('signal_data', {}).get('strategy') == strategy:
                        signals.append(signal)
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error getting signals by strategy: {e}")
            return []
    
    def get_signal_statistics(self, days: int = 7) -> Dict:
        """Get signal statistics for the last N days"""
        try:
            total_signals = 0
            symbols_count = {}
            strategies_count = {}
            confidence_sum = 0
            successful_signals = 0
            
            for i in range(days):
                date = (datetime.now(self.pst_timezone) - timedelta(days=i)).strftime('%Y%m%d')
                daily_signals = self.get_signals_by_date(date)
                
                for signal in daily_signals:
                    signal_data = signal.get('signal_data', {})
                    total_signals += 1
                    
                    # Count symbols
                    symbol = signal_data.get('symbol', 'Unknown')
                    symbols_count[symbol] = symbols_count.get(symbol, 0) + 1
                    
                    # Count strategies
                    strategy = signal_data.get('strategy', 'Unknown')
                    strategies_count[strategy] = strategies_count.get(strategy, 0) + 1
                    
                    # Sum confidence
                    confidence = signal_data.get('confidence', 0)
                    confidence_sum += confidence
                    
                    # Count successful signals (if available)
                    if signal_data.get('success', False):
                        successful_signals += 1
            
            avg_confidence = confidence_sum / total_signals if total_signals > 0 else 0
            success_rate = (successful_signals / total_signals * 100) if total_signals > 0 else 0
            
            return {
                'total_signals': total_signals,
                'avg_confidence': avg_confidence,
                'success_rate': success_rate,
                'top_symbols': dict(sorted(symbols_count.items(), key=lambda x: x[1], reverse=True)[:10]),
                'top_strategies': dict(sorted(strategies_count.items(), key=lambda x: x[1], reverse=True)[:5])
            }
            
        except Exception as e:
            self.logger.error(f"Error getting signal statistics: {e}")
            return {}
    
    def cleanup_old_logs(self, days_to_keep: int = 30) -> bool:
        """Clean up old log files"""
        try:
            cutoff_date = datetime.now(self.pst_timezone) - timedelta(days=days_to_keep)
            
            for filename in os.listdir(self.log_dir):
                if filename.startswith('signals_') and filename.endswith('.json'):
                    try:
                        date_str = filename.replace('signals_', '').replace('.json', '')
                        file_date = datetime.strptime(date_str, '%Y%m%d')
                        
                        if file_date < cutoff_date:
                            file_path = os.path.join(self.log_dir, filename)
                            os.remove(file_path)
                            self.logger.info(f"Removed old log file: {filename}")
                    except Exception as e:
                        self.logger.warning(f"Error processing log file {filename}: {e}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error cleaning up old logs: {e}")
            return False 