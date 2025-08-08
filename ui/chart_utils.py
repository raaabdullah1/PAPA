#!/usr/bin/env python3
"""
Chart Utilities for Crypto Signal Bot Dashboard
Provides chart generation and visualization functions
"""

import logging
from typing import Dict, List, Optional, Tuple
import json
from datetime import datetime, timedelta

class ChartUtils:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def generate_signal_chart_data(self, signals: List[Dict]) -> Dict:
        """
        Generate chart data for signal visualization
        """
        try:
            if not signals:
                return {'error': 'No signals available'}
            
            # Prepare data for different chart types
            chart_data = {
                'timeline': self._prepare_timeline_data(signals),
                'performance': self._prepare_performance_data(signals),
                'strategy_distribution': self._prepare_strategy_data(signals),
                'confidence_distribution': self._prepare_confidence_data(signals)
            }
            
            return chart_data
            
        except Exception as e:
            self.logger.error(f"Error generating signal chart data: {e}")
            return {'error': str(e)}
    
    def _prepare_timeline_data(self, signals: List[Dict]) -> Dict:
        """
        Prepare timeline chart data
        """
        try:
            timeline_data = {
                'labels': [],
                'datasets': [
                    {
                        'label': 'Signal Confidence',
                        'data': [],
                        'borderColor': 'rgb(75, 192, 192)',
                        'backgroundColor': 'rgba(75, 192, 192, 0.2)',
                        'tension': 0.1
                    }
                ]
            }
            
            for signal in signals:
                timestamp = signal.get('timestamp', '')
                confidence = signal.get('confidence', 0)
                
                # Format timestamp for display
                if isinstance(timestamp, str):
                    try:
                        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                        formatted_time = dt.strftime('%H:%M')
                    except:
                        formatted_time = timestamp
                else:
                    formatted_time = str(timestamp)
                
                timeline_data['labels'].append(formatted_time)
                timeline_data['datasets'][0]['data'].append(confidence)
            
            return timeline_data
            
        except Exception as e:
            self.logger.error(f"Error preparing timeline data: {e}")
            return {'error': str(e)}
    
    def _prepare_performance_data(self, signals: List[Dict]) -> Dict:
        """
        Prepare performance chart data
        """
        try:
            # Calculate performance metrics
            total_signals = len(signals)
            high_confidence_signals = sum(1 for s in signals if s.get('confidence', 0) >= 4)
            low_confidence_signals = total_signals - high_confidence_signals
            
            performance_data = {
                'labels': ['High Confidence (≥4)', 'Low Confidence (<4)'],
                'datasets': [
                    {
                        'data': [high_confidence_signals, low_confidence_signals],
                        'backgroundColor': [
                            'rgba(75, 192, 192, 0.8)',
                            'rgba(255, 99, 132, 0.8)'
                        ],
                        'borderColor': [
                            'rgb(75, 192, 192)',
                            'rgb(255, 99, 132)'
                        ],
                        'borderWidth': 1
                    }
                ]
            }
            
            return performance_data
            
        except Exception as e:
            self.logger.error(f"Error preparing performance data: {e}")
            return {'error': str(e)}
    
    def _prepare_strategy_data(self, signals: List[Dict]) -> Dict:
        """
        Prepare strategy distribution chart data
        """
        try:
            strategy_counts = {}
            
            for signal in signals:
                strategy = signal.get('strategy', 'Unknown')
                strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
            
            strategy_data = {
                'labels': list(strategy_counts.keys()),
                'datasets': [
                    {
                        'data': list(strategy_counts.values()),
                        'backgroundColor': [
                            'rgba(255, 99, 132, 0.8)',
                            'rgba(54, 162, 235, 0.8)',
                            'rgba(255, 205, 86, 0.8)',
                            'rgba(75, 192, 192, 0.8)',
                            'rgba(153, 102, 255, 0.8)'
                        ],
                        'borderColor': [
                            'rgb(255, 99, 132)',
                            'rgb(54, 162, 235)',
                            'rgb(255, 205, 86)',
                            'rgb(75, 192, 192)',
                            'rgb(153, 102, 255)'
                        ],
                        'borderWidth': 1
                    }
                ]
            }
            
            return strategy_data
            
        except Exception as e:
            self.logger.error(f"Error preparing strategy data: {e}")
            return {'error': str(e)}
    
    def _prepare_confidence_data(self, signals: List[Dict]) -> Dict:
        """
        Prepare confidence distribution chart data
        """
        try:
            confidence_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
            
            for signal in signals:
                confidence = int(signal.get('confidence', 0))
                if confidence in confidence_counts:
                    confidence_counts[confidence] += 1
            
            confidence_data = {
                'labels': ['1', '2', '3', '4', '5'],
                'datasets': [
                    {
                        'label': 'Signal Count',
                        'data': list(confidence_counts.values()),
                        'backgroundColor': 'rgba(54, 162, 235, 0.8)',
                        'borderColor': 'rgb(54, 162, 235)',
                        'borderWidth': 1
                    }
                ]
            }
            
            return confidence_data
            
        except Exception as e:
            self.logger.error(f"Error preparing confidence data: {e}")
            return {'error': str(e)}
    
    def generate_liquidation_heatmap_data(self, liquidation_data: Dict) -> Dict:
        """
        Generate liquidation heatmap data
        """
        try:
            # This would integrate with Coinglass API data
            # For now, return placeholder data
            heatmap_data = {
                'labels': ['BTC', 'ETH', 'BNB', 'ADA', 'SOL'],
                'datasets': [
                    {
                        'label': 'Long Liquidations',
                        'data': [1000000, 500000, 300000, 200000, 150000],
                        'backgroundColor': 'rgba(255, 99, 132, 0.8)',
                        'borderColor': 'rgb(255, 99, 132)',
                        'borderWidth': 1
                    },
                    {
                        'label': 'Short Liquidations',
                        'data': [800000, 400000, 250000, 180000, 120000],
                        'backgroundColor': 'rgba(75, 192, 192, 0.8)',
                        'borderColor': 'rgb(75, 192, 192)',
                        'borderWidth': 1
                    }
                ]
            }
            
            return heatmap_data
            
        except Exception as e:
            self.logger.error(f"Error generating liquidation heatmap data: {e}")
            return {'error': str(e)}
    
    def generate_portfolio_chart_data(self, portfolio_data: Dict) -> Dict:
        """
        Generate portfolio chart data
        """
        try:
            # Extract portfolio metrics
            account_balance = portfolio_data.get('account_balance', 0)
            total_pnl = portfolio_data.get('total_pnl', 0)
            daily_pnl = portfolio_data.get('daily_pnl', 0)
            open_positions = portfolio_data.get('open_positions', 0)
            max_positions = portfolio_data.get('max_positions', 5)
            
            portfolio_chart_data = {
                'balance_overview': {
                    'labels': ['Account Balance', 'Total P&L', 'Daily P&L'],
                    'datasets': [
                        {
                            'data': [account_balance, total_pnl, daily_pnl],
                            'backgroundColor': [
                                'rgba(75, 192, 192, 0.8)',
                                'rgba(255, 205, 86, 0.8)',
                                'rgba(54, 162, 235, 0.8)'
                            ],
                            'borderColor': [
                                'rgb(75, 192, 192)',
                                'rgb(255, 205, 86)',
                                'rgb(54, 162, 235)'
                            ],
                            'borderWidth': 1
                        }
                    ]
                },
                'position_usage': {
                    'labels': ['Open Positions', 'Available'],
                    'datasets': [
                        {
                            'data': [open_positions, max_positions - open_positions],
                            'backgroundColor': [
                                'rgba(255, 99, 132, 0.8)',
                                'rgba(201, 203, 207, 0.8)'
                            ],
                            'borderColor': [
                                'rgb(255, 99, 132)',
                                'rgb(201, 203, 207)'
                            ],
                            'borderWidth': 1
                        }
                    ]
                }
            }
            
            return portfolio_chart_data
            
        except Exception as e:
            self.logger.error(f"Error generating portfolio chart data: {e}")
            return {'error': str(e)}
    
    def format_chart_config(self, chart_type: str, data: Dict) -> Dict:
        """
        Format chart configuration for Chart.js
        """
        try:
            base_config = {
                'type': chart_type,
                'data': data,
                'options': {
                    'responsive': True,
                    'maintainAspectRatio': False,
                    'plugins': {
                        'legend': {
                            'position': 'top',
                        },
                        'title': {
                            'display': True,
                            'text': f'{chart_type.title()} Chart'
                        }
                    }
                }
            }
            
            # Add specific options based on chart type
            if chart_type == 'line':
                base_config['options']['scales'] = {
                    'y': {
                        'beginAtZero': True
                    }
                }
            elif chart_type == 'bar':
                base_config['options']['scales'] = {
                    'y': {
                        'beginAtZero': True
                    }
                }
            
            return base_config
            
        except Exception as e:
            self.logger.error(f"Error formatting chart config: {e}")
            return {'error': str(e)} 