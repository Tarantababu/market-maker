import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class ForexTradingStrategy:
    def __init__(self, symbol, start_date, end_date):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.data = self.fetch_data()
        self.trades = []
        self.open_trades = []

    def fetch_data(self):
        try:
            data = yf.download(self.symbol, start=self.start_date, end=self.end_date, interval="15m")
            if data.empty:
                st.error(f"No data available for {self.symbol} between {self.start_date} and {self.end_date}")
                return None
            return data
        except Exception as e:
            st.error(f"Error fetching data: {e}")
            return None

    def identify_liquidity_pools(self, window=10):
        self.data['high_pool'] = self.data['High'].rolling(window=window).max()
        self.data['low_pool'] = self.data['Low'].rolling(window=window).min()

    def detect_displacement(self, threshold=0.0005):
        self.data['displacement'] = (self.data['Close'] - self.data['Close'].shift(1)) / self.data['Close'].shift(1)
        self.data['displacement_signal'] = np.where(abs(self.data['displacement']) > threshold, 1, 0)

    def fibonacci_retracement(self, start, end):
        diff = end - start
        return [end - level * diff for level in [0.236, 0.382, 0.5, 0.618, 0.786]]

    def identify_fair_value_gaps(self, threshold=0.0005):
        self.data['gap'] = self.data['Open'] - self.data['Close'].shift(1)
        self.data['fvg'] = np.where(abs(self.data['gap']) > threshold * self.data['Close'].shift(1), 1, 0)

    def generate_signals(self):
        self.data['signal'] = 0
        self.data['entry_price'] = np.nan
        self.data['target'] = np.nan
        
        for i in range(1, len(self.data) - 1):
            if self.data['displacement_signal'].iloc[i] == 1:
                start_price = self.data['Close'].iloc[i-1]
                end_price = self.data['Close'].iloc[i]
                fib_levels = self.fibonacci_retracement(start_price, end_price)
                
                if (self.data['Low'].iloc[i+1] <= fib_levels[4]) and (self.data['High'].iloc[i+1] >= fib_levels[2]):
                    if end_price > start_price:
                        self.data.loc[self.data.index[i+1], 'signal'] = 1
                        self.data.loc[self.data.index[i+1], 'entry_price'] = fib_levels[3]
                        self.data.loc[self.data.index[i+1], 'target'] = self.data['high_pool'].iloc[i+1]
                    else:
                        self.data.loc[self.data.index[i+1], 'signal'] = -1
                        self.data.loc[self.data.index[i+1], 'entry_price'] = fib_levels[3]
                        self.data.loc[self.data.index[i+1], 'target'] = self.data['low_pool'].iloc[i+1]

    def backtest(self, initial_capital=10000, risk_per_trade=0.01, max_open_trades=5, leverage=1, risk_reward_ratio=2):
        self.data['capital'] = initial_capital
        
        for i in range(1, len(self.data)):
            current_price = self.data['Close'].iloc[i]
            
            # Check for new trade signal
            if self.data['signal'].iloc[i] != 0 and len(self.open_trades) < max_open_trades:
                entry_price = self.data['entry_price'].iloc[i]
                position = self.data['signal'].iloc[i]
                
                # Calculate ATR for dynamic stop loss and take profit
                atr = self.data['High'].iloc[i-7:i].max() - self.data['Low'].iloc[i-7:i].min()
                
                # Set stop loss and take profit based on risk_reward_ratio
                stop_loss_price = entry_price - position * atr
                take_profit_price = entry_price + position * atr * risk_reward_ratio
                
                # Calculate position size based on risk per trade and leverage
                risk_amount = risk_per_trade * self.data['capital'].iloc[i-1]
                pip_value = 0.0001 if 'JPY' not in self.symbol else 0.01
                position_size = (risk_amount / (atr / pip_value)) * leverage
                
                self.open_trades.append({
                    'entry_time': self.data.index[i],
                    'position': position,
                    'entry_price': entry_price,
                    'stop_loss': stop_loss_price,
                    'take_profit': take_profit_price,
                    'position_size': position_size
                })
            
            # Check open trades for exit conditions
            closed_trades = []
            for trade in self.open_trades:
                if (trade['position'] == 1 and current_price <= trade['stop_loss']) or \
                   (trade['position'] == -1 and current_price >= trade['stop_loss']):
                    pnl = trade['position'] * (current_price - trade['entry_price']) * trade['position_size']
                    self.data.loc[self.data.index[i], 'capital'] += pnl
                    self.trades.append({
                        'entry_time': trade['entry_time'],
                        'exit_time': self.data.index[i],
                        'position': trade['position'],
                        'entry_price': trade['entry_price'],
                        'exit_price': current_price,
                        'pnl': pnl,
                        'exit_reason': 'Stop Loss'
                    })
                    closed_trades.append(trade)
                
                elif (trade['position'] == 1 and current_price >= trade['take_profit']) or \
                     (trade['position'] == -1 and current_price <= trade['take_profit']):
                    pnl = trade['position'] * (current_price - trade['entry_price']) * trade['position_size']
                    self.data.loc[self.data.index[i], 'capital'] += pnl
                    self.trades.append({
                        'entry_time': trade['entry_time'],
                        'exit_time': self.data.index[i],
                        'position': trade['position'],
                        'entry_price': trade['entry_price'],
                        'exit_price': current_price,
                        'pnl': pnl,
                        'exit_reason': 'Take Profit'
                    })
                    closed_trades.append(trade)
            
            # Remove closed trades from open trades list
            self.open_trades = [trade for trade in self.open_trades if trade not in closed_trades]
            
            # Ensure capital is updated even if no trades are closed
            if i > 0 and self.data['capital'].iloc[i] == self.data['capital'].iloc[0]:
                self.data.loc[self.data.index[i], 'capital'] = self.data['capital'].iloc[i-1]

        # Calculate daily returns
        self.data['daily_returns'] = self.data['capital'].pct_change().fillna(0)

    def calculate_metrics(self, initial_capital):
        total_return = (self.data['capital'].iloc[-1] - initial_capital) / initial_capital
        daily_returns = self.data['daily_returns']
        
        if len(daily_returns) > 1 and daily_returns.std() != 0:
            sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std()
        else:
            sharpe_ratio = 0
        
        max_drawdown = (self.data['capital'].cummax() - self.data['capital']) / self.data['capital'].cummax()
        
        return {
            'Total Return': total_return,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown': max_drawdown.max(),
            'Final Capital': self.data['capital'].iloc[-1]
        }

    def run_strategy(self, initial_capital=10000, risk_per_trade=0.01, max_open_trades=5, leverage=1, risk_reward_ratio=2):
        if self.data is None:
            return None
        self.identify_liquidity_pools()
        self.detect_displacement()
        self.identify_fair_value_gaps()
        self.generate_signals()
        self.backtest(initial_capital, risk_per_trade, max_open_trades, leverage, risk_reward_ratio)
        return self.calculate_metrics(initial_capital)

    def plot_results(self, selected_trade=None):
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, 
                            subplot_titles=(f'{self.symbol} - Price and Signals', 'Equity Curve'))

        # Plot price
        fig.add_trace(go.Scatter(x=self.data.index, y=self.data['Close'], name='Close Price'), row=1, col=1)

        # Plot buy signals
        buy_signals = self.data[self.data['signal'] == 1]
        fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['Close'], mode='markers', 
                                 name='Buy Signal', marker=dict(color='green', symbol='triangle-up', size=10)), row=1, col=1)

        # Plot sell signals
        sell_signals = self.data[self.data['signal'] == -1]
        fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['Close'], mode='markers', 
                                 name='Sell Signal', marker=dict(color='red', symbol='triangle-down', size=10)), row=1, col=1)

        # Plot equity curve
        fig.add_trace(go.Scatter(x=self.data.index, y=self.data['capital'], name='Equity Curve', line=dict(color='blue')), row=2, col=1)

        fig.update_layout(height=600, title_text=f"Strategy Backtest Results for {self.symbol}")
        st.plotly_chart(fig)

        if selected_trade is not None:
            st.write(f"Selected Trade Details:\n {selected_trade}")

# Streamlit interface
st.title('Forex Trading Strategy Backtest')

symbol = st.text_input('Enter Forex Pair Symbol (e.g., EURUSD=X):', 'EURUSD=X')
start_date = st.date_input('Start Date:', datetime.now() - timedelta(days=365))
end_date = st.date_input('End Date:', datetime.now())
initial_capital = st.number_input('Initial Capital:', min_value=1000, value=10000, step=1000)
risk_per_trade = st.slider('Risk per Trade (%):', 0.01, 0.05, 0.01)
max_open_trades = st.slider('Max Open Trades:', 1, 10, 5)
leverage = st.slider('Leverage:', 1, 50, 10)
risk_reward_ratio = st.slider('Risk-Reward Ratio:', 1.0, 3.0, 2.0)

if st.button('Run Backtest'):
    strategy = ForexTradingStrategy(symbol, start_date, end_date)
    metrics = strategy.run_strategy(initial_capital, risk_per_trade, max_open_trades, leverage, risk_reward_ratio)
    
    if metrics:
        st.write('Backtest Metrics:')
        st.write(f"Total Return: {metrics['Total Return']*100:.2f}%")
        st.write(f"Sharpe Ratio: {metrics['Sharpe Ratio']:.2f}")
        st.write(f"Max Drawdown: {metrics['Max Drawdown']*100:.2f}%")
        st.write(f"Final Capital: {metrics['Final Capital']:.2f}")
        
        strategy.plot_results()

