import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class ForexTradingStrategy:
    def __init__(self, symbol, start_date, end_date, rsi_period, rsi_overbought, rsi_oversold):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
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

    def calculate_rsi(self, series, window=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def apply_momentum_filter(self):
        self.data['rsi'] = self.calculate_rsi(self.data['Close'], window=self.rsi_period)
        self.data['trend'] = np.where(self.data['Close'] > self.data['Close'].rolling(window=20).mean(), 1, -1)
        self.data['momentum_filter'] = np.where((self.data['rsi'] < self.rsi_overbought) & (self.data['trend'] == 1), 'buy',
                                                np.where((self.data['rsi'] > self.rsi_oversold) & (self.data['trend'] == -1), 'sell', 'hold'))

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
                    if end_price > start_price and self.data['momentum_filter'].iloc[i] == 'buy':
                        self.data.loc[self.data.index[i+1], 'signal'] = 1
                        self.data.loc[self.data.index[i+1], 'entry_price'] = fib_levels[3]
                        self.data.loc[self.data.index[i+1], 'target'] = self.data['high_pool'].iloc[i+1]
                    elif end_price < start_price and self.data['momentum_filter'].iloc[i] == 'sell':
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
                atr = self.data['High'].iloc[i-20:i].max() - self.data['Low'].iloc[i-20:i].min()
                
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
            
            # Update capital for the current timestamp
            if i > 0:
                self.data.loc[self.data.index[i], 'capital'] = self.data['capital'].iloc[i-1] + sum(trade['pnl'] for trade in self.trades if trade['exit_time'] == self.data.index[i])

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
        self.apply_momentum_filter()
        self.generate_signals()
        self.backtest(initial_capital, risk_per_trade, max_open_trades, leverage, risk_reward_ratio)
        return self.calculate_metrics(initial_capital)

    def plot_results(self, selected_trade=None):
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.1, 
                            subplot_titles=(f'{self.symbol} - Price and Signals', 'RSI', 'Equity Curve'))

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

        # Plot RSI
        fig.add_trace(go.Scatter(x=self.data.index, y=self.data['rsi'], name='RSI'), row=2, col=1)
        fig.add_hline(y=self.rsi_overbought, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=self.rsi_oversold, line_dash="dash", line_color="green", row=2, col=1)

        # Plot equity curve
        fig.add_trace(go.Scatter(x=self.data.index, y=self.data['capital'], name='Equity Curve'), row=3, col=1)

        # If a trade is selected, highlight it on the chart
        if selected_trade is not None:
            fig.add_shape(type="rect",
                x0=selected_trade['entry_time'], y0=0, x1=selected_trade['exit_time'], y1=1,
                xref="x", yref="paper",
                fillcolor="LightSalmon", opacity=0.5,
                layer="below", line_width=0,
            )
            fig.add_trace(go.Scatter(x=[selected_trade['entry_time'], selected_trade['exit_time']],
                                     y=[selected_trade['entry_price'], selected_trade['exit_price']],
                                     mode='lines+markers',
                                     name='Selected Trade',
                                     line=dict(color='black', width=2, dash='dash')), row=1, col=1)

        fig.update_layout(height=1000, title_text="Forex Trading Strategy Results")
        fig.update_xaxes(title_text="Date", row=3, col=1)
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="RSI", row=2, col=1)
        fig.update_yaxes(title_text="Capital", row=3, col=1)

        return fig

# Streamlit app
st.title('Forex Trading Strategy Backtester')

# Sidebar for user inputs
st.sidebar.header('Strategy Parameters')
symbol = st.sidebar.text_input('Forex Symbol', value='EURUSD=X')
start_date = st.sidebar.date_input('Start Date', datetime.now() - timedelta(days=30))
end_date = st.sidebar.date_input('End Date', datetime.now())
initial_capital = st.sidebar.number_input('Initial Capital', value=10000)
risk_per_trade = st.sidebar.slider('Risk per Trade (%)', 0.5, 100.0, 1.0, 0.1) / 100  # Convert to decimal
max_open_trades = st.sidebar.slider('Max Open Trades', 1, 10, 5)
leverage = st.sidebar.slider('Leverage', 1, 100, 1)
risk_reward_ratio = st.sidebar.slider('Risk-Reward Ratio', 1.0, 5.0, 2.0, 0.1)

# RSI inputs
st.sidebar.header('RSI Parameters')
rsi_period = st.sidebar.slider('RSI Period', 5, 30, 14)
rsi_overbought = st.sidebar.slider('RSI Overbought Level', 50, 90, 70)
rsi_oversold = st.sidebar.slider('RSI Oversold Level', 10, 50, 30)

if st.sidebar.button('Run Backtest'):
    with st.spinner('Running backtest...'):
        try:
            strategy = ForexTradingStrategy(symbol, start_date, end_date, rsi_period, rsi_overbought, rsi_oversold)
            results = strategy.run_strategy(initial_capital, risk_per_trade, max_open_trades, leverage, risk_reward_ratio)
            
            if results is not None:
                st.header('Statistics')
                
                # Display backtest results
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Return", f"{results['Total Return']:.2%}")
                col2.metric("Sharpe Ratio", f"{results['Sharpe Ratio']:.2f}")
                col3.metric("Max Drawdown", f"{results['Max Drawdown']:.2%}")
                col4.metric("Final Capital", f"${results['Final Capital']:.2f}")

                trade_df = pd.DataFrame(strategy.trades)
                if not trade_df.empty:
                    # Convert datetime to string for display
                    trade_df['entry_time'] = pd.to_datetime(trade_df['entry_time'])
                    trade_df['exit_time'] = pd.to_datetime(trade_df['exit_time'])
                    trade_df['entry_time_str'] = trade_df['entry_time'].dt.strftime('%Y-%m-%d %H:%M:%S')
                    trade_df['exit_time_str'] = trade_df['exit_time'].dt.strftime('%Y-%m-%d %H:%M:%S')
                    # Round numeric columns
                    trade_df = trade_df.round({'entry_price': 5, 'exit_price': 5, 'pnl': 2})

                    # Additional statistics
                    total_trades = len(trade_df)
                    winning_trades = len(trade_df[trade_df['pnl'] > 0])
                    losing_trades = len(trade_df[trade_df['pnl'] <= 0])
                    win_rate = winning_trades / total_trades if total_trades > 0 else 0

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total Trades", total_trades)
                    col2.metric("Winning Trades", winning_trades)
                    col3.metric("Losing Trades", losing_trades)
                    st.metric("Win Rate", f"{win_rate:.2%}")

                    # Profit Factor
                    total_profit = trade_df[trade_df['pnl'] > 0]['pnl'].sum()
                    total_loss = abs(trade_df[trade_df['pnl'] <= 0]['pnl'].sum())
                    profit_factor = total_profit / total_loss if total_loss != 0 else float('inf')
                    st.metric("Profit Factor", f"{profit_factor:.2f}")

                    # Average Trade
                    avg_trade = trade_df['pnl'].mean()
                    st.metric("Average Trade", f"${avg_trade:.2f}")

                    # Largest Win and Loss
                    largest_win = trade_df['pnl'].max()
                    largest_loss = trade_df['pnl'].min()
                    st.metric("Largest Win", f"${largest_win:.2f}")
                    st.metric("Largest Loss", f"${largest_loss:.2f}")

                    # Total Sum of Profit
                    total_sum_profit = trade_df['pnl'].sum()
                    st.metric("Total Sum of Profit", f"${total_sum_profit:.2f}")

                    st.header('Trade Details')
                    st.dataframe(trade_df[['entry_time_str', 'exit_time_str', 'position', 'entry_price', 'exit_price', 'pnl', 'exit_reason']])

                    st.header('Strategy Performance')
                    selected_trade_index = st.selectbox('Select a trade to highlight', range(len(trade_df)), 
                                                        format_func=lambda x: f"Trade {x+1}: {trade_df.iloc[x]['entry_time_str']} to {trade_df.iloc[x]['exit_time_str']}")
                    selected_trade = trade_df.iloc[selected_trade_index]

                    # Ensure datetime objects for plotting
                    selected_trade['entry_time'] = pd.to_datetime(selected_trade['entry_time'])
                    selected_trade['exit_time'] = pd.to_datetime(selected_trade['exit_time'])

                    fig = strategy.plot_results(selected_trade)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No trades were executed during the backtest period.")
            else:
                st.error("Backtest failed. Please check your inputs and try again.")
        except Exception as e:
            st.error(f"An error occurred: {e}")

st.sidebar.markdown('---')
st.sidebar.write('Developed by Your Name')
