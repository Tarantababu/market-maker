import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class ForexTradingStrategy:
    # ... [All the previous methods remain the same up to the backtest method] ...

    def backtest(self, initial_capital=10000, risk_per_trade=0.01, max_open_trades=5):
        self.data['capital'] = pd.Series([initial_capital] * len(self.data), dtype=float)
        
        for i in range(1, len(self.data)):
            current_price = self.data['Close'].iloc[i]
            
            # Check for new trade signal
            if self.data['signal'].iloc[i] != 0 and len(self.open_trades) < max_open_trades:
                entry_price = self.data['entry_price'].iloc[i]
                position = self.data['signal'].iloc[i]
                
                # Calculate ATR for dynamic stop loss and take profit
                atr = self.data['High'].iloc[i-20:i].max() - self.data['Low'].iloc[i-20:i].min()
                
                # Set stop loss at 1 ATR
                stop_loss_price = entry_price - position * atr
                
                # Set take profit at 2 ATR (1:2 risk-reward ratio)
                take_profit_price = entry_price + position * 2 * atr
                
                # Calculate position size based on risk per trade
                risk_amount = risk_per_trade * self.data['capital'].iloc[i-1]
                pip_value = 0.0001 if 'JPY' not in self.symbol else 0.01
                position_size = risk_amount / (atr / pip_value)
                
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
                    pnl = trade['position_size'] * (trade['stop_loss'] - trade['entry_price']) / trade['entry_price'] * self.data['capital'].iloc[i-1]
                    self.data.loc[self.data.index[i], 'capital'] += pnl
                    self.trades.append({
                        'entry_time': trade['entry_time'],
                        'exit_time': self.data.index[i],
                        'position': trade['position'],
                        'entry_price': trade['entry_price'],
                        'exit_price': trade['stop_loss'],
                        'pnl': pnl,
                        'exit_reason': 'Stop Loss'
                    })
                    closed_trades.append(trade)
                
                elif (trade['position'] == 1 and current_price >= trade['take_profit']) or \
                     (trade['position'] == -1 and current_price <= trade['take_profit']):
                    pnl = trade['position_size'] * (trade['take_profit'] - trade['entry_price']) / trade['entry_price'] * self.data['capital'].iloc[i-1]
                    self.data.loc[self.data.index[i], 'capital'] += pnl
                    self.trades.append({
                        'entry_time': trade['entry_time'],
                        'exit_time': self.data.index[i],
                        'position': trade['position'],
                        'entry_price': trade['entry_price'],
                        'exit_price': trade['take_profit'],
                        'pnl': pnl,
                        'exit_reason': 'Take Profit'
                    })
                    closed_trades.append(trade)
            
            # Remove closed trades from open trades list
            self.open_trades = [trade for trade in self.open_trades if trade not in closed_trades]
            
            if self.data['capital'].iloc[i] == self.data['capital'].iloc[i-1]:
                self.data.loc[self.data.index[i], 'capital'] = self.data['capital'].iloc[i-1]

        # Calculate performance metrics
        total_return = (self.data['capital'].iloc[-1] - initial_capital) / initial_capital
        daily_returns = self.data['capital'].diff().resample('D').last().dropna()
        
        if len(daily_returns) > 1 and daily_returns.std() != 0:
            sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std()
        else:
            sharpe_ratio = np.nan
        
        max_drawdown = (self.data['capital'].cummax() - self.data['capital']) / self.data['capital'].cummax()
        
        return {
            'Total Return': total_return,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown': max_drawdown.max(),
            'Final Capital': self.data['capital'].iloc[-1]
        }

    def run_strategy(self):
        self.identify_liquidity_pools()
        self.detect_displacement()
        self.identify_fair_value_gaps()
        self.generate_signals()
        return self.backtest()

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
        fig.add_trace(go.Scatter(x=self.data.index, y=self.data['capital'], name='Equity Curve'), row=2, col=1)

        # If a trade is selected, highlight it on the chart
        if selected_trade:
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

        fig.update_layout(height=800, title_text="Forex Trading Strategy Results")
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Capital", row=2, col=1)

        return fig

# Streamlit app
st.title('Forex Trading Strategy Backtester')

# Sidebar for user inputs
st.sidebar.header('Strategy Parameters')
symbol = st.sidebar.text_input('Forex Symbol', value='EURUSD=X')
start_date = st.sidebar.date_input('Start Date', datetime.now() - timedelta(days=30))
end_date = st.sidebar.date_input('End Date', datetime.now())
initial_capital = st.sidebar.number_input('Initial Capital', value=10000)
risk_per_trade = st.sidebar.slider('Risk per Trade', 0.01, 0.05, 0.01, 0.01)
max_open_trades = st.sidebar.slider('Max Open Trades', 1, 10, 5)

if st.sidebar.button('Run Backtest'):
    strategy = ForexTradingStrategy(symbol, start_date, end_date)
    results = strategy.run_strategy()

    st.header('Backtest Results')
    for key, value in results.items():
        st.metric(key, f"{value:.4f}")

    st.header('Trade Details')
    trade_df = pd.DataFrame(strategy.trades)
    st.dataframe(trade_df)

    st.header('Strategy Performance')
    selected_trade = None
    if not trade_df.empty:
        selected_trade_index = st.selectbox('Select a trade to highlight', range(len(trade_df)), 
                                            format_func=lambda x: f"Trade {x+1}")
        selected_trade = trade_df.iloc[selected_trade_index]

    fig = strategy.plot_results(selected_trade)
    st.plotly_chart(fig, use_container_width=True)

st.sidebar.markdown('---')
st.sidebar.write('Developed by Your Name')
