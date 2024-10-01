import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class ForexTradingStrategy:
    def __init__(self, symbol, rsi_period, rsi_overbought, rsi_oversold):
        self.symbol = symbol
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.data = None
        self.trades = []
        self.open_trades = []

    def fetch_data(self, period="30d", interval="15m"):
        try:
            data = yf.download(self.symbol, period=period, interval=interval)
            if data.empty:
                st.error(f"No data available for {self.symbol}")
                return None
            return data
        except Exception as e:
            st.error(f"Error fetching data for {self.symbol}: {e}")
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

    def calculate_rsi(self, series, window=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def generate_signal(self):
        self.data['rsi'] = self.calculate_rsi(self.data['Close'], window=self.rsi_period)
        
        last_index = self.data.index[-1]
        current_price = self.data['Close'].iloc[-1]
        current_rsi = self.data['rsi'].iloc[-1]
        
        signal = 0
        entry_price = np.nan
        stop_loss = np.nan
        take_profit = np.nan
        
        if self.data['displacement_signal'].iloc[-1] == 1:
            start_price = self.data['Close'].iloc[-2]
            end_price = current_price
            fib_levels = self.fibonacci_retracement(start_price, end_price)
            
            if end_price > start_price and current_rsi < self.rsi_oversold:
                signal = 1
                entry_price = fib_levels[3]
                stop_loss = fib_levels[4]
                take_profit = self.data['high_pool'].iloc[-1]
            elif end_price < start_price and current_rsi > self.rsi_overbought:
                signal = -1
                entry_price = fib_levels[3]
                stop_loss = fib_levels[2]
                take_profit = self.data['low_pool'].iloc[-1]
        
        return {
            'symbol': self.symbol,
            'timestamp': last_index,
            'current_price': current_price,
            'rsi': current_rsi,
            'signal': signal,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit
        }

    def run_strategy(self):
        self.data = self.fetch_data()
        if self.data is None:
            return None
        self.identify_liquidity_pools()
        self.detect_displacement()
        return self.generate_signal()

def main():
    st.title('Forex Trading Strategy with Real-time Signaling')

    if 'symbols' not in st.session_state:
        st.session_state.symbols = []

    st.sidebar.header('Add New Symbol')
    new_symbol = st.sidebar.text_input('Forex Symbol (e.g., EURUSD=X)')
    rsi_period = st.sidebar.number_input('RSI Period', value=14, min_value=1, max_value=100)
    rsi_overbought = st.sidebar.number_input('RSI Overbought Level', value=70, min_value=50, max_value=100)
    rsi_oversold = st.sidebar.number_input('RSI Oversold Level', value=30, min_value=0, max_value=50)

    if st.sidebar.button('Add Symbol'):
        if new_symbol and new_symbol not in [s['symbol'] for s in st.session_state.symbols]:
            st.session_state.symbols.append({
                'symbol': new_symbol,
                'rsi_period': rsi_period,
                'rsi_overbought': rsi_overbought,
                'rsi_oversold': rsi_oversold
            })
            st.success(f"Added {new_symbol} to the watch list.")
        else:
            st.warning("Symbol already exists or is invalid.")

    if st.button('Refresh Signals'):
        signals = []
        for symbol_data in st.session_state.symbols:
            strategy = ForexTradingStrategy(
                symbol_data['symbol'],
                symbol_data['rsi_period'],
                symbol_data['rsi_overbought'],
                symbol_data['rsi_oversold']
            )
            signal = strategy.run_strategy()
            if signal:
                signals.append(signal)

        if signals:
            st.header('Current Signals')
            df = pd.DataFrame(signals)
            df['signal'] = df['signal'].map({1: 'Buy', -1: 'Sell', 0: 'No Signal'})
            st.dataframe(df.set_index('symbol'))
        else:
            st.info("No signals generated. Try adding more symbols or adjusting parameters.")

    if st.session_state.symbols:
        st.sidebar.header('Remove Symbol')
        symbol_to_remove = st.sidebar.selectbox('Select symbol to remove', 
                                                [s['symbol'] for s in st.session_state.symbols])
        if st.sidebar.button('Remove Symbol'):
            st.session_state.symbols = [s for s in st.session_state.symbols if s['symbol'] != symbol_to_remove]
            st.sidebar.success(f"Removed {symbol_to_remove} from the watch list.")

if __name__ == "__main__":
    main()
