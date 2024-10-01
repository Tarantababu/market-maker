import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import time
import threading

TELEGRAM_TOKEN = "7148511647:AAFlMohYiqPF2GQFtri2qW4H0WU2-j174TQ"
TELEGRAM_CHAT_ID = "5611879467"

def send_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    params = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML"
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        print(f"Message sent successfully: {response.json()}")
        return "sent"
    except requests.RequestException as e:
        print(f"Failed to send message: {e}")
        return "failed"

class ForexTradingStrategy:
    def __init__(self, symbol, rsi_period, rsi_overbought, rsi_oversold, risk_reward_ratio, atr_period):
        self.symbol = symbol
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.risk_reward_ratio = risk_reward_ratio
        self.atr_period = atr_period
        self.data = None

    def fetch_data(self, period="1d", interval="15m"):
        try:
            data = yf.download(self.symbol, period=period, interval=interval)
            if data.empty:
                print(f"No data available for {self.symbol}")
                return None
            return data
        except Exception as e:
            print(f"Error fetching data for {self.symbol}: {e}")
            return None

    def calculate_atr(self, period=14):
        high_low = self.data['High'] - self.data['Low']
        high_close = np.abs(self.data['High'] - self.data['Close'].shift())
        low_close = np.abs(self.data['Low'] - self.data['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        return true_range.rolling(period).mean()

    def calculate_rsi(self, series, window=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def generate_signal(self):
        self.data['rsi'] = self.calculate_rsi(self.data['Close'], window=self.rsi_period)
        self.data['atr'] = self.calculate_atr(period=self.atr_period)
        
        last_index = self.data.index[-1]
        current_price = self.data['Close'].iloc[-1]
        current_rsi = self.data['rsi'].iloc[-1]
        current_atr = self.data['atr'].iloc[-1]
        
        signal = 0
        entry_price = np.nan
        stop_loss = np.nan
        take_profit = np.nan
        
        if current_rsi < self.rsi_oversold:
            signal = 1  # Buy signal
            entry_price = current_price
            stop_loss = entry_price - current_atr
            take_profit = entry_price + (current_atr * self.risk_reward_ratio)
        elif current_rsi > self.rsi_overbought:
            signal = -1  # Sell signal
            entry_price = current_price
            stop_loss = entry_price + current_atr
            take_profit = entry_price - (current_atr * self.risk_reward_ratio)
        
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
        return self.generate_signal()

def send_signals_to_telegram():
    while True:
        current_time = datetime.now()
        if current_time.minute % 15 == 1:
            signals = []
            for symbol_data in st.session_state.symbols:
                strategy = ForexTradingStrategy(
                    symbol_data['symbol'],
                    symbol_data['rsi_period'],
                    symbol_data['rsi_overbought'],
                    symbol_data['rsi_oversold'],
                    symbol_data['risk_reward_ratio'],
                    symbol_data['atr_period']
                )
                signal = strategy.run_strategy()
                if signal and signal['signal'] != 0:
                    signals.append(signal)
            
            if signals:
                message = "<b>Active Signals:</b>\n\n"
                for signal in signals:
                    message += f"<b>Symbol:</b> {signal['symbol']}\n"
                    message += f"<b>Signal:</b> {'Buy' if signal['signal'] == 1 else 'Sell'}\n"
                    message += f"<b>Current Price:</b> {signal['current_price']:.5f}\n"
                    message += f"<b>RSI:</b> {signal['rsi']:.2f}\n"
                    message += f"<b>Entry Price:</b> {signal['entry_price']:.5f}\n"
                    message += f"<b>Stop Loss:</b> {signal['stop_loss']:.5f}\n"
                    message += f"<b>Take Profit:</b> {signal['take_profit']:.5f}\n\n"
                
                send_message(message)
                print(f"Sent signals at {current_time}")
            else:
                print(f"No signals to send at {current_time}")
            
            time.sleep(60)
        else:
            time.sleep(30)

def main():
    st.title('Forex Trading Strategy with Dynamic SL/TP')

    if 'symbols' not in st.session_state:
        st.session_state.symbols = []

    st.sidebar.header('Add New Symbol')
    new_symbol = st.sidebar.text_input('Forex Symbol (e.g., EURUSD=X)')
    rsi_period = st.sidebar.number_input('RSI Period', value=14, min_value=1, max_value=100)
    rsi_overbought = st.sidebar.number_input('RSI Overbought Level', value=70, min_value=50, max_value=100)
    rsi_oversold = st.sidebar.number_input('RSI Oversold Level', value=30, min_value=0, max_value=50)
    risk_reward_ratio = st.sidebar.number_input('Risk-Reward Ratio', value=2.0, min_value=0.1, max_value=10.0, step=0.1)
    atr_period = st.sidebar.number_input('ATR Period', value=14, min_value=1, max_value=100)

    if st.sidebar.button('Add Symbol'):
        if new_symbol and new_symbol not in [s['symbol'] for s in st.session_state.symbols]:
            st.session_state.symbols.append({
                'symbol': new_symbol,
                'rsi_period': rsi_period,
                'rsi_overbought': rsi_overbought,
                'rsi_oversold': rsi_oversold,
                'risk_reward_ratio': risk_reward_ratio,
                'atr_period': atr_period
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
                symbol_data['rsi_oversold'],
                symbol_data['risk_reward_ratio'],
                symbol_data['atr_period']
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

    # Start the Telegram messaging thread
    if 'telegram_thread' not in st.session_state:
        st.session_state.telegram_thread = threading.Thread(target=send_signals_to_telegram, daemon=True)
        st.session_state.telegram_thread.start()

if __name__ == "__main__":
    main()
