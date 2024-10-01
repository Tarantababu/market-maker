import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Existing strategy functions (modified for real-time use)
def resample_data(data, timeframe):
    if timeframe == '1H':
        return data.resample('1h').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'})
    elif timeframe == '4H':
        return data.resample('4h').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'})
    elif timeframe == 'D':
        return data.resample('D').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'})
    else:
        return data

def identify_market_structure(data, timeframe):
    resampled_data = resample_data(data, timeframe)
    
    resampled_data['Higher_High'] = resampled_data['High'] > resampled_data['High'].shift(1)
    resampled_data['Higher_Low'] = resampled_data['Low'] > resampled_data['Low'].shift(1)
    resampled_data['Lower_High'] = resampled_data['High'] < resampled_data['High'].shift(1)
    resampled_data['Lower_Low'] = resampled_data['Low'] < resampled_data['Low'].shift(1)
    
    resampled_data['Bullish'] = (resampled_data['Higher_High'] & resampled_data['Higher_Low']).rolling(window=2).sum() == 2
    resampled_data['Bearish'] = (resampled_data['Lower_High'] & resampled_data['Lower_Low']).rolling(window=2).sum() == 2
    
    return resampled_data[['Bullish', 'Bearish']]

def define_dealing_range(data):
    data['LT_Range_High'] = data['High'].rolling(window=100).max()
    data['LT_Range_Low'] = data['Low'].rolling(window=100).min()
    data['IT_Range_High'] = data['High'].rolling(window=50).max()
    data['IT_Range_Low'] = data['Low'].rolling(window=50).min()
    data['ST_Range_High'] = data['High'].rolling(window=20).max()
    data['ST_Range_Low'] = data['Low'].rolling(window=20).min()
    
    data['LT_Range_Mid'] = (data['LT_Range_High'] + data['LT_Range_Low']) / 2
    data['IT_Range_Mid'] = (data['IT_Range_High'] + data['IT_Range_Low']) / 2
    data['ST_Range_Mid'] = (data['ST_Range_High'] + data['ST_Range_Low']) / 2
    
    data['In_LT_Premium'] = data['Close'] > data['LT_Range_Mid']
    data['In_LT_Discount'] = data['Close'] <= data['LT_Range_Mid']
    data['In_IT_Premium'] = data['Close'] > data['IT_Range_Mid']
    data['In_IT_Discount'] = data['Close'] <= data['IT_Range_Mid']
    data['In_ST_Premium'] = data['Close'] > data['ST_Range_Mid']
    data['In_ST_Discount'] = data['Close'] <= data['ST_Range_Mid']
    
    return data

def identify_fair_value_gaps(data):
    data['FVG_Bullish'] = (data['Low'].shift(1) > data['High'].shift(2)) & (data['Open'] < data['Close'].shift(1))
    data['FVG_Bearish'] = (data['High'].shift(1) < data['Low'].shift(2)) & (data['Open'] > data['Close'].shift(1))
    return data

def evaluate_pd_array(data):
    data['Bullish_FVG_Respected'] = (data['Low'] <= data['Low'].shift(1)) & data['FVG_Bullish'].shift(1)
    data['Bearish_FVG_Respected'] = (data['High'] >= data['High'].shift(1)) & data['FVG_Bearish'].shift(1)
    
    data['Old_High'] = data['High'].rolling(window=20).max().shift(1)
    data['Old_Low'] = data['Low'].rolling(window=20).min().shift(1)
    
    return data

def generate_signals(data, daily_data):
    data['Signal'] = 0
    
    for i in range(1, len(data)):
        current_date = data.index[i].date()
        daily_open = daily_data.loc[daily_data.index.date == current_date, 'Open'].iloc[0]
        
        bullish_structure = data['Bullish_4H'].iloc[i] or data['Bullish_D'].iloc[i]
        bearish_structure = data['Bearish_4H'].iloc[i] or data['Bearish_D'].iloc[i]
        
        bullish_condition = (
            bullish_structure and
            data['In_LT_Discount'].iloc[i] and
            data['In_IT_Discount'].iloc[i] and
            data['In_ST_Discount'].iloc[i] and
            data['Bullish_FVG_Respected'].iloc[i] and
            data['Close'].iloc[i] < daily_open
        )
        
        bearish_condition = (
            bearish_structure and
            data['In_LT_Premium'].iloc[i] and
            data['In_IT_Premium'].iloc[i] and
            data['In_ST_Premium'].iloc[i] and
            data['Bearish_FVG_Respected'].iloc[i] and
            data['Close'].iloc[i] > daily_open
        )
        
        if bullish_condition:
            data.loc[data.index[i], 'Signal'] = 1
        elif bearish_condition:
            data.loc[data.index[i], 'Signal'] = -1
    
    return data

# Streamlit app
st.title('Forex Strategy Live Signals')

# Sidebar for user input
st.sidebar.header('Settings')
symbol = st.sidebar.selectbox('Select Forex Pair', ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'AUDUSD=X', 'USDCAD=X'])
lookback_days = st.sidebar.slider('Lookback Period (days)', 1, 30, 7)

# Function to fetch real-time data
@st.cache_data(ttl=300)
def fetch_data(symbol, lookback_days):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days)
    data = yf.download(symbol, start=start_date, end=end_date, interval='15m')
    return data

# Fetch and process data
data = fetch_data(symbol, lookback_days)

# Apply strategy components
data['Bullish_15M'], data['Bearish_15M'] = identify_market_structure(data, '15m')
data['Bullish_1H'], data['Bearish_1H'] = identify_market_structure(data, '1H')
data['Bullish_4H'], data['Bearish_4H'] = identify_market_structure(data, '4H')
data['Bullish_D'], data['Bearish_D'] = identify_market_structure(data, 'D')

data = define_dealing_range(data)
data = identify_fair_value_gaps(data)
data = evaluate_pd_array(data)

# Get daily data for HTF open
daily_data = resample_data(data, 'D')

# Generate signals
data = generate_signals(data, daily_data)

# Create Plotly chart
fig = go.Figure()

# Candlestick chart
fig.add_trace(go.Candlestick(x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name='Price'))

# Add buy signals
buy_signals = data[data['Signal'] == 1]
fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['Low'],
                         mode='markers',
                         marker=dict(symbol='triangle-up', size=10, color='green'),
                         name='Buy Signal'))

# Add sell signals
sell_signals = data[data['Signal'] == -1]
fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['High'],
                         mode='markers',
                         marker=dict(symbol='triangle-down', size=10, color='red'),
                         name='Sell Signal'))

# Update layout
fig.update_layout(title=f'{symbol} - Price and Signals',
                  xaxis_title='Date',
                  yaxis_title='Price',
                  xaxis_rangeslider_visible=False)

# Display the chart
st.plotly_chart(fig, use_container_width=True)

# Display recent signals
st.subheader('Recent Signals')
recent_signals = data[data['Signal'] != 0].tail(10)
if not recent_signals.empty:
    st.table(recent_signals[['Close', 'Signal']].style.format({'Close': '{:.5f}'}))
else:
    st.write("No recent signals generated.")

# Add a refresh button
if st.button('Refresh Data'):
    st.experimental_rerun()

st.sidebar.info('Data updates every 5 minutes. Click "Refresh Data" for the latest signals.')
