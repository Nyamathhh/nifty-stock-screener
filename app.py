import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
import requests
import yfinance as yf  # ✅ Replaced pandas_datareader

# Function to calculate RSI
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Load NIFTY 500 list from NSE
@st.cache_data
def load_nifty500_symbols():
    url = 'https://archives.nseindia.com/content/indices/ind_nifty500list.csv'
    df = pd.read_csv(url)
    symbols = [symbol + ".NS" for symbol in df['Symbol'].tolist()]
    return symbols

# Streamlit App
st.set_page_config(page_title="NIFTY 500 Screener", layout="wide")
st.title("📈 NIFTY 500 Stock Screener Dashboard")
st.markdown("Analyze top momentum stocks with RSI and price performance filters.")

symbols = load_nifty500_symbols()
end_date = datetime.today()
start_date = end_date - timedelta(days=60)

with st.sidebar:
    st.header("Settings")
    capital = st.number_input("Investment Capital (INR)", value=100000)
    risk_per_trade_pct = st.slider("Risk per Trade (%)", min_value=1, max_value=10, value=5)
    stop_loss_pct = st.slider("Stop Loss (%)", min_value=1, max_value=10, value=5)
    run_button = st.button("Run Screener")

@st.cache_data(show_spinner=False)
def get_filtered_stocks():
    results = []
    for stock in symbols:
        try:
            df = yf.download(stock, start=start_date, end=end_date)
            if df.empty or len(df) < 22:
                continue

            latest_price = df['Close'].iloc[-1]
            return_1m = (df['Close'].iloc[-1] / df['Close'].iloc[-22] - 1) * 100
            return_2w = (df['Close'].iloc[-1] / df['Close'].iloc[-10] - 1) * 100
            rsi = calculate_rsi(df['Close']).iloc[-1]
            volume = df['Volume'].iloc[-1]

            buy_price = latest_price
            stop_loss_price = buy_price * (1 - stop_loss_pct / 100)
            sl_hit = latest_price < stop_loss_price

            if rsi > 40 and rsi < 70 and return_1m > 0:
                results.append({
                    'Stock': stock.replace('.NS', ''),
                    'Latest Price': round(latest_price, 2),
                    '1M Return (%)': round(return_1m, 2),
                    '2W Return (%)': round(return_2w, 2),
                    'RSI': round(rsi, 2),
                    'Volume': int(volume),
                    'Buy Price': round(buy_price, 2),
                    'Stop Loss Price': round(stop_loss_price, 2),
                    'SL Hit?': sl_hit
                })
        except Exception:
            continue

    df_final = pd.DataFrame(results)
    top_stocks = df_final.sort_values(by='1M Return (%)', ascending=False).head(10)
    return top_stocks

if run_button:
    top_stocks = get_filtered_stocks()
    st.success("Top 10 Momentum Stocks with RSI Filter")

    if not top_stocks.empty:
        styled_df = top_stocks.style.format({
            'Latest Price': "₹{:.2f}",
            '1M Return (%)': "{:.2f}%",
            '2W Return (%)': "{:.2f}%",
            'RSI': "{:.2f}",
            'Buy Price': "₹{:.2f}",
            'Stop Loss Price': "₹{:.2f}"
        }).background_gradient(cmap='YlGnBu', subset=['1M Return (%)', '2W Return (%)', 'RSI'])

        st.dataframe(styled_df, use_container_width=True)
    else:
        st.warning("No stocks met the criteria today.")
