import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import yfinance as yf
import os
from streamlit_autorefresh import st_autorefresh

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

# Load NIFTY 500 symbols from GitHub CSV
@st.cache_data
def load_nifty500_symbols():
    url = 'https://raw.githubusercontent.com/Nyamathhh/nifty-stock-screener/main/nifty500.csv'
    df = pd.read_csv(url)
    return df.head(100)['Symbol'].tolist()

# Streamlit App
st.set_page_config(page_title="NIFTY 500 Screener", layout="wide")
st.title("📈 NIFTY 500 Stock Screener Dashboard")
st.markdown("Analyze top 100 momentum stocks with RSI, return, volume, and price charts.")

symbols = [s + ".NS" for s in load_nifty500_symbols()]
end_date = datetime.today()
start_date = end_date - timedelta(days=60)

with st.sidebar:
    st.header("Settings")
    capital = st.number_input("Investment Capital (INR)", value=100000)
    risk_per_trade_pct = st.slider("Risk per Trade (%)", 1, 10, 5)
    stop_loss_pct = st.slider("Stop Loss (%)", 1, 10, 5)
    rsi_min = st.slider("Min RSI", 0, 100, 30)
    rsi_max = st.slider("Max RSI", 0, 100, 75)
    min_return_1m = st.slider("Minimum 1M Return (%)", -10, 10, 0)
    min_volume = st.number_input("Minimum Daily Volume", value=1000000)
    refresh_interval = st.slider("Auto-refresh interval (seconds)", 0, 3600, 0)
    run_button = st.button("Run Screener")

# Optional auto-refresh trigger
if refresh_interval > 0:
    st_autorefresh(interval=refresh_interval * 1000, limit=None, key="auto-refresh")

@st.cache_data(show_spinner=False)
def get_filtered_stocks(rsi_min, rsi_max, min_return_1m, min_volume, stop_loss_pct):
    results = []
    all_data = yf.download(symbols, start=start_date, end=end_date, group_by="ticker", threads=True)
    total = len(symbols)
    progress = st.progress(0)

    for idx, stock in enumerate(symbols):
        try:
            df = all_data[stock].dropna()
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

            if rsi_min < rsi < rsi_max and return_1m > min_return_1m and volume > min_volume:
                results.append({
                    'Stock': stock.replace('.NS', ''),
                    'Symbol': stock,
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
        progress.progress((idx + 1) / total)

    df_final = pd.DataFrame(results)
    if df_final.empty:
        return pd.DataFrame()
    return df_final.sort_values(by='1M Return (%)', ascending=False).head(10)

# Log performance
def log_performance(df):
    df['Date'] = datetime.today().strftime('%Y-%m-%d')
    history_file = 'portfolio_history.csv'
    if os.path.exists(history_file):
        prev = pd.read_csv(history_file)
        df = pd.concat([prev, df], ignore_index=True)
    df.to_csv(history_file, index=False)

# Run
if run_button or refresh_interval > 0:
    top_stocks = get_filtered_stocks(rsi_min, rsi_max, min_return_1m, min_volume, stop_loss_pct)
    st.success("Top 10 Momentum Stocks with RSI Filter")

    if not top_stocks.empty:
        log_performance(top_stocks.drop(columns=['Symbol']))
        styled_df = top_stocks.style.background_gradient(cmap='YlGnBu', subset=['1M Return (%)', '2W Return (%)', 'RSI'])
        st.dataframe(styled_df, use_container_width=True)

        csv = top_stocks.drop(columns=['Symbol']).to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download CSV", csv, "top_stocks.csv", "text/csv")

        st.subheader("📊 Stock Charts")
        cols = st.columns(2)
        for i, row in top_stocks.iterrows():
            with cols[i % 2]:
                data = yf.download(row['Symbol'], start=start_date, end=end_date)
                data['RSI'] = calculate_rsi(data['Close'])

                fig, ax1 = plt.subplots(figsize=(6, 3))
                ax1.plot(data.index, data['Close'], label='Close Price', color='blue')
                ax1.set_ylabel('Price', color='blue')
                ax1.tick_params(axis='y', labelcolor='blue')
                ax1.set_title(row['Stock'])

                ax2 = ax1.twinx()
                ax2.plot(data.index, data['RSI'], label='RSI', color='orange')
                ax2.set_ylabel('RSI', color='orange')
                ax2.tick_params(axis='y', labelcolor='orange')
                ax2.axhline(70, color='red', linestyle='--', linewidth=0.5)
                ax2.axhline(30, color='green', linestyle='--', linewidth=0.5)

                fig.tight_layout()
                st.pyplot(fig)
    else:
        st.warning("No stocks met the criteria today. Try adjusting filters.")
