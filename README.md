# 📈 NIFTY 500 Stock Screener Web App

This is a Streamlit-based web app that analyzes the top NIFTY 500 stocks based on:

- 📊 1-Month & 2-Week Momentum
- 💡 RSI (Relative Strength Index) Filter
- 🔒 Stop Loss Monitoring
- 📈 Volume-based Signal Check

## 🚀 Features

- Pulls **live stock data** from Yahoo Finance
- Screens top 10 momentum stocks with RSI between 40–70
- Automatically computes stop-loss and flags risks
- Elegant dashboard built with Streamlit

## 🛠 Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/Nyamathhh/nifty-stock-screener.git
   cd nifty-stock-screener
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the app:
   ```bash
   streamlit run app.py
   ```

## 🌐 Live Deployment

Deployed on [Streamlit Cloud](https://streamlit.io/cloud) with GitHub integration.
