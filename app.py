import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import lightgbm as lgb
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(page_title="AlphaEngine | Quant Scanner", layout="wide")
st.title("🚀 AlphaEngine (50-EMA Optimized)")
st.markdown("---")

# 1. MARKET REGIME CHECK (FAST 50-EMA)
@st.cache_data(ttl=3600)
def check_market_regime():
    nifty = yf.download("^NSEI", period="1y", progress=False)
    # yfinance update fix
    if isinstance(nifty.columns, pd.MultiIndex):
        nifty.columns = [col[0] for col in nifty.columns]
        
    nifty['EMA_50'] = nifty['Close'].ewm(span=50, adjust=False).mean()
    current_close = nifty['Close'].iloc[-1]
    current_ema = nifty['EMA_50'].iloc[-1]
    
    regime = "BULLISH" if current_close > current_ema else "BEARISH"
    return regime, current_close, current_ema

regime, close, ema = check_market_regime()

# REGIME FILTER LOGIC
if regime == "BEARISH":
    st.error(f"🚨 MARKET REGIME: BEARISH (Nifty at {close:.2f} is below 50-EMA {ema:.2f})")
    st.warning("🛡️ SYSTEM IN CASH MODE: No trades to be executed today. Capital is protected.")
    st.stop() # Code stops here, avoids taking any trades
else:
    st.success(f"🟢 MARKET REGIME: BULLISH (Nifty at {close:.2f} is above 50-EMA {ema:.2f})")
    st.info("System is scanning for high-probability momentum stocks...")

# 2. SCANNER LOGIC (Executes only if Bullish)
@st.cache_data(ttl=3600)
def get_stock_data():
    # Top high-liquidity stocks (Fast Scan)
    tickers = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS', 
               'SBIN.NS', 'BHARTIARTL.NS', 'ITC.NS', 'L&T.NS', 'BAJFINANCE.NS',
               'AXISBANK.NS', 'KOTAKBANK.NS', 'TATAMOTORS.NS', 'SUNPHARMA.NS', 
               'MARUTI.NS', 'NTPC.NS', 'TATASTEEL.NS', 'POWERGRID.NS', 'ASIANPAINT.NS', 'M&M.NS']
    
    df_list = []
    for ticker in tickers:
        try:
            df = yf.download(ticker, period="2y", progress=False)
            if not df.empty:
                df = df.reset_index()
                df['Ticker'] = ticker
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [col[0] for col in df.columns]
                df_list.append(df)
        except:
            pass
    return pd.concat(df_list, ignore_index=True)

def process_and_predict(df):
    df = df.sort_values(by=['Ticker', 'Date']).copy()
    
    # Feature Engineering
    df['Return_1M'] = df.groupby('Ticker')['Close'].pct_change(21)
    df['Return_3M'] = df.groupby('Ticker')['Close'].pct_change(63)
    df['Return_6M'] = df.groupby('Ticker')['Close'].pct_change(126)
    df['SMA_200'] = df.groupby('Ticker')['Close'].transform(lambda x: x.rolling(200).mean())
    df['Dist_SMA200'] = (df['Close'] / df['SMA_200']) - 1
    
    df['Fwd_Return_21D'] = df.groupby('Ticker')['Close'].shift(-21) / df['Close'] - 1
    df['Target'] = np.where(df['Fwd_Return_21D'] > 0.02, 1, 0)
    
    features = ['Return_1M', 'Return_3M', 'Return_6M', 'Dist_SMA200']
    
    train_df = df.dropna(subset=['Target'] + features)
    latest_df = df.groupby('Ticker').tail(1).dropna(subset=features)
    
    # Model Training
    X_train = train_df[features]
    y_train = train_df['Target']
    model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, random_state=42, verbose=-1)
    model.fit(X_train, y_train)
    
    # Prediction & Sizing
    latest_df['Alpha_Probability'] = model.predict_proba(latest_df[features])[:, 1]
    
    # Half-Kelly for Safety
    b_odds = 2.5
    p = latest_df['Alpha_Probability']
    q = 1 - p
    latest_df['Kelly_Fraction'] = ((b_odds * p) - q) / b_odds
    latest_df['Half_Kelly_Alloc_%'] = (latest_df['Kelly_Fraction'] / 2) * 100
    
    results = latest_df[['Ticker', 'Close', 'Alpha_Probability', 'Half_Kelly_Alloc_%']].copy()
    results['Alpha_Probability'] = (results['Alpha_Probability'] * 100).round(2)
    results['Half_Kelly_Alloc_%'] = np.where(results['Half_Kelly_Alloc_%'] < 0, 0, results['Half_Kelly_Alloc_%']).round(2)
    
    return results.sort_values(by='Alpha_Probability', ascending=False)

# 3. UI BUTTON
if st.button("Run Market Scan"):
    with st.spinner("Analyzing market data..."):
        raw_data = get_stock_data()
        predictions = process_and_predict(raw_data)
        
        top_picks = predictions[predictions['Alpha_Probability'] > 65.0]
        
        if not top_picks.empty:
            st.subheader("🔥 Top High-Probability Stocks")
            st.dataframe(top_picks.style.format({"Close": "₹{:.2f}", "Alpha_Probability": "{:.2f}%", "Half_Kelly_Alloc_%": "{:.2f}%"}))
            st.success("Rule: Use Limit Orders near 'Close' price & set a strict 7% GTT Stop-Loss.")
        else:
            st.warning("No stocks met the >65% probability threshold today. Sit tight!")
