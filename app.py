import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import lightgbm as lgb
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(page_title="AlphaEngine | Quant Scanner", layout="wide")
st.title("🚀 AlphaEngine (50-EMA & MTF Optimized)")
st.markdown("---")

# 1. MARKET REGIME CHECK (FAST 50-EMA)
@st.cache_data(ttl=3600)
def check_market_regime():
    nifty = yf.download("^NSEI", period="1y", progress=False)
    # yfinance version fix
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
    st.stop() # Execution completely stops here
else:
    st.success(f"🟢 MARKET REGIME: BULLISH (Nifty at {close:.2f} is above 50-EMA {ema:.2f})")
    st.info("System is ready to scan the entire Nifty 500 universe for momentum leaders...")

# 2. SCANNER LOGIC (Executes only if Bullish)
@st.cache_data(ttl=3600)
def get_stock_data():
    # Fetch Live Nifty 500 List from NSE
    try:
        url = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
        nifty500 = pd.read_csv(url)
        tickers = [sym + ".NS" for sym in nifty500['Symbol'].tolist()]
    except:
        # Emergency Fallback (In case NSE website is slow)
        tickers = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'ITC.NS', 'L&T.NS', 'BAJFINANCE.NS']
    
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
    def process_and_predict(df):
    # SAFETY CHECK 1: Agar data hi nahi aaya
    if df.empty:
        return pd.DataFrame()
        
    df = df.sort_values(by=['Ticker', 'Date']).copy()
    
    # SAFETY CHECK 2: Yahoo Finance ke khali data (NaN) ko purane data se fill karna
    df = df.ffill()
    
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
    
    # SAFETY CHECK 3: Agar filter hone ke baad data zero bache, toh crash mat ho
    if train_df.empty or latest_df.empty:
        return pd.DataFrame()
        
    # Model Training
    X_train = train_df[features]
    y_train = train_df['Target']
    
    # SAFETY CHECK 4: Agar sabhi stocks target hit na karein
    if len(y_train.unique()) < 2:
        return pd.DataFrame()

    model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, random_state=42, verbose=-1)
    model.fit(X_train, y_train)
    
    # Prediction & Sizing
    latest_df['Alpha_Probability'] = model.predict_proba(latest_df[features])[:, 1]
    
    # Kelly Criterion
    b_odds = 2.5
    p = latest_df['Alpha_Probability']
    q = 1 - p
    latest_df['Kelly_Fraction'] = ((b_odds * p) - q) / b_odds
    
    # Capital Calculations
    latest_df['Base_Kelly_%'] = (latest_df['Kelly_Fraction'] / 2) * 100
    latest_df['Base_Kelly_%'] = np.where(latest_df['Base_Kelly_%'] < 0, 0, latest_df['Base_Kelly_%'])
    
    latest_df['MTF_Alloc_(1.5x)_%'] = latest_df['Base_Kelly_%'] * 1.5
    
    results = latest_df[['Ticker', 'Close', 'Alpha_Probability', 'Base_Kelly_%', 'MTF_Alloc_(1.5x)_%']].copy()
    results['Alpha_Probability'] = (results['Alpha_Probability'] * 100).round(2)
    results['Base_Kelly_%'] = results['Base_Kelly_%'].round(2)
    results['MTF_Alloc_(1.5x)_%'] = results['MTF_Alloc_(1.5x)_%'].round(2)
    
    return results.sort_values(by='Alpha_Probability', ascending=False)
# 3. UI BUTTON
if st.button("Run Nifty 500 Scan"):
    # Since 500 stocks take time to download, a spinner shows the progress
    with st.spinner("Downloading and analyzing Nifty 500 stocks (This will take 2-3 minutes)..."):
        raw_data = get_stock_data()
        predictions = process_and_predict(raw_data)
        
        top_picks = predictions[predictions['Alpha_Probability'] > 65.0]
        
        if not top_picks.empty:
            st.subheader("🔥 Top High-Probability Stocks")
            st.dataframe(
                top_picks.style.format({
                    "Close": "₹{:.2f}", 
                    "Alpha_Probability": "{:.2f}%", 
                    "Base_Kelly_%": "{:.2f}%",
                    "MTF_Alloc_(1.5x)_%": "{:.2f}%"
                }),
                use_container_width=True
            )
            st.success("Rule: Use Limit Orders near 'Close' price & set a strict 7% GTT Stop-Loss.")
        else:
            st.warning("No stocks met the >65% probability threshold today. Sit tight!")
