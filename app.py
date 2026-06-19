import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import lightgbm as lgb
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(page_title="AlphaEngine | Master Quant Scanner", layout="wide")
st.title("🚀 AlphaEngine (The Master Edition)")
st.markdown("---")

# ==========================================
# 1. MARKET REGIME CHECK (FAST 50-EMA)
# ==========================================
@st.cache_data(ttl=3600)
def check_market_regime():
    nifty = yf.download("^NSEI", period="1y", progress=False)
    if isinstance(nifty.columns, pd.MultiIndex):
        nifty.columns = [col[0] for col in nifty.columns]
        
    nifty['EMA_50'] = nifty['Close'].ewm(span=50, adjust=False).mean()
    current_close = nifty['Close'].iloc[-1]
    current_ema = nifty['EMA_50'].iloc[-1]
    
    regime = "BULLISH" if current_close > current_ema else "BEARISH"
    return regime, current_close, current_ema

try:
    regime, close, ema = check_market_regime()
    if regime == "BEARISH":
        st.error(f"🚨 MARKET REGIME: BEARISH (Nifty at {close:.2f} is below 50-EMA {ema:.2f})")
        st.warning("🛡️ SYSTEM IN CASH MODE: Market is highly unstable. Capital protection active. No trades today.")
        st.stop() # Stops execution if market is bad
    else:
        st.success(f"🟢 MARKET REGIME: BULLISH (Nifty at {close:.2f} is above 50-EMA {ema:.2f})")
        st.info("System is ready. All safety locks (No Falling Knives, Data Error Handling) are active.")
except Exception as e:
    st.error("Market data fetch error from servers. Please refresh the page.")
    st.stop()

# ==========================================
# 2. DATA FETCHING (NIFTY 500)
# ==========================================
@st.cache_data(ttl=3600)
def get_stock_data():
    try:
        url = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
        nifty500 = pd.read_csv(url)
        tickers = [sym + ".NS" for sym in nifty500['Symbol'].tolist()]
    except:
        tickers = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS', 'SBIN.NS']
    
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
            
    if not df_list:
        return pd.DataFrame()
    return pd.concat(df_list, ignore_index=True)

# ==========================================
# 3. CORE QUANT LOGIC & PREDICTION
# ==========================================
def process_and_predict(df):
    if df.empty:
        return pd.DataFrame()
        
    df = df.sort_values(by=['Ticker', 'Date']).copy()
    df = df.ffill() # Fixes Yahoo Finance missing data bugs
    
    # Feature Engineering
    df['Return_1M'] = df.groupby('Ticker')['Close'].pct_change(21)
    df['Return_3M'] = df.groupby('Ticker')['Close'].pct_change(63)
    df['Return_6M'] = df.groupby('Ticker')['Close'].pct_change(126)
    df['SMA_200'] = df.groupby('Ticker')['Close'].transform(lambda x: x.rolling(200).mean())
    df['Dist_SMA200'] = (df['Close'] / df['SMA_200']) - 1
    
    # Target for AI (2% in 21 Days)
    df['Fwd_Return_21D'] = df.groupby('Ticker')['Close'].shift(-21) / df['Close'] - 1
    df['Target'] = np.where(df['Fwd_Return_21D'] > 0.02, 1, 0)
    
    features = ['Return_1M', 'Return_3M', 'Return_6M', 'Dist_SMA200']
    
    train_df = df.dropna(subset=['Target'] + features)
    latest_df = df.groupby('Ticker').tail(1).dropna(subset=features)
    
    # 🛑 STRICT MOMENTUM FILTER (REMOVES FALLING KNIVES LIKE ZENSARTECH)
    latest_df = latest_df[(latest_df['Dist_SMA200'] > 0) & (latest_df['Return_1M'] > 0)]
    
    if train_df.empty or latest_df.empty or len(train_df['Target'].unique()) < 2:
        return pd.DataFrame()
        
    # Model Training
    X_train = train_df[features]
    y_train = train_df['Target']
    model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, random_state=42, verbose=-1)
    model.fit(X_train, y_train)
    
    # Prediction
    latest_df['Alpha_Probability'] = model.predict_proba(latest_df[features])[:, 1]
    
    # Sizing (Kelly & MTF 1.5x)
    b_odds = 2.5
    p = latest_df['Alpha_Probability']
    q = 1 - p
    latest_df['Kelly_Fraction'] = ((b_odds * p) - q) / b_odds
    
    latest_df['Base_Kelly_%'] = (latest_df['Kelly_Fraction'] / 2) * 100
    latest_df['Base_Kelly_%'] = np.where(latest_df['Base_Kelly_%'] < 0, 0, latest_df['Base_Kelly_%'])
    latest_df['MTF_Alloc_(1.5x)_%'] = latest_df['Base_Kelly_%'] * 1.5
    
    results = latest_df[['Ticker', 'Close', 'Alpha_Probability', 'Base_Kelly_%', 'MTF_Alloc_(1.5x)_%']].copy()
    results['Alpha_Probability'] = (results['Alpha_Probability'] * 100).round(2)
    results['Base_Kelly_%'] = results['Base_Kelly_%'].round(2)
    results['MTF_Alloc_(1.5x)_%'] = results['MTF_Alloc_(1.5x)_%'].round(2)
    
    return results.sort_values(by='Alpha_Probability', ascending=False)

# ==========================================
# 4. USER INTERFACE
# ==========================================
if st.button("Run Master Scan"):
    with st.spinner("Crunching Nifty 500 data & Filtering falling knives... (Takes 2-3 mins)"):
        raw_data = get_stock_data()
        predictions = process_and_predict(raw_data)
        
        if predictions.empty:
            st.error("Market data processing failed. The market might be closed or data servers are down.")
        else:
            # Final output threshold
            top_picks = predictions[predictions['Alpha_Probability'] > 65.0]
            
            if not top_picks.empty:
                st.subheader("🔥 Ultra-Filtered High-Probability Stocks")
                st.dataframe(
                    top_picks.style.format({
                        "Close": "₹{:.2f}", 
                        "Alpha_Probability": "{:.2f}%", 
                        "Base_Kelly_%": "{:.2f}%",
                        "MTF_Alloc_(1.5x)_%": "{:.2f}%"
                    }),
                    use_container_width=True
                )
                st.success("Rule: Execute Limit Orders at 'Close' price & set a strict 7% GTT Stop-Loss in your broker.")
            else:
                st.warning("No stocks passed the strict momentum and probability checks today. Sit tight!")
