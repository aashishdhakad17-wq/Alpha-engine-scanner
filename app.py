import streamlit as st
import numpy as np
import pandas as pd
import requests
import yfinance as yf
import lightgbm as lgb
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(page_title="AlphaEngine Ultra | Rocket Scanner", layout="wide")

# ==========================================
# 1. MARKET REGIME & MTF ENGINE
# ==========================================
def get_market_regime():
    """Nifty 50 का 200 SMA देखकर मार्केट का मूड तय करना"""
    try:
        nifty = yf.download("^NSEI", start=(datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'), progress=False)
        if not nifty.empty:
            nifty.columns = [col[0] if isinstance(col, tuple) else col for col in nifty.columns]
            nifty['SMA200'] = nifty['Close'].rolling(200).mean()
            latest_close = float(nifty['Close'].iloc[-1])
            sma200 = float(nifty['SMA200'].iloc[-1])
            
            if latest_close > sma200:
                return "BULLISH 📈", "HIGH RISK ALLOWED (MTF 1.5x on Top Picks)", "#00FF00"
            else:
                return "BEARISH 📉", "LOW RISK / CASH ONLY (Strictly No MTF, Keep 50% Cash)", "#FF0000"
    except:
        pass
    return "UNKNOWN 🔍", "MODERATE RISK (No Leverage, Standard Allocation)", "#FFFF00"

# ==========================================
# 2. DYNAMIC NIFTY 500 SCANNER
# ==========================================
@st.cache_data(ttl=3600)
def fetch_live_nifty500():
    url = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(url, headers=headers, timeout=15)
        from io import StringIO
        df = pd.read_csv(StringIO(response.text))
        tickers = [str(symbol) + ".NS" for symbol in df['Symbol'].tolist()]
        bad_tickers = ["L&TFH.NS", "PPAP.NS"]
        return [t for t in tickers if t not in bad_tickers]
    except:
        return ["SUZLON.NS", "IRFC.NS", "RVNL.NS", "ZOMATO.NS", "TRENT.NS", "HFCL.NS", "WOCKPHARMA.NS"]

def run_stage1_rocket_filter(tickers):
    start_dt = (datetime.now() - timedelta(days=45)).strftime('%Y-%m-%d')
    end_dt = datetime.now().strftime('%Y-%m-%d')
    
    bulk_data = yf.download(tickers, start=start_dt, end=end_dt, progress=False, group_by='ticker', threads=True)
    
    valid_candidates = []
    for ticker in tickers:
        try:
            if ticker in bulk_data and not bulk_data[ticker].dropna().empty:
                df = bulk_data[ticker].dropna()
                if len(df) < 15: continue
                
                month_return = (float(df['Close'].iloc[-1]) / float(df['Close'].iloc[0])) - 1
                avg_vol = df['Volume'].mean()
                
                # कड़ा नियम: 1 महीने में कम से कम 10% का धमाका और भारी वॉल्यूम
                if month_return > 0.10 and df['Volume'].iloc[-1] > (avg_vol * 0.8):
                    valid_candidates.append({
                        'Ticker': ticker,
                        'Recent_Return': month_return * 100,
                        'Close': float(df['Close'].iloc[-1])
                    })
        except:
            continue
    
    f_df = pd.DataFrame(valid_candidates)
    if not f_df.empty:
        return f_df.sort_values(by='Recent_Return', ascending=False).head(25)['Ticker'].tolist()
    return tickers[:10]

# ==========================================
# 3. PIPELINE ENGINES
# ==========================================
class DataEngine:
    def __init__(self, tickers, start_date="2020-01-01"):
        self.tickers = tickers
        self.start_date = start_date

    def fetch_data(self):
        all_dfs = []
        end_date = datetime.now().strftime('%Y-%m-%d')
        for ticker in self.tickers:
            try:
                df = yf.download(ticker, start=self.start_date, end=end_date, progress=False)
                if not df.empty:
                    df = df.reset_index()
                    df['Ticker'] = ticker
                    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
                    all_dfs.append(df)
            except:
                continue
        return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

class FeatureEngineer:
    def build_features(self, raw_df, is_live=False):
        df = raw_df.sort_values(by=['Ticker', 'Date']).copy()
        df.set_index(['Ticker', 'Date'], inplace=True)
        
        df['Return_1M'] = df.groupby(level=0)['Close'].pct_change(21)
        df['Return_3M'] = df.groupby(level=0)['Close'].pct_change(63)
        df['Return_6M'] = df.groupby(level=0)['Close'].pct_change(126)
        
        high, low = df['High'], df['Low']
        close_prev = df.groupby(level=0)['Close'].shift(1)
        tr = pd.concat([high - low, (high - close_prev).abs(), (low - close_prev).abs()], axis=1).max(axis=1)
        df['ATR_14'] = tr.groupby(level=0).transform(lambda x: x.rolling(14).mean())
        df['Norm_ATR'] = df['ATR_14'] / df['Close']
        
        df['Dist_SMA200'] = df['Close'] / df.groupby(level=0)['Close'].transform(lambda x: x.rolling(200).mean()) - 1
        df['High_52W'] = df.groupby(level=0)['High'].transform(lambda x: x.rolling(252).max())
        df['Dist_52W_High'] = (df['Close'] / df['High_52W']) - 1
        
        df = df.reset_index()
        df['Alpha_Score'] = (df['Return_1M'] * 0.4) + (df['Return_3M'] * 0.2) + (df['Return_6M'] * 0.4)
        df['RS_Rating'] = df.groupby('Date')['Alpha_Score'].rank(pct=True) * 99
        
        feature_cols = ['Return_1M', 'Return_3M', 'Return_6M', 'Norm_ATR', 'Dist_SMA200', 'Dist_52W_High', 'RS_Rating']
        
        if is_live:
            return df.dropna(subset=feature_cols).copy(), feature_cols
            
        df['Fwd_Return'] = df.groupby('Ticker')['Close'].shift(-44) / df['Close'] - 1
        df['Top_Quartile_Threshold'] = df.groupby('Date')['Fwd_Return'].transform(lambda x: x.quantile(0.75))
        # सुपर टारगेट: अगले 2 महीने में न्यूनतम 21% का रिटर्न (Odds Ratio = 3)
        df['Target'] = np.where((df['Fwd_Return'] >= 0.21) & (df['Fwd_Return'] >= df['Top_Quartile_Threshold']), 1, 0)
        
        return df.dropna(subset=feature_cols + ['Target']).copy(), feature_cols

class ModelEngine:
    def __init__(self, feature_cols):
        self.feature_cols = feature_cols
        self.model = None

    def train_model(self, data_df):
        X = data_df[self.feature_cols]
        y = data_df['Target']
        self.model = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.04, random_state=42, class_weight='balanced', verbose=-1)
        self.model.fit(X, y)
        return self.model

# ==========================================
# 4. DASHBOARD USER INTERFACE
# ==========================================
st.title("🚀 AlphaEngine Ultra: Institutional Rocket Stock Scanner")
st.markdown("यह सिस्टम लाइव मार्केट रेजिम और केली क्राइटेरियन पोजीशन साइजिंग के साथ टॉप अल्फा लीडर्स को स्कैन करता है।")
st.write("---")

# लाइव मार्केट रेजिम कार्ड दिखाना
regime, mtf_guide, color = get_market_regime()
col1, col2 = st.columns(2)
with col1:
    st.markdown(f"### Current Market Regime: <span style='color:{color}'>{regime}</span>", unsafe_allow_html=True)
with col2:
    st.markdown(f"### MTF Risk Rule: <span style='color:{color}'>{mtf_guide}</span>", unsafe_allow_html=True)

st.write("---")

if 'trained_model' not in st.session_state:
    st.session_state.trained_model = None
if 'feature_cols' not in st.session_state:
    st.session_state.feature_cols = None

tab1, tab2 = st.tabs(["🦅 Run High-Probability Rocket Scan", "📊 Train AI Brain"])

with tab1:
    st.header("🎯 Live Rocket Stock Alpha Selection")
    if st.button("🔥 Scan Entire Market Universe"):
        with st.spinner("🔄 Nifty 500 से सुपर-स्पीड 'चीता' स्टॉक्स चुने जा रहे हैं..."):
            tickers = fetch_live_nifty500()
            rocket_universe = run_stage1_rocket_filter(tickers)
            
        with st.spinner("🔄 एआई एल्गोरिदम लाइव प्रेडिक्शन की गणना कर रहा है..."):
            live_start = (datetime.now() - timedelta(days=400)).strftime('%Y-%m-%d')
            dp = DataEngine(tickers=rocket_universe, start_date=live_start)
            raw_data = dp.fetch_data()
            
            if raw_data.empty:
                st.error("डेटा उपलब्ध नहीं हो पाया।")
            else:
                fe = FeatureEngineer()
                live_df, f_cols = fe.build_features(raw_data, is_live=True)
                
                latest_date = live_df['Date'].max()
                current_slice = live_df[live_df['Date'] == latest_date].copy()
                
                if st.session_state.trained_model is not None:
                    current_slice['Alpha_Probability'] = st.session_state.trained_model.predict_proba(current_slice[st.session_state.feature_cols])[:, 1]
                    sort_col = 'Alpha_Probability'
                else:
                    current_slice['Alpha_Probability'] = current_slice['RS_Rating'] / 100
                    sort_col = 'RS_Rating'
                
                # केली क्राइटेरियन कैलकुलेशन (Odds ratio b = 3, target 21% / SL 7%)
                b_odds = 3.0
                p_win = current_slice['Alpha_Probability']
                # हाफ केली फॉर्मूला सुरक्षा के लिए
                current_slice['Kelly_Allocation'] = (p_win - ((1.0 - p_win) / b_odds)) * 0.5
                current_slice['Kelly_Allocation'] = current_slice['Kelly_Allocation'].apply(lambda x: max(0.0, x))
                
                output_df = current_slice[['Ticker', 'Alpha_Probability', 'Kelly_Allocation', 'Close', 'RS_Rating', 'Dist_52W_High']].sort_values(by=sort_col, ascending=False).head(5)
                
                # केवल उन स्टॉक्स को दिखाना जो वास्तव में मजबूत हैं
                output_df = output_df[output_df['Kelly_Allocation'] > 0]
                
                if output_df.empty:
                    st.warning("🚨 बाज़ार में इस समय कोई भी स्टॉक केली क्राइटेरियन के कड़े रिस्क मानदंडों को पूरा नहीं कर रहा है। कैश में बैठें।")
                else:
                    output_df['Close'] = output_df['Close'].map('₹{:,.2f}'.format)
                    output_df['Alpha_Probability'] = output_df['Alpha_Probability'].map('{:.2%}'.format)
                    output_df['Kelly_Allocation'] = output_df['Kelly_Allocation'].map('{:.2%}'.format)
                    output_df['RS_Rating'] = output_df['RS_Rating'].map('{:.2f}'.format)
                    output_df['Dist_52W_High'] = output_df['Dist_52W_High'].map('{:.2%}'.format)
                    
                    st.metric(label="Scan Output Date", value=str(latest_date.date()))
                    st.dataframe(output_df, use_container_width=True)

with tab2:
    st.header("📊 Model Optimization & Training")
    if st.button("🚀 Train Model on High-Alpha Data"):
        with st.spinner("🔄 ऐतिहासिक पैटर्न्स पर एआई इंजन को ऑप्टिमाइज़ किया जा रहा है..."):
            tickers = fetch_live_nifty500()
            rocket_universe = run_stage1_rocket_filter(tickers)
            
            dp = DataEngine(tickers=rocket_universe, start_date="2021-01-01")
            raw_data = dp.fetch_data()
            
            fe = FeatureEngineer()
            processed_data, feature_cols = fe.build_features(raw_data)
            
            ml_core = ModelEngine(feature_cols=feature_cols)
            ml_core.train_model(processed_data)
            
            st.session_state.trained_model = ml_core.model
            st.session_state.feature_cols = feature_cols
            st.success("🎯 नया रॉकेट एआई मॉडल पूरी तरह ट्रेन हो चुका है! अब टैब 1 पर जाकर लाइव स्कैन करें।")
