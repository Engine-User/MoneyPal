import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.stats import norm
import requests
import json
import time
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")

# ──────────────────────── PAGE CONFIG ────────────────────────
st.set_page_config(
    page_title="Asset Analyser",
    layout="wide",
    page_icon="💲",
    initial_sidebar_state="expanded",
)

# ──────────────────────── CONSTANTS / INSTRUMENT MAP ────────────────────────
INSTRUMENT_MAP = {
    "Indian Indices": {
        "NIFTY 50": "^NSEI",
        "SENSEX": "^BSESN",
        "BANK NIFTY": "^NSEBANK",
        "NIFTY IT": "^CNXIT",
        "NIFTY MIDCAP 50": "^NSEMDCP50",
    },
    "Indian Stocks": {
        "Reliance": "RELIANCE.NS",
        "Bandhan Bank": "BANDHANBNK.NS",
        "Kotak Bank": "KOTAKBANK.NS",
        "HDFC Bank": "HDFCBANK.NS",
        "TCS": "TCS.NS",
        "Infosys": "INFY.NS",
        "ICICI Bank": "ICICIBANK.NS",
        "SBI": "SBIN.NS",
        "Bharti Airtel": "BHARTIARTL.NS",
        "ITC": "ITC.NS",
        "L&T": "LT.NS"

    },
    "US Stocks": {
        "Apple": "AAPL",
        "Google": "GOOGL",
        "Microsoft": "MSFT",
        "Nvidia": "NVDA",
        "Amazon": "AMZN",
        "Tesla": "TSLA",
        "Meta": "META",
    },
    "Commodities": {
        "Gold": "GC=F",
        "Silver": "SI=F",
        "Crude Oil": "CL=F",
        "Natural Gas": "NG=F",
        "Copper": "HG=F",
    },
    "Crypto": {
        "Bitcoin": "BTC-USD",
        "Ethereum": "ETH-USD",
        "Solana": "SOL-USD",
    },
    "International Indices": {
        "S&P 500": "^GSPC",
        "Dow Jones": "^DJI",
        "NASDAQ": "^IXIC",
        "FTSE 100": "^FTSE",
        "Nikkei 225": "^N225",
    },
}

TICKER_SYMBOLS = [
    ("NSE NIFTY 50", "^NSEI"),
    ("BSE SENSEX", "^BSESN"),
    ("BANK NIFTY", "^NSEBANK"),
    ("NIFTY IT", "^CNXIT"),
    ("DOW", "^DJI"),
    ("S&P 500", "^GSPC"),
    ("NASDAQ", "^IXIC"),
    ("CRUDE OIL", "CL=F"),
    ("GOLD", "GC=F"),
    ("SILVER", "SI=F"),
]

PERIOD_OPTIONS = {
    "1 Month": "1mo",
    "3 Months": "3mo",
    "6 Months": "6mo",
    "1 Year": "1y",
    "2 Years": "2y",
    "5 Years": "5y",
    "10 Years": "10y",
    "Max": "max",
}

INDICATOR_GROUPS = {
    "Trend": [
        "SMA 20", "SMA 50", "SMA 100", "SMA 200",
        "EMA 9", "EMA 12", "EMA 20", "EMA 26", "EMA 50",
        "DEMA 20", "TEMA 20", "WMA 20",
        "Supertrend", "Ichimoku", "VWAP",
        "Parabolic SAR", "HMA 20",
    ],
    "Momentum": [
        "RSI 14", "MACD", "Stochastic", "Stochastic RSI",
        "Williams %R", "CCI 20", "ROC 12", "MFI 14",
        "Ultimate Oscillator", "Awesome Oscillator",
        "TSI", "PPO",
    ],
    "Volatility": [
        "Bollinger Bands", "ATR 14", "Keltner Channel",
        "Donchian Channel", "Chaikin Volatility",
        "Historical Volatility", "Normalized ATR",
    ],
    "Volume": [
        "OBV", "VWAP", "AD Line", "CMF 20",
        "Force Index", "EFI", "Volume SMA 20",
    ],
    "Overlap": [
        "Pivot Points", "Fibonacci Retracement",
    ],
}


# ──────────────────────── GLOBAL HTML/CSS ────────────────────────
def get_currency_symbol(sym: str) -> str:
    indian_indices = ["^NSEI", "^BSESN", "^NSEBANK", "^CNXIT", "^NSEMDCP50", "^CRSLDX"]
    if sym.endswith(".NS") or sym.endswith(".BO") or sym in indian_indices:
        return "₹"
    elif sym.endswith(".L"): return "£"
    elif sym.endswith(".DE"): return "€"
    elif sym in ["USDJPY=X"]: return "¥"
    if sym in ["NIFTY 50", "NIFTY IT", "SENSEX", "BANK NIFTY", "NIFTY MIDCAP 50"]:
        return "₹"
    return "$"

def inject_css():
    st.markdown(
        """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300..700&display=swap');

    /* Apply custom font globally but explicitly exempt Streamlit's material glyph ligatures */
    *:not(.stIcon):not(.material-symbols-rounded):not([translate="no"]):not([class*="icon"]):not([class*="Icon"]) { 
        font-family: 'Space Grotesk', sans-serif !important; 
    }
    .stApp {
        background: linear-gradient(170deg, #0a0f1e 0%, ##000000 35%, #131c31 65%, #0f172a 100%);
    }

    /* ──── Sidebar ──── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1321 0%, #151d2e 100%);
        border-right: 1px solid rgba(6, 182, 212, 0.15);
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #06b6d4 !important;
    }

    /* ──── Cards ──── */
    .glass-card {
        background: linear-gradient(145deg, rgba(30,41,59,0.85), rgba(15,23,42,0.95));
        border: 1px solid rgba(6,182,212,0.12);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 16px;
        backdrop-filter: blur(20px);
        box-shadow: 0 8px 32px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.04);
        transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .glass-card:hover {
        border-color: rgba(6,182,212,0.3);
        box-shadow: 0 16px 48px rgba(6,182,212,0.08), 0 8px 32px rgba(0,0,0,0.5);
        transform: translateY(-2px);
    }

    /* ──── Section Headers ──── */
    .section-title {
        font-size: 1.75rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin: 40px 0 8px;
        background: linear-gradient(135deg, #FDFD96, #FDFD96, #FFFF33, #FFD700, #D2B48C,#FAFAD2, #FFD700, #FFA500, #FFFF00, #F8F8FF, #FFFFFF, #F5F5F5, #F8F8FF );
        background-size: 250% 250%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: shimmer 6s ease-in-out infinite;
    }
    @keyframes shimmer {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    .section-subtitle {
        font-size: 0.92rem;
        color: #94a3b8;
        margin-bottom: 24px;
        font-weight: 400;
    }

    /* ──── Metrics ──── */
    .metric-box {
        background: linear-gradient(145deg, #1e293b, #0f172a);
        border: 1px solid rgba(6,182,212,0.1);
        border-radius: 14px;
        padding: 18px 16px;
        text-align: center;
        box-shadow: 0 4px 16px rgba(0,0,0,0.3);
        transition: all 0.3s ease;
    }
    .metric-box:hover {
        border-color: rgba(6,182,212,0.3);
        transform: translateY(-3px);
        box-shadow: 0 8px 28px rgba(6,182,212,0.1);
    }
    .metric-label {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #64748b;
        font-weight: 600;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.35rem;
        font-weight: 700;
        color: #e2e8f0;
    }
    .metric-delta-pos { color: #22c55e; font-size: 0.85rem; font-weight: 600; }
    .metric-delta-neg { color: #ef4444; font-size: 0.85rem; font-weight: 600; }

    /* ──── Live Ticker Bar ──── */
    .ticker-bar {
        position: fixed;
        top: 50px;
        left: 0;
        right: 0;
        z-index: 999999;
        background: linear-gradient(90deg, #0f172a 0%, #000000 60%, #000000 50%, #1e293b 75%, #0f172a 100%);
        border-bottom: 2px solid rgba(6,182,212,0.8);
        padding: 12.5px 0;
        overflow: hidden;
        height: 115px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.9), inset 0 2px 15px rgba(255,255,255,0.1);
        display: flex;
        align-items: center;
    }
    .ticker-content {
        display: flex;
        animation: scroll-ticker 40s linear infinite;
        white-space: nowrap;
        align-items: center;
        height: 100%;
        will-change: transform;
    }
    .ticker-content:hover {
        animation-play-state: paused;
    }
    @keyframes scroll-ticker {
        0% { transform: translate3d(-50%, 0, 0); }
        100% { transform: translate3d(0, 0, 0); }
    }
    .ticker-item {
        display: inline-flex;
        align-items: center;
        padding: 0 30px;
    }

    /* ──── Streamlit component overrides ──── */
    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, #1e293b, #0f172a);
        border: 1px solid rgba(6,182,212,0.08);
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.25);
    }
    div[data-testid="stMetric"]:hover {
        border-color: rgba(6,182,212,0.25);
    }
    div[data-testid="stMetric"] label { color: #64748b !important; font-weight: 600 !important; text-transform: uppercase; font-size: 0.7rem !important; letter-spacing: 0.06em; }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: #e2e8f0 !important; font-weight: 700 !important; }

    .stSelectbox > div > div, .stMultiSelect > div > div, .stNumberInput > div > div > input {
        background: linear-gradient(145deg, rgba(30,41,75,0.7), rgba(15,23,42,0.9)) !important;
        color: #e2e8f0 !important;
        border: 1px solid rgba(6,182,212,0.25) !important;
        border-radius: 10px !important;
        box-shadow: inset 2px 2px 6px rgba(0,0,0,0.4) !important;
    }

    /* Metallic tags/pills for multiselect */
    div[data-testid="stMultiSelect"] span[data-baseweb="tag"] {
        background: linear-gradient(145deg, #1e293b, #0f172a) !important;
        border: 1px solid rgba(6,182,212,0.4) !important;
        box-shadow: 2px 2px 6px rgba(0,0,0,0.5), inset 1px 1px 3px rgba(255,255,255,0.1) !important;
        color: #e2e8f0 !important;
        border-radius: 8px !important;
        padding: 4px 10px !important;
        font-weight: 600 !important;
    }
    div[data-testid="stMultiSelect"] span[data-baseweb="tag"] span[title] {
        color: #06b6d4 !important;
        font-weight: 700 !important;
        letter-spacing: 0.02em;
    }
    div[data-testid="stMultiSelect"] span[data-baseweb="tag"] span[role="presentation"] svg {
        fill: #94a3b8 !important;
    }
    div[data-testid="stMultiSelect"] span[data-baseweb="tag"] span[role="presentation"]:hover svg {
        fill: #ef4444 !important;
    }

    .sidebar-brand-block {
        background: linear-gradient(150deg, rgba(30,41,59,0.9), rgba(15,23,42,0.95));
        border: 1px solid rgba(6,182,212,0.4);
        border-top: 1px solid rgba(255,255,255,0.2);
        border-bottom: 2px solid rgba(6,182,212,0.8);
        border-radius: 12px;
        padding: 20px 15px;
        margin-bottom: 25px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.8), 0 0 15px rgba(6,182,212,0.2), inset 0 2px 10px rgba(255,255,255,0.1);
        text-align: center;
    }

    .ticker-card {
        display: inline-flex;
        flex-direction: column;
        justify-content: center;
        background: linear-gradient(145deg, rgba(30,41,59,0.9), rgba(15,23,42,1));
        border: 1px solid rgba(6,182,212,0.5);
        border-top: 1px solid rgba(255,255,255,0.3);
        border-bottom: 2px solid rgba(6,182,212,0.9);
        border-radius: 12px;
        padding: 12px 20px;
        margin: 0 12px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.9), 0 0 15px rgba(6,182,212,0.15), inset 0 2px 5px rgba(255,255,255,0.1);
        min-width: 170px;
        text-align: center;
        backdrop-filter: blur(12px);
    }

    .stButton > button {
        background: linear-gradient(135deg, #06b6d4, #0891b2) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        padding: 8px 24px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 14px rgba(6,182,212,0.25) !important;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #0891b2, #0e7490) !important;
        box-shadow: 0 6px 20px rgba(6,182,212,0.35) !important;
        transform: translateY(-1px) !important;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid rgba(6,182,212,0.1);
        border-radius: 12px;
        overflow: hidden;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: rgba(15,23,42,0.6);
        border-radius: 12px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: #94a3b8;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(6,182,212,0.15) !important;
        color: #06b6d4 !important;
    }

    .stExpander {
        background: linear-gradient(145deg, rgba(30,41,59,0.6), rgba(15,23,42,0.8));
        border: 1px solid rgba(6,182,212,0.08);
        border-radius: 12px;
    }

    /* Hide default Streamlit header */
    header[data-testid="stHeader"] {
        display: none !important;
    }

        /* ──── Page padding ──── */
    .main .block-container {
        padding-top: 200px;
        padding-bottom: 0px;
    }

    /* Push sidebar content below the fixed ticker */
    [data-testid="stSidebarUserContent"] {
        padding-top: 175px !important;
    }

    /* Gradient divider */
    .gradient-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(6,182,212,0.3), rgba(139,92,246,0.2), transparent);
        margin: 32px 0;
    }
    
""", unsafe_allow_html=True)



inject_css()


# ──────────────────────── HELPERS ────────────────────────
@st.cache_data(ttl=60, show_spinner=False)
def fetch_ticker_data(symbols: list, period: str = "5d"):
    """Fetch price data for live ticker."""
    results = []
    for name, sym in symbols:
        try:
            t = yf.Ticker(sym)
            h = t.history(period=period, auto_adjust=True)
            if len(h) >= 2:
                price = float(h["Close"].iloc[-1])
                prev = float(h["Close"].iloc[-2])
                chg = round((price - prev) / prev * 100, 2)
                results.append({"name": name, "price": price, "change": chg, "prev": prev})
        except Exception:
            pass
    return results


@st.cache_data(ttl=300, show_spinner=False)
def fetch_comparison_data(symbols: list, period: str):
    """Download OHLCV data for multiple symbols."""
    if not symbols:
        return pd.DataFrame()
    try:
        data = yf.download(symbols, period=period, auto_adjust=True, progress=False, threads=True)
        if isinstance(data.columns, pd.MultiIndex):
            close = data["Close"]
        else:
            close = data[["Close"]]
            close.columns = symbols
        return close.dropna(how="all")
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300, show_spinner=False)
def fetch_single_ohlcv(symbol: str, period: str):
    """Download OHLCV for a single symbol."""
    try:
        df = yf.download(symbol, period=period, auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df.dropna()
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=600, show_spinner=False)
def fetch_fundamentals(symbol: str):
    """Get fundamental data for a symbol."""
    try:
        return yf.Ticker(symbol).info
    except Exception:
        return {}


@st.cache_data(ttl=120, show_spinner=False)
def fetch_nse_option_chain(index_name: str):
    """Fetch NSE option chain via direct API."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
        }
        session = requests.Session()
        session.get("https://www.nseindia.com", headers=headers, timeout=10)
        url = f"https://www.nseindia.com/api/option-chain-indices?symbol={index_name}"
        resp = session.get(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


def get_reverse_map():
    """Create symbol -> display name mapping."""
    rmap = {}
    for cat, items in INSTRUMENT_MAP.items():
        for name, sym in items.items():
            rmap[sym] = name
    return rmap


REVERSE_MAP = get_reverse_map()


def sym_label(sym):
    return REVERSE_MAP.get(sym, sym.replace(".NS", "").replace("^", "").replace("=F", "").replace("-USD", ""))


def format_number(num, currency=""):
    if num is None or (isinstance(num, float) and np.isnan(num)):
        return "N/A"
    if abs(num) >= 1e12:
        return f"{currency}{num/1e12:.2f}T"
    if abs(num) >= 1e9:
        return f"{currency}{num/1e9:.2f}B"
    if abs(num) >= 1e7:
        return f"{currency}{num/1e7:.2f}Cr"
    if abs(num) >= 1e5:
        return f"{currency}{num/1e5:.2f}L"
    return f"{currency}{num:,.2f}"


# ──────────────────────── PLOTLY THEME ────────────────────────
PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Space Grotesk, sans-serif", color="#e2e8f0"),
    xaxis=dict(
        gridcolor="rgba(100,116,139,0.08)",
        zerolinecolor="rgba(100,116,139,0.1)",
    ),
    yaxis=dict(
        gridcolor="rgba(100,116,139,0.08)",
        zerolinecolor="rgba(100,116,139,0.1)",
    ),
    legend=dict(
        bgcolor="rgba(15,23,42,0.8)",
        bordercolor="rgba(6,182,212,0.15)",
        borderwidth=1,
        font=dict(size=11),
    ),
    margin=dict(l=40, r=20, t=50, b=40),
    hoverlabel=dict(
        bgcolor="#1e293b",
        bordercolor="rgba(6,182,212,0.3)",
        font=dict(color="#e2e8f0", size=12),
    ),
)

COLOR_SEQUENCE = [
    "#FFEFD5", "#FFD700", "#FFFF00", "#f59e0b", "#ef4444",
    "#ec4899", "#14b8a6", "#f97316", "#6366f1", "#84cc16",
    "#0ea5e9", "#a855f7", "#10b981", "#eab308", "#f43f5e",
]


# ──────────────────────── INDICATOR CALCULATOR ────────────────────────
def compute_indicators(df: pd.DataFrame, selected_indicators: list) -> pd.DataFrame:
    """Compute selected technical indicators and return merged DataFrame."""
    if df.empty:
        return df

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"] if "Volume" in df.columns else pd.Series(0, index=df.index)
    result = df.copy()

    def safe(fn):
        try:
            return fn()
        except Exception:
            return None

    for ind in selected_indicators:
        # ──── Trend ────
        if ind == "SMA 20":
            s = safe(lambda: ta.sma(close, 20))
            if s is not None: result["SMA_20"] = s
        elif ind == "SMA 50":
            s = safe(lambda: ta.sma(close, 50))
            if s is not None: result["SMA_50"] = s
        elif ind == "SMA 100":
            s = safe(lambda: ta.sma(close, 100))
            if s is not None: result["SMA_100"] = s
        elif ind == "SMA 200":
            s = safe(lambda: ta.sma(close, 200))
            if s is not None: result["SMA_200"] = s
        elif ind == "EMA 9":
            s = safe(lambda: ta.ema(close, 9))
            if s is not None: result["EMA_9"] = s
        elif ind == "EMA 12":
            s = safe(lambda: ta.ema(close, 12))
            if s is not None: result["EMA_12"] = s
        elif ind == "EMA 20":
            s = safe(lambda: ta.ema(close, 20))
            if s is not None: result["EMA_20"] = s
        elif ind == "EMA 26":
            s = safe(lambda: ta.ema(close, 26))
            if s is not None: result["EMA_26"] = s
        elif ind == "EMA 50":
            s = safe(lambda: ta.ema(close, 50))
            if s is not None: result["EMA_50"] = s
        elif ind == "DEMA 20":
            s = safe(lambda: ta.dema(close, 20))
            if s is not None: result["DEMA_20"] = s
        elif ind == "TEMA 20":
            s = safe(lambda: ta.tema(close, 20))
            if s is not None: result["TEMA_20"] = s
        elif ind == "WMA 20":
            s = safe(lambda: ta.wma(close, 20))
            if s is not None: result["WMA_20"] = s
        elif ind == "HMA 20":
            s = safe(lambda: ta.hma(close, 20))
            if s is not None: result["HMA_20"] = s
        elif ind == "Supertrend":
            s = safe(lambda: ta.supertrend(high, low, close, 7, 3))
            if s is not None:
                for col in s.columns:
                    result[col] = s[col]
        elif ind == "Parabolic SAR":
            s = safe(lambda: ta.psar(high, low, close))
            if s is not None:
                for col in s.columns:
                    result[col] = s[col]
        elif ind == "Ichimoku":
            s = safe(lambda: ta.ichimoku(high, low, close))
            if s is not None and isinstance(s, tuple):
                for col in s[0].columns:
                    result[col] = s[0][col]

        # ──── Momentum ────
        elif ind == "RSI 14":
            s = safe(lambda: ta.rsi(close, 14))
            if s is not None: result["RSI_14"] = s
        elif ind == "MACD":
            s = safe(lambda: ta.macd(close))
            if s is not None:
                for col in s.columns:
                    result[col] = s[col]
        elif ind == "Stochastic":
            s = safe(lambda: ta.stoch(high, low, close))
            if s is not None:
                for col in s.columns:
                    result[col] = s[col]
        elif ind == "Stochastic RSI":
            s = safe(lambda: ta.stochrsi(close))
            if s is not None:
                for col in s.columns:
                    result[col] = s[col]
        elif ind == "Williams %R":
            s = safe(lambda: ta.willr(high, low, close))
            if s is not None: result["WILLR_14"] = s
        elif ind == "CCI 20":
            s = safe(lambda: ta.cci(high, low, close, 20))
            if s is not None: result["CCI_20"] = s
        elif ind == "ROC 12":
            s = safe(lambda: ta.roc(close, 12))
            if s is not None: result["ROC_12"] = s
        elif ind == "MFI 14":
            s = safe(lambda: ta.mfi(high, low, close, volume, 14))
            if s is not None: result["MFI_14"] = s
        elif ind == "Ultimate Oscillator":
            s = safe(lambda: ta.uo(high, low, close))
            if s is not None: result["UO"] = s
        elif ind == "Awesome Oscillator":
            s = safe(lambda: ta.ao(high, low))
            if s is not None: result["AO"] = s
        elif ind == "TSI":
            s = safe(lambda: ta.tsi(close))
            if s is not None:
                if isinstance(s, pd.DataFrame):
                    for col in s.columns:
                        result[col] = s[col]
                else:
                    result["TSI"] = s
        elif ind == "PPO":
            s = safe(lambda: ta.ppo(close))
            if s is not None:
                if isinstance(s, pd.DataFrame):
                    for col in s.columns:
                        result[col] = s[col]
                else:
                    result["PPO"] = s

        # ──── Volatility ────
        elif ind == "Bollinger Bands":
            s = safe(lambda: ta.bbands(close, 20, 2))
            if s is not None:
                for col in s.columns:
                    result[col] = s[col]
        elif ind == "ATR 14":
            s = safe(lambda: ta.atr(high, low, close, 14))
            if s is not None: result["ATR_14"] = s
        elif ind == "Keltner Channel":
            s = safe(lambda: ta.kc(high, low, close))
            if s is not None:
                for col in s.columns:
                    result[col] = s[col]
        elif ind == "Donchian Channel":
            s = safe(lambda: ta.donchian(high, low, close))
            if s is not None:
                for col in s.columns:
                    result[col] = s[col]
        elif ind == "Historical Volatility":
            hv = safe(lambda: close.pct_change().rolling(20).std() * np.sqrt(252) * 100)
            if hv is not None: result["HV_20"] = hv
        elif ind == "Normalized ATR":
            atr_val = safe(lambda: ta.atr(high, low, close, 14))
            if atr_val is not None: result["NATR"] = (atr_val / close) * 100
        elif ind == "Chaikin Volatility":
            s = safe(lambda: ta.ema(high - low, 10).pct_change(10) * 100)
            if s is not None: result["Chaikin_Vol"] = s

        # ──── Volume ────
        elif ind == "OBV":
            s = safe(lambda: ta.obv(close, volume))
            if s is not None: result["OBV"] = s
        elif ind == "VWAP":
            s = safe(lambda: ta.vwap(high, low, close, volume))
            if s is not None: result["VWAP"] = s
        elif ind == "AD Line":
            s = safe(lambda: ta.ad(high, low, close, volume))
            if s is not None: result["AD"] = s
        elif ind == "CMF 20":
            s = safe(lambda: ta.cmf(high, low, close, volume, 20))
            if s is not None: result["CMF_20"] = s
        elif ind == "Force Index":
            s = safe(lambda: ta.efi(close, volume, 13))
            if s is not None: result["EFI_13"] = s
        elif ind == "EFI":
            s = safe(lambda: ta.efi(close, volume, 13))
            if s is not None: result["EFI_13"] = s
        elif ind == "Volume SMA 20":
            s = safe(lambda: ta.sma(volume, 20))
            if s is not None: result["Vol_SMA_20"] = s

    return result


# ──────────────────────── BLACK-SCHOLES ────────────────────────
def black_scholes(S, K, T, r, sigma, option_type="call"):
    if T <= 0 or sigma <= 0:
        intrinsic = max(S - K, 0) if option_type == "call" else max(K - S, 0)
        return intrinsic, 0, 0, 0, 0, 0
    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        delta = norm.cdf(d1)
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = norm.cdf(d1) - 1
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100
    rho = (K * T * np.exp(-r * T) * (norm.cdf(d2) if option_type == "call" else -norm.cdf(-d2))) / 100
    return price, delta, gamma, theta, vega, rho


# ══════════════════════════════════════════════════════════════
#                          SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    sidebar_header_html = (
        "<div class='sidebar-brand-block'>"
        "<h2 style='margin:0 0 5px 0; font-size:1.6rem; font-weight:800; background: linear-gradient(135deg, #FDFD96, #FDFD96, #FFFF33, #FFD700, #D2B48C,#FAFAD2, #FFD700, #FFA500, #FFFF00, #F8F8FF, #FFFFFF, #F5F5F5, #F8F8FF ); background-size: 250% 250%; -webkit-background-clip: text; -webkit-text-fill-color: transparent; animation: shimmer 6s ease-in-out infinite;'>Analyze Global Markets</h2>"
        "<p style='color:#94a3b8; font-size:0.8rem; font-weight:500; margin:0 0 15px 0; letter-spacing:0.02em;'>Trading Strategies & Quant Analysis</p>"
        "<div style='height:1px; background:linear-gradient(90deg,transparent,rgba(6,182,212,0.5),transparent); margin:15px 0;'></div>"
        "<p style='color:#64748b; font-size:0.65rem; font-weight:700; text-transform:uppercase; margin:0 0 5px 0; letter-spacing:0.1em;'>Designed & Developed By</p>"
        "<p style='color:#f8fafc; font-size:1.1rem; font-weight:800; margin:0 0 5px 0; text-shadow:0 2px 5px rgba(0,0,0,0.5);'>Engineer</p>"
        "<p style='color:#cbd5e1; font-size:0.75rem; font-weight:500; margin:0;'>contact: ggengineerco@gmail.com</p>"
        "</div>"
    )
    st.markdown(sidebar_header_html, unsafe_allow_html=True)

    st.markdown("Instrument Selection")
    selected_categories = st.multiselect(
        "Asset Classes",
        list(INSTRUMENT_MAP.keys()),
        default=["Indian Indices", "US Stocks", "Commodities"],
        help="Select categories to populate instrument list",
    )

    available_instruments = {}
    for cat in selected_categories:
        available_instruments.update(INSTRUMENT_MAP[cat])

    selected_names = st.multiselect(
        "Instruments",
        list(available_instruments.keys()),
        default=list(available_instruments.keys())[:6] if available_instruments else [],
        help="Select instruments to analyze",
    )
    selected_symbols = [available_instruments[n] for n in selected_names]

    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

    st.markdown("Time Period")
    period_label = st.selectbox("Select Analysis Period", list(PERIOD_OPTIONS.keys()), index=4)
    period = PERIOD_OPTIONS[period_label]

    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

    st.markdown("Technical Indicators")
    selected_indicator_list = []
    for group_name, indicators in INDICATOR_GROUPS.items():
        with st.expander(f"{'📈' if group_name=='Trend' else '⚡' if group_name=='Momentum' else '📉' if group_name=='Volatility' else '📊' if group_name=='Volume' else '🔄'} {group_name}", expanded=False):
            for ind in indicators:
                if st.checkbox(ind, value=ind in ["RSI 14", "MACD", "Bollinger Bands", "SMA 50", "EMA 20", "ATR 14"], key=f"cb_{group_name}_{ind}"):
                    if ind not in selected_indicator_list:
                        selected_indicator_list.append(ind)

    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
#    st.markdown(
#        '<p style="color:#475569; font-size:0.7rem; text-align:center;">DataGod Core v1.0 · Built with Streamlit</p>',
#        unsafe_allow_html=True,
#    )


# ══════════════════════════════════════════════════════════════
#                     MAIN APPLICATION
# ══════════════════════════════════════════════════════════════

# ──── Header (above the ticker) ────
st.markdown(
    """
<div style="text-align:center; padding: 10px 0 5px;">
    <h1 style="font-size:2.8rem; font-weight:800; letter-spacing:-0.03em;
               background: linear-gradient(135deg, #FDFD96, #FDFD96, #FFFF33, #FFD700, #D2B48C,#FAFAD2, #FFD700, #FFA500, #FFFF00, #F8F8FF, #FFFFFF, #F5F5F5, #F8F8FF );
               -webkit-background-clip: text; -webkit-text-fill-color: transparent;
               animation: shimmer 4s ease-in-out infinite;
               margin-bottom:4px;">
        💲MoneyPal💲 
    </h1>
    <p style="color:#64748b; font-size:1rem; font-weight:400;">
        Cross-Market Intelligence · Technical & Fundamental Analysis · Quant Strategies · Live Data
    </p>
</div>
""",
    unsafe_allow_html=True,
)

# ──── LIVE SCROLLING TICKER BAR (fixed full-width below header) ────
ticker_data = fetch_ticker_data(TICKER_SYMBOLS)

if ticker_data:
    ticker_items_html = ""
    for t in ticker_data:
        color = "#22c55e" if t["change"] >= 0 else "#ef4444"
        arrow = "▲" if t["change"] >= 0 else "▼"
        last = t["price"]
        prev = t["prev"]
        pct = t["change"]
        
        item_html = (
            f"<div class='ticker-card'>"
            f"<div style='color:#94a3b8; font-size:0.75rem; font-weight:700; text-transform:uppercase; margin-bottom:5px; letter-spacing:0.05em; white-space:nowrap;'>{t['name']}</div>"
            f"<div style='color:#e2e8f0; font-size:1.35rem; font-weight:800; line-height:1.2; text-shadow:0 2px 6px rgba(0,0,0,0.7);'>{get_currency_symbol(t['name'])}{last:,.2f}</div>"
            f"<div style='font-size:0.85rem; font-weight:700; margin-top:3px; color:{color}; text-shadow:0 1px 3px rgba(0,0,0,0.5);'>{arrow} {pct:+.2f}%</div>"
            f"</div>"
        )
        ticker_items_html += item_html

    # Duplicate for seamless scroll
    full_ticker = ticker_items_html + ticker_items_html
else:
    full_ticker = "<div style='color:#94a3b8; font-size:0.9rem; padding: 0 40px;'>Live market data currently unavailable. Retrying...</div>"

st.markdown(
    f"""
    <div class="ticker-bar">
        <div class="ticker-content">
            {full_ticker}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Spacer to push content below the fixed ticker
st.markdown('<div style="height: 130px;"></div>', unsafe_allow_html=True)

st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#              SECTION 1: CROSS-MARKET COMPARISON
# ══════════════════════════════════════════════════════════════
st.markdown('<p class="section-title">Cross-Market Comparison</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="section-subtitle">Compare normalized performance across instruments, asset classes and geographies</p>',
    unsafe_allow_html=True,
)

if selected_symbols:
    with st.spinner("Loading market data..."):
        close_data = fetch_comparison_data(selected_symbols, period)

    if not close_data.empty:
        # ─── Metrics Row ───
        mcols = st.columns(min(len(selected_symbols), 8))
        for idx, sym in enumerate(selected_symbols[:8]):
            if sym in close_data.columns:
                series = close_data[sym].dropna()
                if len(series) >= 2:
                    last = float(series.iloc[-1])
                    prev = float(series.iloc[-2]) if len(series) > 1 else last
                    first = float(series.iloc[0])
                    pct = round((last - first) / first * 100, 2)
                    day_chg = round((last - prev) / prev * 100, 2) if prev else 0
                    with mcols[idx % len(mcols)]:
                        card_html = f"""
                        <div class="glass-card" style="padding: 14px 10px; margin-bottom: 12px; text-align: center; height: 100%; box-sizing: border-box;">
                            <div style="color: #94a3b8; font-size: 0.75rem; font-weight: 700; text-transform: uppercase; margin-bottom: 4px; letter-spacing: 0.05em; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{sym_label(sym)}</div>
                            <div style="color: #e2e8f0; font-size: 1.35rem; font-weight: 800; line-height: 1.2;">{get_currency_symbol(sym)}{last:,.2f}</div>
                            <div style="font-size: 0.85rem; font-weight: 700; margin-top: 4px; color: {'#22c55e' if pct >= 0 else '#ef4444'};">
                                {'▲' if pct >= 0 else '▼'} {pct:+.2f}% <span style="font-size: 0.7rem; font-weight: 500; color: #94a3b8;">({period_label})</span>
                            </div>
                            <div style="background: rgba(15,23,42,0.4); border-radius: 6px; padding: 6px; margin-top: 10px; font-size: 0.7rem; color: #cbd5e1; display: flex; justify-content: space-around; border: 1px solid rgba(255,255,255,0.05);">
                                <div style="display: flex; flex-direction: column;">
                                    <span style="color: #64748b; font-size: 0.6rem; text-transform: uppercase;">Yday</span>
                                    <span style="font-weight: 600;">{get_currency_symbol(sym)}{prev:,.2f}</span>
                                </div>
                                <div style="display: flex; flex-direction: column;">
                                    <span style="color: #64748b; font-size: 0.6rem; text-transform: uppercase;">Today</span>
                                    <span style="font-weight: 600;">{get_currency_symbol(sym)}{last:,.2f}</span>
                                </div>
                            </div>
                        </div>
                        """
                        st.markdown(card_html, unsafe_allow_html=True)

        st.markdown("")

        # ─── Chart Type Selection ───
        chart_tab1, chart_tab2, chart_tab3 = st.tabs(["Normalized Performance", "Individual Prices", "Drawdown Analysis"])

        with chart_tab1:
            normalized = close_data.div(close_data.iloc[0]) * 100
            fig = go.Figure()
            for i, col in enumerate(normalized.columns):
                fig.add_trace(
                    go.Scatter(
                        x=normalized.index,
                        y=normalized[col],
                        name=sym_label(col),
                        line=dict(color=COLOR_SEQUENCE[i % len(COLOR_SEQUENCE)], width=2.5),
                        hovertemplate=f"<b>{sym_label(col)}</b><br>Date: %{{x|%Y-%m-%d}}<br>Value: %{{y:.2f}}<extra></extra>",
                    )
                )
            fig.add_hline(y=100, line_dash="dot", line_color="rgba(100,116,139,0.3)", annotation_text="Baseline (100)")
            fig.update_layout(
                PLOTLY_LAYOUT,
                title=dict(text="Normalized Performance Comparison (Base = 100)", font=dict(size=16)),
                height=550,
                yaxis_title="Indexed Value",
                xaxis_title="",
                hovermode="x unified",
            )
            st.plotly_chart(fig, use_container_width=True, key="norm_chart")

        with chart_tab2:
            sym_for_price = st.multiselect(
                "Select instruments for price chart",
                selected_symbols,
                default=selected_symbols[:3],
                format_func=sym_label,
                key="price_select",
            )
            if sym_for_price:
                # If multiple instruments have very different scales, use secondary axis for 2, subplots for more
                if len(sym_for_price) <= 2:
                    fig2 = make_subplots(specs=[[{"secondary_y": len(sym_for_price) > 1}]])
                    for i, s in enumerate(sym_for_price):
                        if s in close_data.columns:
                            fig2.add_trace(
                                go.Scatter(
                                    x=close_data.index,
                                    y=close_data[s],
                                    name=sym_label(s),
                                    line=dict(color=COLOR_SEQUENCE[i], width=2.5),
                                ),
                                secondary_y=(i == 1 and len(sym_for_price) > 1),
                            )
                    fig2.update_layout(PLOTLY_LAYOUT, height=500, title="Price Comparison")
                    if len(sym_for_price) > 1:
                        fig2.update_yaxes(title_text=sym_label(sym_for_price[0]), secondary_y=False)
                        fig2.update_yaxes(title_text=sym_label(sym_for_price[1]), secondary_y=True)
                    st.plotly_chart(fig2, use_container_width=True, key="price_chart")
                else:
                    fig2 = make_subplots(
                        rows=len(sym_for_price), cols=1,
                        shared_xaxes=True,
                        subplot_titles=[sym_label(s) for s in sym_for_price],
                        vertical_spacing=0.04,
                    )
                    for i, s in enumerate(sym_for_price):
                        if s in close_data.columns:
                            fig2.add_trace(
                                go.Scatter(
                                    x=close_data.index,
                                    y=close_data[s],
                                    name=sym_label(s),
                                    line=dict(color=COLOR_SEQUENCE[i % len(COLOR_SEQUENCE)], width=2),
                                    showlegend=True,
                                ),
                                row=i + 1, col=1,
                            )
                    fig2.update_layout(
                        PLOTLY_LAYOUT,
                        height=250 * len(sym_for_price),
                        title="Individual Price Charts",
                    )
                    st.plotly_chart(fig2, use_container_width=True, key="price_multi_chart")

        with chart_tab3:
            fig3 = go.Figure()
            for i, col in enumerate(close_data.columns):
                series = close_data[col].dropna()
                cummax = series.cummax()
                drawdown = (series - cummax) / cummax * 100
                fig3.add_trace(
                    go.Scatter(
                        x=drawdown.index,
                        y=drawdown,
                        name=sym_label(col),
                        fill="tozeroy",
                        line=dict(color=COLOR_SEQUENCE[i % len(COLOR_SEQUENCE)], width=1.5),
                        fillcolor=f"rgba({int(COLOR_SEQUENCE[i % len(COLOR_SEQUENCE)][1:3], 16)},{int(COLOR_SEQUENCE[i % len(COLOR_SEQUENCE)][3:5], 16)},{int(COLOR_SEQUENCE[i % len(COLOR_SEQUENCE)][5:7], 16)},0.1)",
                    )
                )
            fig3.update_layout(
                PLOTLY_LAYOUT,
                title="Maximum Drawdown Analysis",
                height=450,
                yaxis_title="Drawdown (%)",
            )
            st.plotly_chart(fig3, use_container_width=True, key="dd_chart")

        # ─── Returns Summary Table ───
        with st.expander("Returns & Statistics Summary", expanded=False):
            stats_data = []
            for sym in selected_symbols:
                if sym in close_data.columns:
                    s = close_data[sym].dropna()
                    if len(s) >= 2:
                        total_ret = (s.iloc[-1] / s.iloc[0] - 1) * 100
                        daily_ret = s.pct_change().dropna()
                        ann_vol = daily_ret.std() * np.sqrt(252) * 100
                        sharpe = (daily_ret.mean() * 252) / (daily_ret.std() * np.sqrt(252)) if daily_ret.std() > 0 else 0
                        max_dd = ((s - s.cummax()) / s.cummax()).min() * 100
                        stats_data.append({
                            "Instrument": sym_label(sym),
                            "Current Price": f"{s.iloc[-1]:,.2f}",
                            f"Total Return ({period_label})": f"{total_ret:+.2f}%",
                            "Ann. Volatility": f"{ann_vol:.2f}%",
                            "Sharpe Ratio": f"{sharpe:.3f}",
                            "Max Drawdown": f"{max_dd:.2f}%",
                            "Best Day": f"{daily_ret.max()*100:+.2f}%",
                            "Worst Day": f"{daily_ret.min()*100:+.2f}%",
                        })
            if stats_data:
                st.dataframe(pd.DataFrame(stats_data), use_container_width=True, hide_index=True)
else:
    st.info("👈 Select instruments from the sidebar to begin analysis.")

st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#          SECTION 2: TECHNICAL ANALYSIS (50 INDICATORS)
# ══════════════════════════════════════════════════════════════
st.markdown('<p class="section-title">Technical Analysis Engine</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="section-subtitle">50+ technical indicators with interactive visualization across all instruments</p>',
    unsafe_allow_html=True,
)

if selected_symbols:
    ta_col1, ta_col2 = st.columns([1, 3])

    with ta_col1:
        ta_options = (
            list(INSTRUMENT_MAP.get("Indian Stocks", {}).values()) +
            list(INSTRUMENT_MAP.get("US Stocks", {}).values()) +
            list(INSTRUMENT_MAP.get("Indian Indices", {}).values()) +
            list(INSTRUMENT_MAP.get("International Indices", {}).values())
        )
        ta_sym = st.selectbox(
            "Instrument",
            ta_options,
            format_func=sym_label,
            key="ta_instrument",
        )

    with ta_col2:
        st.markdown(f"**Selected Indicators ({len(selected_indicator_list)}):** {', '.join(selected_indicator_list[:15])}" + ("..." if len(selected_indicator_list) > 15 else ""))

    if ta_sym:
        with st.spinner(f"Loading {sym_label(ta_sym)} data..."):
            ohlcv = fetch_single_ohlcv(ta_sym, period)

        if not ohlcv.empty:
            result_df = compute_indicators(ohlcv, selected_indicator_list)
            indicator_cols = [c for c in result_df.columns if c not in ["Open", "High", "Low", "Close", "Volume"]]

            # Separate overlay indicators from oscillator indicators
            overlay_indicators = []
            oscillator_indicators = []
            volume_indicators = []

            for col in indicator_cols:
                col_lower = col.lower()
                if any(kw in col_lower for kw in [
                    "sma", "ema", "dema", "tema", "wma", "hma", "bbu", "bbm", "bbl",
                    "kcu", "kcl", "kcb", "dcu", "dcl", "dcm", "vwap", "psar",
                    "isa_", "isb_", "its_", "iks_", "supert",
                ]):
                    overlay_indicators.append(col)
                elif any(kw in col_lower for kw in ["obv", "ad", "cmf", "efi", "vol_sma", "force"]):
                    volume_indicators.append(col)
                else:
                    oscillator_indicators.append(col)

            # ──── Main Candlestick Chart with Overlays ────
            ta_tab1, ta_tab2, ta_tab3 = st.tabs(["Price + Overlays", "Oscillators", "Data Table"])

            with ta_tab1:
                fig = make_subplots(
                    rows=2, cols=1,
                    shared_xaxes=True,
                    row_heights=[0.75, 0.25],
                    vertical_spacing=0.05,
                )
                # Candlestick
                fig.add_trace(
                    go.Candlestick(
                        x=result_df.index,
                        open=result_df["Open"],
                        high=result_df["High"],
                        low=result_df["Low"],
                        close=result_df["Close"],
                        name="Price",
                        increasing_line_color="#22c55e",
                        decreasing_line_color="#ef4444",
                        increasing_fillcolor="#22c55e",
                        decreasing_fillcolor="#ef4444",
                    ),
                    row=1, col=1,
                )
                # Overlay indicators
                for i, col in enumerate(overlay_indicators):
                    if col in result_df.columns:
                        fig.add_trace(
                            go.Scatter(
                                x=result_df.index,
                                y=result_df[col],
                                name=col,
                                line=dict(
                                    color=COLOR_SEQUENCE[(i + 2) % len(COLOR_SEQUENCE)],
                                    width=1.5,
                                    dash="dot" if "bb" in col.lower() or "kc" in col.lower() or "dc" in col.lower() else "solid",
                                ),
                                opacity=0.85,
                            ),
                            row=1, col=1,
                        )
                # Volume
                colors = ["#22c55e" if c >= o else "#ef4444" for c, o in zip(result_df["Close"], result_df["Open"])]
                fig.add_trace(
                    go.Bar(
                        x=result_df.index,
                        y=result_df["Volume"],
                        name="Volume",
                        marker_color=colors,
                        opacity=0.4,
                    ),
                    row=2, col=1,
                )
                # Volume overlays
                for col in volume_indicators:
                    if col in result_df.columns:
                        fig.add_trace(
                            go.Scatter(
                                x=result_df.index,
                                y=result_df[col],
                                name=col,
                                line=dict(width=1.5),
                            ),
                            row=2, col=1,
                        )

                fig.update_layout(
                    PLOTLY_LAYOUT,
                    title=f"{sym_label(ta_sym)} — Technical Chart",
                    height=680,
                    xaxis_rangeslider_visible=False,
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10)),
                )
                fig.update_xaxes(gridcolor="rgba(100,116,139,0.06)")
                fig.update_yaxes(gridcolor="rgba(100,116,139,0.06)")
                st.plotly_chart(fig, use_container_width=True, key="ta_candle")

            with ta_tab2:
                if oscillator_indicators:
                    # Group oscillators into subplots (max 4 per chart)
                    osc_groups = [oscillator_indicators[i:i+3] for i in range(0, len(oscillator_indicators), 3)]
                    for grp_idx, grp in enumerate(osc_groups):
                        fig_osc = make_subplots(
                            rows=len(grp), cols=1,
                            shared_xaxes=True,
                            vertical_spacing=0.08,
                            subplot_titles=grp,
                        )
                        for i, col in enumerate(grp):
                            if col in result_df.columns:
                                fig_osc.add_trace(
                                    go.Scatter(
                                        x=result_df.index,
                                        y=result_df[col],
                                        name=col,
                                        line=dict(color=COLOR_SEQUENCE[i % len(COLOR_SEQUENCE)], width=2),
                                        fill="tozeroy" if "macd" not in col.lower() else None,
                                        fillcolor=f"rgba({int(COLOR_SEQUENCE[i % len(COLOR_SEQUENCE)][1:3], 16)},{int(COLOR_SEQUENCE[i % len(COLOR_SEQUENCE)][3:5], 16)},{int(COLOR_SEQUENCE[i % len(COLOR_SEQUENCE)][5:7], 16)},0.08)" if "macd" not in col.lower() else None,
                                    ),
                                    row=i + 1, col=1,
                                )
                                # Add reference lines for common oscillators
                                if "rsi" in col.lower():
                                    fig_osc.add_hline(y=70, line_dash="dash", line_color="rgba(239,68,68,0.4)", row=i+1, col=1)
                                    fig_osc.add_hline(y=30, line_dash="dash", line_color="rgba(34,197,94,0.4)", row=i+1, col=1)
                                elif "cci" in col.lower():
                                    fig_osc.add_hline(y=100, line_dash="dash", line_color="rgba(239,68,68,0.4)", row=i+1, col=1)
                                    fig_osc.add_hline(y=-100, line_dash="dash", line_color="rgba(34,197,94,0.4)", row=i+1, col=1)

                        fig_osc.update_layout(
                            PLOTLY_LAYOUT,
                            height=200 * len(grp) + 50,
                            showlegend=False,
                        )
                        fig_osc.update_xaxes(gridcolor="rgba(100,116,139,0.06)")
                        fig_osc.update_yaxes(gridcolor="rgba(100,116,139,0.06)")
                        st.plotly_chart(fig_osc, use_container_width=True, key=f"osc_{grp_idx}")
                else:
                    st.info("Select momentum/volatility indicators from the sidebar to view oscillator charts.")

            with ta_tab3:
                display_cols = ["Open", "High", "Low", "Close", "Volume"] + indicator_cols
                display_df = result_df[display_cols].tail(50).round(4)
                st.dataframe(display_df, use_container_width=True, height=500)

            # ─── Indicator comparison across instruments ───
            with st.expander("🔄 Compare Indicator Across Instruments", expanded=False):
                compare_ind = st.selectbox(
                    "Select Indicator",
                    ["RSI 14", "ATR 14", "MACD", "CCI 20", "ROC 12", "Historical Volatility"],
                    key="compare_indicator",
                )
                if len(selected_symbols) > 1:
                    comp_fig = go.Figure()
                    for i, sym in enumerate(selected_symbols[:8]):
                        sym_df = fetch_single_ohlcv(sym, period)
                        if not sym_df.empty:
                            sym_result = compute_indicators(sym_df, [compare_ind])
                            ind_cols = [c for c in sym_result.columns if c not in ["Open", "High", "Low", "Close", "Volume"]]
                            if ind_cols:
                                comp_fig.add_trace(
                                    go.Scatter(
                                        x=sym_result.index,
                                        y=sym_result[ind_cols[0]],
                                        name=sym_label(sym),
                                        line=dict(color=COLOR_SEQUENCE[i % len(COLOR_SEQUENCE)], width=2),
                                    )
                                )
                    comp_fig.update_layout(
                        PLOTLY_LAYOUT,
                        title=f"{compare_ind} — Cross-Instrument Comparison",
                        height=400,
                        hovermode="x unified",
                    )
                    st.plotly_chart(comp_fig, use_container_width=True, key="comp_ind_chart")
        else:
            st.warning(f"No data available for {sym_label(ta_sym)}.")

st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#              SECTION 3: FUNDAMENTAL ANALYSIS
# ══════════════════════════════════════════════════════════════
st.markdown('<p class="section-title">Fundamental Analysis</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="section-subtitle">Financials and key metrics for selected stocks</p>',
    unsafe_allow_html=True,
)

stock_symbols = list(INSTRUMENT_MAP.get("Indian Stocks", {}).values()) + list(INSTRUMENT_MAP.get("US Stocks", {}).values())
fund_options = stock_symbols

if fund_options:
    fund_sym = st.selectbox(
        "Select Instrument for Fundamentals",
        fund_options,
        format_func=sym_label,
        key="fund_instrument",
    )

    if fund_sym:
        info = fetch_fundamentals(fund_sym)
        if info:
            currency_sym = get_currency_symbol(fund_sym)

            # ─── Key Metrics Cards ───
            f_cols = st.columns(5)
            with f_cols[0]:
                curr_price = info.get("currentPrice", info.get("regularMarketPrice", None))
                st.metric("Current Price", f"{currency_sym}{curr_price:,.2f}" if curr_price else "N/A")
            with f_cols[1]:
                mcap = info.get("marketCap", 0)
                st.metric("Market Cap", format_number(mcap, currency_sym))
            with f_cols[2]:
                pe = info.get("trailingPE", None)
                st.metric("P/E Ratio", f"{pe:.2f}" if pe else "N/A")
            with f_cols[3]:
                eps = info.get("trailingEps", None)
                st.metric("EPS", f"{currency_sym}{eps:.2f}" if eps else "N/A")
            with f_cols[4]:
                div_yield = info.get("dividendYield", None)
                st.metric("Dividend Yield", f"{div_yield*100:.2f}%" if div_yield else "N/A")

            f_cols2 = st.columns(5)
            with f_cols2[0]:
                pb = info.get("priceToBook", None)
                st.metric("P/B Ratio", f"{pb:.2f}" if pb else "N/A")
            with f_cols2[1]:
                roe = info.get("returnOnEquity", None)
                st.metric("ROE", f"{roe*100:.2f}%" if roe else "N/A")
            with f_cols2[2]:
                de = info.get("debtToEquity", None)
                st.metric("Debt/Equity", f"{de:.2f}" if de else "N/A")
            with f_cols2[3]:
                beta_val = info.get("beta", None)
                st.metric("Beta", f"{beta_val:.3f}" if beta_val else "N/A")
            with f_cols2[4]:
                target_price = info.get("targetMeanPrice", None)
                st.metric("Target Price", f"{currency_sym}{target_price:,.2f}" if target_price else "N/A")

            # ─── Company Info ───
            with st.expander("Company Profile", expanded=False):
                prof_cols = st.columns(2)
                with prof_cols[0]:
                    emp_count = info.get('fullTimeEmployees', 'N/A')
                    emp_str = f"{emp_count:,}" if isinstance(emp_count, (int, float)) else emp_count
                    
                    st.markdown(f"""
                    <div class="glass-card" style="padding: 16px; margin-bottom: 0; background: rgba(15,23,42,0.5);">
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; font-size: 0.85rem;">
                            <div style="color: #94a3b8;">Name</div><div style="color: #e2e8f0; font-weight: 600; text-align: right;">{info.get('longName', 'N/A')}</div>
                            <div style="color: #94a3b8;">Sector</div><div style="color: #e2e8f0; font-weight: 600; text-align: right;">{info.get('sector', 'N/A')}</div>
                            <div style="color: #94a3b8;">Industry</div><div style="color: #e2e8f0; font-weight: 600; text-align: right;">{info.get('industry', 'N/A')}</div>
                            <div style="color: #94a3b8;">Country</div><div style="color: #e2e8f0; font-weight: 600; text-align: right;">{info.get('country', 'N/A')}</div>
                            <div style="color: #94a3b8;">Employees</div><div style="color: #e2e8f0; font-weight: 600; text-align: right;">{emp_str}</div>
                            <div style="color: #94a3b8;">Exchange</div><div style="color: #e2e8f0; font-weight: 600; text-align: right;">{info.get('exchange', 'N/A')}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                with prof_cols[1]:
                    st.markdown(f"""
                    <div class="glass-card" style="padding: 16px; margin-bottom: 0; background: rgba(15,23,42,0.5);">
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; font-size: 0.85rem;">
                            <div style="color: #94a3b8;">52W High</div><div style="color: #22c55e; font-weight: 600; text-align: right;">{currency_sym}{info.get('fiftyTwoWeekHigh', 'N/A')}</div>
                            <div style="color: #94a3b8;">52W Low</div><div style="color: #ef4444; font-weight: 600; text-align: right;">{currency_sym}{info.get('fiftyTwoWeekLow', 'N/A')}</div>
                            <div style="color: #94a3b8;">50D MA</div><div style="color: #e2e8f0; font-weight: 600; text-align: right;">{currency_sym}{info.get('fiftyDayAverage', 'N/A')}</div>
                            <div style="color: #94a3b8;">200D MA</div><div style="color: #e2e8f0; font-weight: 600; text-align: right;">{currency_sym}{info.get('twoHundredDayAverage', 'N/A')}</div>
                            <div style="color: #94a3b8;">Avg Volume</div><div style="color: #e2e8f0; font-weight: 600; text-align: right;">{format_number(info.get('averageVolume', 0))}</div>
                            <div style="color: #94a3b8;">Revenue</div><div style="color: #e2e8f0; font-weight: 600; text-align: right;">{format_number(info.get('totalRevenue', 0), currency_sym)}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                business_summary = info.get("longBusinessSummary", "")
                website = info.get("website", "#")
                if business_summary:
                    summary_text = business_summary[:250] + "..." if len(business_summary) > 250 else business_summary
                    st.markdown(f"""
                    <div class="glass-card" style="margin-top: 15px; padding: 15px; background: rgba(15,23,42,0.6);">
                        <div style="font-size: 0.9rem; color: #cbd5e1; line-height: 1.5; margin-bottom: 10px;">
                            <strong>About:</strong> {summary_text}
                        </div>
                        <div style="text-align: right;">
                            <a href="{website}" target="_blank" style="color: #06b6d4; text-decoration: none; font-weight: 600; font-size: 0.85rem;">
                                Visit Company Website ↗
                            </a>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                        # ─── Historical Performance Chart ───
            with st.expander("Historical Performance Analysis", expanded=True):
                hist_periods = {"1Y": "1y", "2Y": "2y", "5Y": "5y", "10Y": "10y", "Max": "max"}
                hist_period_sel = st.radio(
                    "Performance Period",
                    list(hist_periods.keys()),
                    horizontal=True,
                    index=2,
                    key="hist_perf_period",
                )
                hist_df = fetch_single_ohlcv(fund_sym, hist_periods[hist_period_sel])

                if not hist_df.empty:
                    # Performance metrics
                    total_return = (hist_df["Close"].iloc[-1] / hist_df["Close"].iloc[0] - 1) * 100
                    daily_returns = hist_df["Close"].pct_change().dropna()
                    ann_return = ((1 + daily_returns.mean()) ** 252 - 1) * 100
                    ann_vol = daily_returns.std() * np.sqrt(252) * 100
                    sharpe = ann_return / ann_vol if ann_vol > 0 else 0
                    max_dd = ((hist_df["Close"] - hist_df["Close"].cummax()) / hist_df["Close"].cummax()).min() * 100
                    calmar = ann_return / abs(max_dd) if max_dd != 0 else 0
                    sortino_denom = daily_returns[daily_returns < 0].std() * np.sqrt(252) * 100
                    sortino = ann_return / sortino_denom if sortino_denom > 0 else 0
                    win_rate = (daily_returns > 0).sum() / len(daily_returns) * 100
                    skewness = daily_returns.skew()
                    kurtosis_val = daily_returns.kurtosis()

                    perf_cols = st.columns(5)
                    with perf_cols[0]:
                        st.metric(f"Total Return ({hist_period_sel})", f"{total_return:+.2f}%")
                    with perf_cols[1]:
                        st.metric("Annualized Return", f"{ann_return:+.2f}%")
                    with perf_cols[2]:
                        st.metric("Annualized Vol", f"{ann_vol:.2f}%")
                    with perf_cols[3]:
                        st.metric("Sharpe Ratio", f"{sharpe:.3f}")
                    with perf_cols[4]:
                        st.metric("Max Drawdown", f"{max_dd:.2f}%")

                    perf_cols2 = st.columns(5)
                    with perf_cols2[0]:
                        st.metric("Calmar Ratio", f"{calmar:.3f}")
                    with perf_cols2[1]:
                        st.metric("Sortino Ratio", f"{sortino:.3f}")
                    with perf_cols2[2]:
                        st.metric("Win Rate", f"{win_rate:.1f}%")
                    with perf_cols2[3]:
                        st.metric("Skewness", f"{skewness:.3f}")
                    with perf_cols2[4]:
                        st.metric("Kurtosis", f"{kurtosis_val:.3f}")

                    st.markdown("")

                    # ─── Price + Volume Chart ───
                    fig_hist = make_subplots(
                        rows=3, cols=1,
                        shared_xaxes=True,
                        row_heights=[0.55, 0.20, 0.25],
                        vertical_spacing=0.04,
                        subplot_titles=["Price History", "Volume", "Rolling Returns (30D)"],
                    )
                    fig_hist.add_trace(
                        go.Candlestick(
                            x=hist_df.index,
                            open=hist_df["Open"],
                            high=hist_df["High"],
                            low=hist_df["Low"],
                            close=hist_df["Close"],
                            name="OHLC",
                            increasing_line_color="#22c55e",
                            decreasing_line_color="#ef4444",
                            increasing_fillcolor="#22c55e",
                            decreasing_fillcolor="#ef4444",
                        ),
                        row=1, col=1,
                    )
                    # Add 50 & 200 SMA for context
                    if len(hist_df) >= 50:
                        sma50 = hist_df["Close"].rolling(50).mean()
                        fig_hist.add_trace(
                            go.Scatter(x=hist_df.index, y=sma50, name="SMA 50",
                                       line=dict(color="#06b6d4", width=1.5, dash="dot"), opacity=0.7),
                            row=1, col=1,
                        )
                    if len(hist_df) >= 200:
                        sma200 = hist_df["Close"].rolling(200).mean()
                        fig_hist.add_trace(
                            go.Scatter(x=hist_df.index, y=sma200, name="SMA 200",
                                       line=dict(color="#f59e0b", width=1.5, dash="dot"), opacity=0.7),
                            row=1, col=1,
                        )
                    # Volume bars
                    vol_colors = ["#22c55e" if c >= o else "#ef4444"
                                  for c, o in zip(hist_df["Close"], hist_df["Open"])]
                    fig_hist.add_trace(
                        go.Bar(x=hist_df.index, y=hist_df["Volume"], name="Volume",
                               marker_color=vol_colors, opacity=0.5),
                        row=2, col=1,
                    )
                    # Rolling 30d return
                    roll_30 = hist_df["Close"].pct_change(30) * 100
                    roll_colors = ["#22c55e" if v >= 0 else "#ef4444" for v in roll_30.fillna(0)]
                    fig_hist.add_trace(
                        go.Bar(x=hist_df.index, y=roll_30, name="30D Return %",
                               marker_color=roll_colors, opacity=0.6),
                        row=3, col=1,
                    )
                    fig_hist.add_hline(y=0, line_dash="dash", line_color="rgba(100,116,139,0.3)", row=3, col=1)

                    fig_hist.update_layout(
                        PLOTLY_LAYOUT,
                        height=750,
                        xaxis_rangeslider_visible=False,
                        showlegend=True,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10)),
                    )
                    fig_hist.update_xaxes(gridcolor="rgba(100,116,139,0.06)")
                    fig_hist.update_yaxes(gridcolor="rgba(100,116,139,0.06)")
                    st.plotly_chart(fig_hist, use_container_width=True, key="hist_perf_chart")

                    # ─── Monthly Returns Heatmap ───
                    st.markdown("**Monthly Returns Heatmap**")
                    monthly = hist_df["Close"].resample("ME").last().pct_change() * 100
                    monthly_df = pd.DataFrame({
                        "Year": monthly.index.year,
                        "Month": monthly.index.month,
                        "Return": monthly.values,
                    }).dropna()

                    if not monthly_df.empty:
                        pivot = monthly_df.pivot_table(index="Year", columns="Month", values="Return", aggfunc="mean")
                        month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                                        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
                        pivot.columns = [month_labels[int(c) - 1] for c in pivot.columns]
                        pivot = pivot.sort_index(ascending=False)

                        fig_hm = go.Figure(data=go.Heatmap(
                            z=pivot.values,
                            x=pivot.columns.tolist(),
                            y=[str(y) for y in pivot.index.tolist()],
                            colorscale=[
                                [0, "#ef4444"],
                                [0.35, "#ef4444"],
                                [0.5, "#1e293b"],
                                [0.65, "#22c55e"],
                                [1, "#22c55e"],
                            ],
                            zmid=0,
                            text=[[f"{v:.1f}%" if not np.isnan(v) else "" for v in row] for row in pivot.values],
                            texttemplate="%{text}",
                            textfont=dict(size=11, color="#e2e8f0"),
                            hovertemplate="Year: %{y}<br>Month: %{x}<br>Return: %{z:.2f}%<extra></extra>",
                            colorbar=dict(
                                title=dict(text="Return %", font=dict(color="#94a3b8")),
                                tickfont=dict(color="#94a3b8"),
                            ),
                        ))
                        fig_hm.update_layout(
                            PLOTLY_LAYOUT,
                            height=max(300, len(pivot) * 30 + 100),
                            title="Monthly Returns Heatmap (%)",
                            xaxis=dict(side="top", tickangle=0),
                        )
                        st.plotly_chart(fig_hm, use_container_width=True, key="monthly_heatmap")

                    # ─── Return Distribution ───
                    st.markdown("**Return Distribution**")
                    dist_cols = st.columns(2)
                    with dist_cols[0]:
                        fig_dist = go.Figure()
                        fig_dist.add_trace(go.Histogram(
                            x=daily_returns * 100,
                            nbinsx=80,
                            name="Daily Returns",
                            marker_color="#06b6d4",
                            opacity=0.7,
                        ))
                        fig_dist.add_vline(x=0, line_dash="dash", line_color="rgba(100,116,139,0.5)")
                        fig_dist.add_vline(x=daily_returns.mean() * 100, line_dash="dot",
                                           line_color="#f59e0b",
                                           annotation_text=f"Mean: {daily_returns.mean()*100:.3f}%")
                        fig_dist.update_layout(
                            PLOTLY_LAYOUT,
                            title="Daily Return Distribution",
                            height=350,
                            xaxis_title="Return (%)",
                            yaxis_title="Frequency",
                        )
                        st.plotly_chart(fig_dist, use_container_width=True, key="dist_chart")

                    with dist_cols[1]:
                        # Cumulative returns
                        cum_ret = (1 + daily_returns).cumprod()
                        fig_cum = go.Figure()
                        fig_cum.add_trace(go.Scatter(
                            x=cum_ret.index,
                            y=cum_ret * 100,
                            name="Cumulative Return",
                            fill="tozeroy",
                            line=dict(color="#8b5cf6", width=2),
                            fillcolor="rgba(139,92,246,0.1)",
                        ))
                        fig_cum.add_hline(y=100, line_dash="dot", line_color="rgba(100,116,139,0.3)",
                                          annotation_text="Baseline")
                        fig_cum.update_layout(
                            PLOTLY_LAYOUT,
                            title=f"Cumulative Growth of {currency_sym}100",
                            height=350,
                            yaxis_title="Value",
                        )
                        st.plotly_chart(fig_cum, use_container_width=True, key="cum_chart")

                    # ─── Yearly Returns ───
                    yearly = hist_df["Close"].resample("YE").last().pct_change().dropna() * 100
                    if not yearly.empty:
                        st.markdown("**Yearly Returns**")
                        yr_colors = ["#22c55e" if v >= 0 else "#ef4444" for v in yearly.values]
                        fig_yr = go.Figure(go.Bar(
                            x=[str(d.year) for d in yearly.index],
                            y=yearly.values,
                            marker_color=yr_colors,
                            text=[f"{v:+.1f}%" for v in yearly.values],
                            textposition="outside",
                            textfont=dict(size=11, color="#e2e8f0"),
                        ))
                        fig_yr.add_hline(y=0, line_dash="dash", line_color="rgba(100,116,139,0.3)")
                        fig_yr.update_layout(
                            PLOTLY_LAYOUT,
                            height=380,
                            title="Calendar Year Returns",
                            yaxis_title="Return (%)",
                        )
                        st.plotly_chart(fig_yr, use_container_width=True, key="yearly_chart")
                else:
                    st.warning("Insufficient historical data for this period.")

            # ─── Peer Comparison (if stock) ───
            if info.get("sector"):
                with st.expander("Peer Comparison", expanded=False):
                    #st.markdown(f"**Sector:** {info.get('sector', 'N/A')} · **Industry:** {info.get('industry', 'N/A')}")

                    # Find peers in the selected stock symbols
                    peer_data = []
                    for sym in fund_options:
                        p_info = fetch_fundamentals(sym)
                        if p_info:
                            peer_data.append({
                                "Instrument": sym_label(sym),
                                "Price": f"{p_info.get('currentPrice', p_info.get('regularMarketPrice', 'N/A'))}",
                                "Market Cap": format_number(p_info.get("marketCap", 0), get_currency_symbol(sym)),
                                "P/E": round(p_info.get("trailingPE", 0) or 0, 2),
                                "P/B": round(p_info.get("priceToBook", 0) or 0, 2),
                                "ROE %": round((p_info.get("returnOnEquity", 0) or 0) * 100, 2),
                                "D/E": round(p_info.get("debtToEquity", 0) or 0, 2),
                                "Div Yield %": round((p_info.get("dividendYield", 0) or 0) * 100, 2),
                                "Beta": round(p_info.get("beta", 0) or 0, 3),
                            })
                    if peer_data:
                        st.dataframe(pd.DataFrame(peer_data), use_container_width=True, hide_index=True)

                        # Visual comparison
                        peer_df = pd.DataFrame(peer_data)
                        metric_to_compare = st.selectbox(
                            "Compare Metric",
                            ["P/E", "P/B", "ROE %", "D/E", "Div Yield %", "Beta"],
                            key="peer_metric",
                        )
                        if metric_to_compare in peer_df.columns:
                            fig_peer = go.Figure(go.Bar(
                                x=peer_df["Instrument"],
                                y=pd.to_numeric(peer_df[metric_to_compare], errors="coerce"),
                                marker_color=COLOR_SEQUENCE[:len(peer_df)],
                                text=peer_df[metric_to_compare],
                                textposition="outside",
                                textfont=dict(color="#e2e8f0", size=12),
                            ))
                            fig_peer.update_layout(
                                PLOTLY_LAYOUT,
                                height=380,
                                title=f"{metric_to_compare} Comparison",
                                yaxis_title=metric_to_compare,
                            )
                            st.plotly_chart(fig_peer, use_container_width=True, key="peer_bar")
        else:
            st.warning("Fundamental data not available for this instrument.")

st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#          SECTION 4: OPTION CHAIN & OPTIONS PRICING
# ══════════════════════════════════════════════════════════════
st.markdown('<p class="section-title">Options Trading Intelligence</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="section-subtitle">Live option chains, Black-Scholes pricing, and Greeks analysis</p>',
    unsafe_allow_html=True,
)

opt_tab1, opt_tab2 = st.tabs(["Live Option Chain", "Options Pricing & Greeks"])

with opt_tab1:
    oc_col1, oc_col2 = st.columns([1, 4])
    with oc_col1:
        index_choice = st.selectbox("Index", ["NIFTY", "BANKNIFTY", "FINNIFTY", "SENSEX", "MIDCPNIFTY", "NIFTYIT"], key="oc_index")
        refresh_oc = st.button("🔴 Refresh Chain", key="refresh_oc")

    chain_data = fetch_nse_option_chain(index_choice)

    if chain_data and "records" in chain_data:
        records = chain_data["records"]
        filtered = chain_data.get("filtered", records)
        spot = records.get("underlyingValue", 0)

        # Compute aggregate data
        total_ce_oi = sum(item.get("CE", {}).get("openInterest", 0) for item in filtered.get("data", []))
        total_pe_oi = sum(item.get("PE", {}).get("openInterest", 0) for item in filtered.get("data", []))
        pcr = round(total_pe_oi / total_ce_oi, 3) if total_ce_oi > 0 else 0
        total_ce_chg = sum(item.get("CE", {}).get("changeinOpenInterest", 0) for item in filtered.get("data", []))
        total_pe_chg = sum(item.get("PE", {}).get("changeinOpenInterest", 0) for item in filtered.get("data", []))

        with oc_col2:
            spot_cols = st.columns(5)
            with spot_cols[0]:
                st.metric("Spot Price", f"₹{spot:,.2f}")
            with spot_cols[1]:
                st.metric("PCR (OI)", f"{pcr:.3f}",
                           delta_color="normal" if pcr > 1 else "inverse")
            with spot_cols[2]:
                st.metric("Total CE OI", format_number(total_ce_oi))
            with spot_cols[3]:
                st.metric("Total PE OI", format_number(total_pe_oi))
            with spot_cols[4]:
                sentiment = "🟢 Bullish" if pcr > 1 else "🔴 Bearish" if pcr < 0.7 else "🟡 Neutral"
                st.metric("Sentiment", sentiment)

        # Build option chain table
        rows = []
        for item in records.get("data", []):
            ce = item.get("CE", {})
            pe = item.get("PE", {})
            strike = item.get("strikePrice", 0)
            if ce or pe:
                rows.append({
                    "CE OI Chg": ce.get("changeinOpenInterest", 0),
                    "CE OI": ce.get("openInterest", 0),
                    "CE Volume": ce.get("totalTradedVolume", 0),
                    "CE IV": round(ce.get("impliedVolatility", 0), 2),
                    "CE LTP": round(ce.get("lastPrice", 0), 2),
                    "CE Chg": round(ce.get("change", 0), 2),
                    "Strike": strike,
                    "PE Chg": round(pe.get("change", 0), 2),
                    "PE LTP": round(pe.get("lastPrice", 0), 2),
                    "PE IV": round(pe.get("impliedVolatility", 0), 2),
                    "PE Volume": pe.get("totalTradedVolume", 0),
                    "PE OI": pe.get("openInterest", 0),
                    "PE OI Chg": pe.get("changeinOpenInterest", 0),
                })

        if rows:
            df_chain = pd.DataFrame(rows)

            # Filter around ATM
            atm_strike = min(df_chain["Strike"], key=lambda x: abs(x - spot))
            strikes_range = st.slider(
                "Strikes around ATM",
                min_value=5,
                max_value=min(30, len(df_chain) // 2),
                value=12,
                key="oc_range",
            )
            atm_idx = df_chain[df_chain["Strike"] == atm_strike].index[0]
            start_idx = max(0, atm_idx - strikes_range)
            end_idx = min(len(df_chain), atm_idx + strikes_range + 1)
            df_display = df_chain.iloc[start_idx:end_idx].reset_index(drop=True)

            # Style function
            def style_chain(row):
                styles = [""] * len(row)
                strike_idx = list(row.index).index("Strike")
                if row["Strike"] == atm_strike:
                    styles = ["background-color: rgba(6,182,212,0.15); font-weight: bold"] * len(row)
                elif row["Strike"] < spot:
                    # ITM calls
                    for i, col in enumerate(row.index):
                        if col.startswith("CE"):
                            styles[i] = "background-color: rgba(34,197,94,0.06)"
                elif row["Strike"] > spot:
                    # ITM puts
                    for i, col in enumerate(row.index):
                        if col.startswith("PE"):
                            styles[i] = "background-color: rgba(239,68,68,0.06)"
                return styles

            styled_df = df_display.style.apply(style_chain, axis=1).format({
                "CE OI": "{:,.0f}",
                "CE OI Chg": "{:+,.0f}",
                "CE Volume": "{:,.0f}",
                "PE OI": "{:,.0f}",
                "PE OI Chg": "{:+,.0f}",
                "PE Volume": "{:,.0f}",
                "Strike": "{:,.0f}",
                "CE LTP": "₹{:.2f}",
                "PE LTP": "₹{:.2f}",
                "CE Chg": "{:+.2f}",
                "PE Chg": "{:+.2f}",
                "CE IV": "{:.1f}%",
                "PE IV": "{:.1f}%",
            })
            st.dataframe(styled_df, use_container_width=True, hide_index=True, height=500)

            # ─── OI Charts ───
            oi_cols = st.columns(2)
            with oi_cols[0]:
                fig_oi = go.Figure()
                fig_oi.add_trace(go.Bar(
                    x=df_display["Strike"], y=df_display["CE OI"],
                    name="CE OI", marker_color="#22c55e", opacity=0.7,
                ))
                fig_oi.add_trace(go.Bar(
                    x=df_display["Strike"], y=df_display["PE OI"],
                    name="PE OI", marker_color="#ef4444", opacity=0.7,
                ))
                fig_oi.add_vline(x=spot, line_dash="dash", line_color="#06b6d4",
                                 annotation_text=f"Spot: {spot}")
                fig_oi.update_layout(
                    PLOTLY_LAYOUT,
                    title="Open Interest Distribution",
                    height=380,
                    barmode="group",
                    xaxis_title="Strike Price",
                    yaxis_title="Open Interest",
                )
                st.plotly_chart(fig_oi, use_container_width=True, key="oi_dist")

            with oi_cols[1]:
                fig_oi_chg = go.Figure()
                fig_oi_chg.add_trace(go.Bar(
                    x=df_display["Strike"], y=df_display["CE OI Chg"],
                    name="CE OI Change", marker_color="#22c55e", opacity=0.7,
                ))
                fig_oi_chg.add_trace(go.Bar(
                    x=df_display["Strike"], y=df_display["PE OI Chg"],
                    name="PE OI Change", marker_color="#ef4444", opacity=0.7,
                ))
                fig_oi_chg.add_vline(x=spot, line_dash="dash", line_color="#06b6d4",
                                     annotation_text=f"Spot: {spot}")
                fig_oi_chg.update_layout(
                    PLOTLY_LAYOUT,
                    title="Change in Open Interest",
                    height=380,
                    barmode="group",
                    xaxis_title="Strike Price",
                    yaxis_title="Change in OI",
                )
                st.plotly_chart(fig_oi_chg, use_container_width=True, key="oi_chg")

            # ─── IV Smile ───
            iv_data = df_display[df_display["CE IV"] > 0]
            if not iv_data.empty:
                fig_iv = go.Figure()
                fig_iv.add_trace(go.Scatter(
                    x=iv_data["Strike"], y=iv_data["CE IV"],
                    name="CE IV", mode="lines+markers",
                    line=dict(color="#22c55e", width=2),
                    marker=dict(size=6),
                ))
                fig_iv.add_trace(go.Scatter(
                    x=iv_data["Strike"], y=iv_data["PE IV"],
                    name="PE IV", mode="lines+markers",
                    line=dict(color="#ef4444", width=2),
                    marker=dict(size=6),
                ))
                fig_iv.add_vline(x=spot, line_dash="dash", line_color="#06b6d4",
                                 annotation_text=f"Spot: {spot}")
                fig_iv.update_layout(
                    PLOTLY_LAYOUT,
                    title="Implied Volatility Smile",
                    height=380,
                    xaxis_title="Strike Price",
                    yaxis_title="IV (%)",
                )
                st.plotly_chart(fig_iv, use_container_width=True, key="iv_smile")
    else:
        st.info("Live option chain data will be available once the market opens.")

with opt_tab2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    bs_col1, bs_col2 = st.columns([1, 2])

    with bs_col1:
        st.markdown("### Input Parameters")
        bs_spot = st.number_input("Spot Price (S)", value=24000.0, step=100.0, key="bs_spot")
        bs_strike = st.number_input("Strike Price (K)", value=24000.0, step=100.0, key="bs_strike")
        bs_days = st.number_input("Days to Expiry", value=7, min_value=0, max_value=365, key="bs_days")
        bs_vol = st.slider("Implied Volatility (%)", min_value=5, max_value=100, value=18, key="bs_vol") / 100
        bs_rate = st.slider("Risk-Free Rate (%)", min_value=0.0, max_value=15.0, value=6.5, step=0.1, key="bs_rate") / 100

    T = bs_days / 365
    call_price, call_delta, call_gamma, call_theta, call_vega, call_rho = black_scholes(bs_spot, bs_strike, T, bs_rate, bs_vol, "call")
    put_price, put_delta, put_gamma, put_theta, put_vega, put_rho = black_scholes(bs_spot, bs_strike, T, bs_rate, bs_vol, "put")

    with bs_col2:
        st.markdown("### Option Prices")
        price_cols = st.columns(2)
        with price_cols[0]:
            st.markdown(
                f"""
                <div class="metric-box" style="border-left: 4px solid #22c55e;">
                    <div class="metric-label">CALL PRICE</div>
                    <div class="metric-value" style="color:#22c55e; font-size:2rem;">₹{call_price:.2f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with price_cols[1]:
            st.markdown(
                f"""
                <div class="metric-box" style="border-left: 4px solid #ef4444;">
                    <div class="metric-label">PUT PRICE</div>
                    <div class="metric-value" style="color:#ef4444; font-size:2rem;">₹{put_price:.2f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("### Greeks")
        greek_cols = st.columns(5)
        greeks_call = [
            ("Δ Delta", f"{call_delta:.4f}"),
            ("Γ Gamma", f"{call_gamma:.6f}"),
            ("Θ Theta", f"{call_theta:.4f}"),
            ("ν Vega", f"{call_vega:.4f}"),
            ("ρ Rho", f"{call_rho:.4f}"),
        ]
        greeks_put = [
            ("Δ Delta", f"{put_delta:.4f}"),
            ("Γ Gamma", f"{put_gamma:.6f}"),
            ("Θ Theta", f"{put_theta:.4f}"),
            ("ν Vega", f"{put_vega:.4f}"),
            ("ρ Rho", f"{put_rho:.4f}"),
        ]

        st.markdown("**Call Greeks**")
        g_cols = st.columns(5)
        for i, (label, val) in enumerate(greeks_call):
            with g_cols[i]:
                st.markdown(
                    f'<div class="metric-box"><div class="metric-label">{label}</div>'
                    f'<div class="metric-value" style="font-size:1.1rem; color:#22c55e;">{val}</div></div>',
                    unsafe_allow_html=True,
                )

        st.markdown("**Put Greeks**")
        g_cols2 = st.columns(5)
        for i, (label, val) in enumerate(greeks_put):
            with g_cols2[i]:
                st.markdown(
                    f'<div class="metric-box"><div class="metric-label">{label}</div>'
                    f'<div class="metric-value" style="font-size:1.1rem; color:#ef4444;">{val}</div></div>',
                    unsafe_allow_html=True,
                )

    st.markdown("</div>", unsafe_allow_html=True)

    # ─── Greeks Sensitivity Charts ───
    with st.expander("Greeks Sensitivity Analysis", expanded=False):
        sens_col1, sens_col2 = st.columns(2)

        spot_range = np.linspace(bs_spot * 0.85, bs_spot * 1.15, 100)

        with sens_col1:
            # Delta vs Spot
            call_deltas = [black_scholes(s, bs_strike, T, bs_rate, bs_vol, "call")[1] for s in spot_range]
            put_deltas = [black_scholes(s, bs_strike, T, bs_rate, bs_vol, "put")[1] for s in spot_range]
            fig_delta = go.Figure()
            fig_delta.add_trace(go.Scatter(x=spot_range, y=call_deltas, name="Call Delta",
                                           line=dict(color="#22c55e", width=2.5)))
            fig_delta.add_trace(go.Scatter(x=spot_range, y=put_deltas, name="Put Delta",
                                           line=dict(color="#ef4444", width=2.5)))
            fig_delta.add_vline(x=bs_strike, line_dash="dash", line_color="rgba(100,116,139,0.4)",
                                annotation_text="Strike")
            fig_delta.update_layout(PLOTLY_LAYOUT, title="Delta vs Spot Price", height=350,
                                    xaxis_title="Spot Price", yaxis_title="Delta")
            st.plotly_chart(fig_delta, use_container_width=True, key="delta_sens")

        with sens_col2:
            # Gamma vs Spot
            call_gammas = [black_scholes(s, bs_strike, T, bs_rate, bs_vol, "call")[2] for s in spot_range]
            fig_gamma = go.Figure()
            fig_gamma.add_trace(go.Scatter(x=spot_range, y=call_gammas, name="Gamma",
                                            line=dict(color="#f59e0b", width=2.5),
                                            fill="tozeroy",
                                            fillcolor="rgba(245,158,11,0.1)"))
            fig_gamma.add_vline(x=bs_strike, line_dash="dash", line_color="rgba(100,116,139,0.4)",
                                annotation_text="Strike")
            fig_gamma.update_layout(PLOTLY_LAYOUT, title="Gamma vs Spot Price", height=350,
                                    xaxis_title="Spot Price", yaxis_title="Gamma")
            st.plotly_chart(fig_gamma, use_container_width=True, key="gamma_sens")

        sens_col3, sens_col4 = st.columns(2)

        with sens_col3:
            # Option price vs Spot
            call_prices_range = [black_scholes(s, bs_strike, T, bs_rate, bs_vol, "call")[0] for s in spot_range]
            put_prices_range = [black_scholes(s, bs_strike, T, bs_rate, bs_vol, "put")[0] for s in spot_range]
            intrinsic_call = [max(s - bs_strike, 0) for s in spot_range]
            intrinsic_put = [max(bs_strike - s, 0) for s in spot_range]
            fig_pv = go.Figure()
            fig_pv.add_trace(go.Scatter(x=spot_range, y=call_prices_range, name="Call Price",
                                         line=dict(color="#22c55e", width=2.5)))
            fig_pv.add_trace(go.Scatter(x=spot_range, y=put_prices_range, name="Put Price",
                                         line=dict(color="#ef4444", width=2.5)))
            fig_pv.add_trace(go.Scatter(x=spot_range, y=intrinsic_call, name="Call Intrinsic",
                                         line=dict(color="#22c55e", width=1, dash="dot"), opacity=0.5))
            fig_pv.add_trace(go.Scatter(x=spot_range, y=intrinsic_put, name="Put Intrinsic",
                                         line=dict(color="#ef4444", width=1, dash="dot"), opacity=0.5))
            fig_pv.add_vline(x=bs_strike, line_dash="dash", line_color="rgba(100,116,139,0.4)",
                             annotation_text="Strike")
            fig_pv.update_layout(PLOTLY_LAYOUT, title="Option Price vs Spot", height=350,
                                 xaxis_title="Spot Price", yaxis_title="Option Price (₹)")
            st.plotly_chart(fig_pv, use_container_width=True, key="price_vs_spot")

        with sens_col4:
            # Theta vs Days to Expiry
            days_range = np.arange(1, max(bs_days + 10, 30))
            call_thetas = [black_scholes(bs_spot, bs_strike, d/365, bs_rate, bs_vol, "call")[3] for d in days_range]
            put_thetas = [black_scholes(bs_spot, bs_strike, d/365, bs_rate, bs_vol, "put")[3] for d in days_range]
            fig_theta = go.Figure()
            fig_theta.add_trace(go.Scatter(x=days_range, y=call_thetas, name="Call Theta",
                                            line=dict(color="#22c55e", width=2.5)))
            fig_theta.add_trace(go.Scatter(x=days_range, y=put_thetas, name="Put Theta",
                                            line=dict(color="#ef4444", width=2.5)))
            fig_theta.update_layout(PLOTLY_LAYOUT, title="Theta Decay vs Days to Expiry", height=350,
                                    xaxis_title="Days to Expiry", yaxis_title="Theta (₹/day)")
            st.plotly_chart(fig_theta, use_container_width=True, key="theta_decay")

    # ─── P/L Payoff Diagram ───
    with st.expander("Strategy Payoff Calculator", expanded=False):
        strat_type = st.selectbox(
            "Strategy",
            ["Long Call", "Long Put", "Covered Call", "Protective Put",
             "Bull Call Spread", "Bear Put Spread", "Long Straddle", "Long Strangle"],
            key="strategy_type",
        )

        payoff_spots = np.linspace(bs_spot * 0.8, bs_spot * 1.2, 200)

        if strat_type == "Long Call":
            payoff = [max(s - bs_strike, 0) - call_price for s in payoff_spots]
            be_point = bs_strike + call_price
        elif strat_type == "Long Put":
            payoff = [max(bs_strike - s, 0) - put_price for s in payoff_spots]
            be_point = bs_strike - put_price
        elif strat_type == "Covered Call":
            payoff = [(s - bs_spot) + call_price - max(s - bs_strike, 0) for s in payoff_spots]
            be_point = bs_spot - call_price
        elif strat_type == "Protective Put":
            payoff = [(s - bs_spot) - put_price + max(bs_strike - s, 0) for s in payoff_spots]
            be_point = bs_spot + put_price
        elif strat_type == "Bull Call Spread":
            strike2 = bs_strike + 500
            c2 = black_scholes(bs_spot, strike2, T, bs_rate, bs_vol, "call")[0]
            cost = call_price - c2
            payoff = [max(s - bs_strike, 0) - max(s - strike2, 0) - cost for s in payoff_spots]
            be_point = bs_strike + cost
        elif strat_type == "Bear Put Spread":
            strike2 = bs_strike - 500
            p2 = black_scholes(bs_spot, strike2, T, bs_rate, bs_vol, "put")[0]
            cost = put_price - p2
            payoff = [max(bs_strike - s, 0) - max(strike2 - s, 0) - cost for s in payoff_spots]
            be_point = bs_strike - cost
        elif strat_type == "Long Straddle":
            total_cost = call_price + put_price
            payoff = [max(s - bs_strike, 0) + max(bs_strike - s, 0) - total_cost for s in payoff_spots]
            be_point = bs_strike  # two break-evens
        elif strat_type == "Long Strangle":
            strike_call = bs_strike + 250
            strike_put = bs_strike - 250
            c_strangle = black_scholes(bs_spot, strike_call, T, bs_rate, bs_vol, "call")[0]
            p_strangle = black_scholes(bs_spot, strike_put, T, bs_rate, bs_vol, "put")[0]
            total_cost = c_strangle + p_strangle
            payoff = [max(s - strike_call, 0) + max(strike_put - s, 0) - total_cost for s in payoff_spots]
            be_point = strike_call  # approximate
        else:
            payoff = [0] * len(payoff_spots)
            be_point = bs_strike

        payoff_colors = ["#22c55e" if p >= 0 else "#ef4444" for p in payoff]

        fig_payoff = go.Figure()
        # Profit zone
        fig_payoff.add_trace(go.Scatter(
            x=payoff_spots,
            y=[max(p, 0) for p in payoff],
            fill="tozeroy",
            fillcolor="rgba(34,197,94,0.15)",
            line=dict(width=0),
            name="Profit",
            showlegend=False,
        ))
        # Loss zone
        fig_payoff.add_trace(go.Scatter(
            x=payoff_spots,
            y=[min(p, 0) for p in payoff],
            fill="tozeroy",
            fillcolor="rgba(239,68,68,0.15)",
            line=dict(width=0),
            name="Loss",
            showlegend=False,
        ))
        # Main line
        fig_payoff.add_trace(go.Scatter(
            x=payoff_spots, y=payoff,
            name=strat_type,
            line=dict(color="#06b6d4", width=3),
        ))
        fig_payoff.add_hline(y=0, line_dash="dash", line_color="rgba(100,116,139,0.4)")
        fig_payoff.add_vline(x=bs_spot, line_dash="dot", line_color="#f59e0b",
                             annotation_text=f"Spot: {bs_spot}")
        fig_payoff.add_vline(x=bs_strike, line_dash="dot", line_color="#8b5cf6",
                             annotation_text=f"Strike: {bs_strike}")

        max_profit = max(payoff)
        max_loss = min(payoff)

        fig_payoff.update_layout(
            PLOTLY_LAYOUT,
            title=f"{strat_type} Payoff at Expiry",
            height=420,
            xaxis_title="Spot Price at Expiry",
            yaxis_title="P/L (₹)",
        )
        st.plotly_chart(fig_payoff, use_container_width=True, key="payoff_chart")

        pnl_cols = st.columns(3)
        with pnl_cols[0]:
            st.metric("Max Profit", f"₹{max_profit:,.2f}" if max_profit < 1e8 else "Unlimited")
        with pnl_cols[1]:
            st.metric("Max Loss", f"₹{max_loss:,.2f}")
        with pnl_cols[2]:
            st.metric("Break-Even", f"₹{be_point:,.2f}")


st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#          SECTION 5: QUANTITATIVE STRATEGIES & METRICS
# ══════════════════════════════════════════════════════════════
st.markdown('<p class="section-title">Quantitative Strategies & Risk Metrics</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="section-subtitle">Analyse correlations, rolling analytics, and strategy signals using quantitative methods.</p>',
    unsafe_allow_html=True,
)

if selected_symbols and len(selected_symbols) >= 2:
    quant_data = fetch_comparison_data(selected_symbols, period)

    if not quant_data.empty:
        returns = quant_data.pct_change().dropna()

        quant_tab1, quant_tab2, quant_tab3 = st.tabs(
            ["Correlation & Risk", "Rolling Analytics", "Strategy Signals"]
        )

        with quant_tab1:
            corr_cols = st.columns(2)

            with corr_cols[0]:
                # Correlation Matrix
                corr_matrix = returns.rename(columns={c: sym_label(c) for c in returns.columns}).corr()
                fig_corr = go.Figure(data=go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns.tolist(),
                    y=corr_matrix.index.tolist(),
                    colorscale=[
                        [0, "#ef4444"],
                        [0.5, "#1e293b"],
                        [1, "#22c55e"],
                    ],
                    zmid=0,
                    text=[[f"{v:.2f}" for v in row] for row in corr_matrix.values],
                    texttemplate="%{text}",
                    textfont=dict(size=11, color="#e2e8f0"),
                    hovertemplate="%{x} vs %{y}: %{z:.3f}<extra></extra>",
                    colorbar=dict(title=dict(text="Corr", font=dict(color="#94a3b8")), tickfont=dict(color="#94a3b8")),
                ))
                fig_corr.update_layout(
                    PLOTLY_LAYOUT,
                    title="Correlation Matrix",
                    height=450,
                    xaxis=dict(tickangle=45),
                )
                st.plotly_chart(fig_corr, use_container_width=True, key="corr_matrix")

            with corr_cols[1]:
                # Risk-Return Scatter
                ann_returns = returns.mean() * 252 * 100
                ann_vols = returns.std() * np.sqrt(252) * 100
                sharpe_ratios = ann_returns / ann_vols

                fig_rr = go.Figure()
                for i, sym in enumerate(returns.columns):
                    fig_rr.add_trace(go.Scatter(
                        x=[ann_vols[sym]],
                        y=[ann_returns[sym]],
                        mode="markers+text",
                        name=sym_label(sym),
                        text=[sym_label(sym)],
                        textposition="top center",
                        textfont=dict(size=11, color=COLOR_SEQUENCE[i % len(COLOR_SEQUENCE)]),
                        marker=dict(
                            size=max(12, abs(sharpe_ratios[sym]) * 15),
                            color=COLOR_SEQUENCE[i % len(COLOR_SEQUENCE)],
                            line=dict(width=2, color="#0f172a"),
                        ),
                        hovertemplate=f"<b>{sym_label(sym)}</b><br>Return: %{{y:.2f}}%<br>Vol: %{{x:.2f}}%<br>Sharpe: {sharpe_ratios[sym]:.3f}<extra></extra>",
                    ))

                fig_rr.update_layout(
                    PLOTLY_LAYOUT,
                    title="Risk-Return Profile (Annualized)",
                    height=450,
                    xaxis_title="Annualized Volatility (%)",
                    yaxis_title="Annualized Return (%)",
                    showlegend=False,
                )
                st.plotly_chart(fig_rr, use_container_width=True, key="risk_return")

            # Risk Metrics Table
            st.markdown("**Comprehensive Risk Metrics**")
            risk_data = []
            for sym in returns.columns:
                r = returns[sym].dropna()
                ann_ret = r.mean() * 252 * 100
                ann_v = r.std() * np.sqrt(252) * 100
                sharpe = ann_ret / ann_v if ann_v > 0 else 0
                sortino_d = r[r < 0].std() * np.sqrt(252) * 100
                sortino_r = ann_ret / sortino_d if sortino_d > 0 else 0
                cum = (1 + r).cumprod()
                max_dd = ((cum - cum.cummax()) / cum.cummax()).min() * 100
                calmar_r = ann_ret / abs(max_dd) if max_dd != 0 else 0
                var_95 = np.percentile(r, 5) * 100
                cvar_95 = r[r <= np.percentile(r, 5)].mean() * 100

                risk_data.append({
                    "Instrument": sym_label(sym),
                    "Ann. Return": f"{ann_ret:+.2f}%",
                    "Ann. Vol": f"{ann_v:.2f}%",
                    "Sharpe": f"{sharpe:.3f}",
                    "Sortino": f"{sortino_r:.3f}",
                    "Max DD": f"{max_dd:.2f}%",
                    "Calmar": f"{calmar_r:.3f}",
                    "VaR (95%)": f"{var_95:.2f}%",
                    "CVaR (95%)": f"{cvar_95:.2f}%",
                    "Skew": f"{r.skew():.3f}",
                    "Kurtosis": f"{r.kurtosis():.3f}",
                })

            st.dataframe(pd.DataFrame(risk_data), use_container_width=True, hide_index=True)

        with quant_tab2:
            rolling_sym = st.selectbox(
                "Instrument for Rolling Analysis",
                selected_symbols,
                format_func=sym_label,
                key="rolling_sym",
            )
            rolling_window = st.select_slider(
                "Rolling Window (days)",
                options=[10, 20, 30, 50, 60, 90, 120, 252],
                value=30,
                key="roll_window",
            )

            if rolling_sym in returns.columns:
                r_series = returns[rolling_sym].dropna()

                roll_fig = make_subplots(
                    rows=4, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.06,
                    subplot_titles=[
                        f"Rolling {rolling_window}D Return",
                        f"Rolling {rolling_window}D Volatility (Ann.)",
                        f"Rolling {rolling_window}D Sharpe Ratio",
                        f"Rolling {rolling_window}D Max Drawdown",
                    ],
                    row_heights=[0.25, 0.25, 0.25, 0.25],
                )

                # Rolling Return
                roll_ret = r_series.rolling(rolling_window).mean() * 252 * 100
                roll_fig.add_trace(go.Scatter(
                    x=roll_ret.index, y=roll_ret,
                    name="Rolling Return",
                    line=dict(color="#06b6d4", width=2),
                    fill="tozeroy", fillcolor="rgba(6,182,212,0.08)",
                ), row=1, col=1)
                roll_fig.add_hline(y=0, line_dash="dash", line_color="rgba(100,116,139,0.3)", row=1, col=1)

                # Rolling Volatility
                roll_vol = r_series.rolling(rolling_window).std() * np.sqrt(252) * 100
                roll_fig.add_trace(go.Scatter(
                    x=roll_vol.index, y=roll_vol,
                    name="Rolling Vol",
                    line=dict(color="#f59e0b", width=2),
                    fill="tozeroy", fillcolor="rgba(245,158,11,0.08)",
                ), row=2, col=1)

                # Rolling Sharpe
                roll_sharpe = roll_ret / roll_vol
                roll_fig.add_trace(go.Scatter(
                    x=roll_sharpe.index, y=roll_sharpe,
                    name="Rolling Sharpe",
                    line=dict(color="#8b5cf6", width=2),
                    fill="tozeroy", fillcolor="rgba(139,92,246,0.08)",
                ), row=3, col=1)
                roll_fig.add_hline(y=0, line_dash="dash", line_color="rgba(100,116,139,0.3)", row=3, col=1)
                roll_fig.add_hline(y=1, line_dash="dot", line_color="rgba(34,197,94,0.3)", row=3, col=1,
                                   annotation_text="Good (1.0)")

                # Rolling Max DD
                cum = (1 + r_series).cumprod()
                roll_dd = pd.Series(index=r_series.index, dtype=float)
                for i in range(rolling_window, len(cum)):
                    window = cum.iloc[i - rolling_window:i + 1]
                    dd = ((window - window.cummax()) / window.cummax()).min() * 100
                    roll_dd.iloc[i] = dd
                roll_fig.add_trace(go.Scatter(
                    x=roll_dd.index, y=roll_dd,
                    name="Rolling Max DD",
                    line=dict(color="#ef4444", width=2),
                    fill="tozeroy", fillcolor="rgba(239,68,68,0.08)",
                ), row=4, col=1)

                roll_fig.update_layout(
                    PLOTLY_LAYOUT,
                    height=900,
                    showlegend=False,
                )
                roll_fig.update_xaxes(gridcolor="rgba(100,116,139,0.06)")
                roll_fig.update_yaxes(gridcolor="rgba(100,116,139,0.06)")
                st.plotly_chart(roll_fig, use_container_width=True, key="rolling_analytics")

        with quant_tab3:
            signal_sym = st.selectbox(
                "Instrument for Signals",
                selected_symbols,
                format_func=sym_label,
                key="signal_sym",
            )

            if signal_sym:
                sig_df = fetch_single_ohlcv(signal_sym, period)
                if not sig_df.empty:
                    close = sig_df["Close"]

                    # Generate signals
                    sma_20 = ta.sma(close, 20)
                    sma_50 = ta.sma(close, 50)
                    rsi = ta.rsi(close, 14)
                    macd_df = ta.macd(close)
                    bb = ta.bbands(close, 20, 2)

                    signals = pd.DataFrame(index=sig_df.index)
                    signals["Close"] = close

                    # SMA Crossover
                    if sma_20 is not None and sma_50 is not None:
                        signals["SMA_20"] = sma_20
                        signals["SMA_50"] = sma_50
                        signals["SMA_Signal"] = 0
                        signals.loc[sma_20 > sma_50, "SMA_Signal"] = 1
                        signals.loc[sma_20 < sma_50, "SMA_Signal"] = -1
                        signals["SMA_Cross"] = signals["SMA_Signal"].diff()

                    # RSI Signal
                    if rsi is not None:
                        signals["RSI"] = rsi
                        signals["RSI_Signal"] = 0
                        signals.loc[rsi < 30, "RSI_Signal"] = 1   # Oversold = Buy
                        signals.loc[rsi > 70, "RSI_Signal"] = -1  # Overbought = Sell

                    # MACD Signal
                    if macd_df is not None:
                        signals["MACD"] = macd_df.iloc[:, 0]
                        signals["MACD_Signal_Line"] = macd_df.iloc[:, 1] if macd_df.shape[1] > 1 else 0
                        signals["MACD_Hist"] = macd_df.iloc[:, 2] if macd_df.shape[1] > 2 else 0

                    # BB Signal
                    if bb is not None:
                        bbu = bb.iloc[:, 0]
                        bbm = bb.iloc[:, 1]
                        bbl = bb.iloc[:, 2]
                        signals["BB_Signal"] = 0
                        signals.loc[close < bbl, "BB_Signal"] = 1   # Below lower = Buy
                        signals.loc[close > bbu, "BB_Signal"] = -1  # Above upper = Sell

                    # ─── Signal Summary ───
                    latest = signals.iloc[-1]
                    st.markdown(f"**📡 Latest Signals for {sym_label(signal_sym)}** (as of {signals.index[-1].strftime('%Y-%m-%d')})")

                    sig_cols = st.columns(4)
                    with sig_cols[0]:
                        sma_sig = "🟢 BUY" if latest.get("SMA_Signal", 0) == 1 else "🔴 SELL" if latest.get("SMA_Signal", 0) == -1 else "⚪ NEUTRAL"
                        st.markdown(
                            f'<div class="metric-box"><div class="metric-label">SMA 20/50 Crossover</div>'
                            f'<div class="metric-value" style="font-size:1.1rem;">{sma_sig}</div></div>',
                            unsafe_allow_html=True,
                        )
                    with sig_cols[1]:
                        rsi_val = latest.get("RSI", 50)
                        rsi_sig = "🟢 OVERSOLD" if rsi_val < 30 else "🔴 OVERBOUGHT" if rsi_val > 70 else "⚪ NEUTRAL"
                        st.markdown(
                            f'<div class="metric-box"><div class="metric-label">RSI ({rsi_val:.1f})</div>'
                            f'<div class="metric-value" style="font-size:1.1rem;">{rsi_sig}</div></div>',
                            unsafe_allow_html=True,
                        )
                    with sig_cols[2]:
                        macd_h = latest.get("MACD_Hist", 0)
                        macd_sig = "🟢 BULLISH" if macd_h > 0 else "🔴 BEARISH"
                        st.markdown(
                            f'<div class="metric-box"><div class="metric-label">MACD Histogram</div>'
                            f'<div class="metric-value" style="font-size:1.1rem;">{macd_sig}</div></div>',
                            unsafe_allow_html=True,
                        )
                    with sig_cols[3]:
                        bb_sig_val = latest.get("BB_Signal", 0)
                        bb_sig = "🟢 OVERSOLD" if bb_sig_val == 1 else "🔴 OVERBOUGHT" if bb_sig_val == -1 else "⚪ IN BAND"
                        st.markdown(
                            f'<div class="metric-box"><div class="metric-label">Bollinger Band</div>'
                            f'<div class="metric-value" style="font-size:1.1rem;">{bb_sig}</div></div>',
                            unsafe_allow_html=True,
                        )

                    # ─── Signal Chart ───
                    fig_sig = make_subplots(
                        rows=3, cols=1,
                        shared_xaxes=True,
                        row_heights=[0.5, 0.25, 0.25],
                        vertical_spacing=0.05,
                        subplot_titles=["Price with SMA & Bollinger Bands", "RSI", "MACD"],
                    )

                    # Price
                    fig_sig.add_trace(go.Scatter(
                        x=sig_df.index, y=close, name="Close",
                        line=dict(color="#e2e8f0", width=2),
                    ), row=1, col=1)

                    if "SMA_20" in signals.columns:
                        fig_sig.add_trace(go.Scatter(
                            x=sig_df.index, y=signals["SMA_20"], name="SMA 20",
                            line=dict(color="#06b6d4", width=1.5, dash="dot"),
                        ), row=1, col=1)
                    if "SMA_50" in signals.columns:
                        fig_sig.add_trace(go.Scatter(
                            x=sig_df.index, y=signals["SMA_50"], name="SMA 50",
                            line=dict(color="#f59e0b", width=1.5, dash="dot"),
                        ), row=1, col=1)

                                        # BB Bands on chart
                    if bb is not None:
                        fig_sig.add_trace(go.Scatter(
                            x=sig_df.index, y=bb.iloc[:, 0], name="BB Upper",
                            line=dict(color="rgba(139,92,246,0.5)", width=1, dash="dash"),
                            showlegend=True,
                        ), row=1, col=1)
                        fig_sig.add_trace(go.Scatter(
                            x=sig_df.index, y=bb.iloc[:, 2], name="BB Lower",
                            line=dict(color="rgba(139,92,246,0.5)", width=1, dash="dash"),
                            fill="tonexty",
                            fillcolor="rgba(139,92,246,0.05)",
                            showlegend=True,
                        ), row=1, col=1)
                        fig_sig.add_trace(go.Scatter(
                            x=sig_df.index, y=bb.iloc[:, 1], name="BB Mid",
                            line=dict(color="rgba(139,92,246,0.3)", width=1, dash="dot"),
                            showlegend=False,
                        ), row=1, col=1)

                    # Buy / Sell markers from SMA crossover
                    if "SMA_Cross" in signals.columns:
                        buy_signals = signals[signals["SMA_Cross"] == 2]
                        sell_signals = signals[signals["SMA_Cross"] == -2]
                        if not buy_signals.empty:
                            fig_sig.add_trace(go.Scatter(
                                x=buy_signals.index,
                                y=buy_signals["Close"],
                                mode="markers",
                                name="Buy Signal",
                                marker=dict(
                                    symbol="triangle-up",
                                    size=12,
                                    color="#22c55e",
                                    line=dict(width=1, color="#0f172a"),
                                ),
                                hovertemplate="BUY @ ₹%{y:.2f}<br>%{x|%Y-%m-%d}<extra></extra>",
                            ), row=1, col=1)
                        if not sell_signals.empty:
                            fig_sig.add_trace(go.Scatter(
                                x=sell_signals.index,
                                y=sell_signals["Close"],
                                mode="markers",
                                name="Sell Signal",
                                marker=dict(
                                    symbol="triangle-down",
                                    size=12,
                                    color="#ef4444",
                                    line=dict(width=1, color="#0f172a"),
                                ),
                                hovertemplate="SELL @ ₹%{y:.2f}<br>%{x|%Y-%m-%d}<extra></extra>",
                            ), row=1, col=1)

                    # RSI subplot
                    if "RSI" in signals.columns:
                        fig_sig.add_trace(go.Scatter(
                            x=sig_df.index, y=signals["RSI"], name="RSI 14",
                            line=dict(color="#06b6d4", width=2),
                        ), row=2, col=1)
                        fig_sig.add_hline(y=70, line_dash="dash", line_color="rgba(239,68,68,0.5)",
                                          row=2, col=1, annotation_text="Overbought (70)")
                        fig_sig.add_hline(y=30, line_dash="dash", line_color="rgba(34,197,94,0.5)",
                                          row=2, col=1, annotation_text="Oversold (30)")
                        fig_sig.add_hline(y=50, line_dash="dot", line_color="rgba(100,116,139,0.2)",
                                          row=2, col=1)
                        # Shade overbought/oversold zones
                        fig_sig.add_hrect(y0=70, y1=100, fillcolor="rgba(239,68,68,0.05)",
                                          line_width=0, row=2, col=1)
                        fig_sig.add_hrect(y0=0, y1=30, fillcolor="rgba(34,197,94,0.05)",
                                          line_width=0, row=2, col=1)

                    # MACD subplot
                    if "MACD" in signals.columns:
                        fig_sig.add_trace(go.Scatter(
                            x=sig_df.index, y=signals["MACD"], name="MACD",
                            line=dict(color="#06b6d4", width=2),
                        ), row=3, col=1)
                        if "MACD_Signal_Line" in signals.columns:
                            fig_sig.add_trace(go.Scatter(
                                x=sig_df.index, y=signals["MACD_Signal_Line"], name="Signal Line",
                                line=dict(color="#f59e0b", width=1.5, dash="dot"),
                            ), row=3, col=1)
                        if "MACD_Hist" in signals.columns:
                            hist_colors = [
                                "#22c55e" if v >= 0 else "#ef4444"
                                for v in signals["MACD_Hist"].fillna(0)
                            ]
                            fig_sig.add_trace(go.Bar(
                                x=sig_df.index, y=signals["MACD_Hist"], name="MACD Histogram",
                                marker_color=hist_colors, opacity=0.5,
                            ), row=3, col=1)
                        fig_sig.add_hline(y=0, line_dash="dash",
                                          line_color="rgba(100,116,139,0.3)", row=3, col=1)

                    fig_sig.update_layout(
                        PLOTLY_LAYOUT,
                        height=850,
                        showlegend=True,
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1,
                            font=dict(size=9),
                        ),
                        xaxis_rangeslider_visible=False,
                    )
                    fig_sig.update_xaxes(gridcolor="rgba(100,116,139,0.06)")
                    fig_sig.update_yaxes(gridcolor="rgba(100,116,139,0.06)")
                    st.plotly_chart(fig_sig, use_container_width=True, key="signal_chart")

                    # ─── Backtest simple SMA crossover strategy ───
                    with st.expander("🧪 Simple SMA Crossover Backtest", expanded=False):
                        if "SMA_Signal" in signals.columns:
                            bt = signals[["Close", "SMA_Signal"]].dropna().copy()
                            bt["Daily_Return"] = bt["Close"].pct_change()
                            bt["Strategy_Return"] = bt["Daily_Return"] * bt["SMA_Signal"].shift(1)
                            bt["Buy_Hold_Cum"] = (1 + bt["Daily_Return"]).cumprod()
                            bt["Strategy_Cum"] = (1 + bt["Strategy_Return"]).cumprod()

                            bt_fig = go.Figure()
                            bt_fig.add_trace(go.Scatter(
                                x=bt.index, y=bt["Buy_Hold_Cum"],
                                name="Buy & Hold",
                                line=dict(color="#64748b", width=2),
                            ))
                            bt_fig.add_trace(go.Scatter(
                                x=bt.index, y=bt["Strategy_Cum"],
                                name="SMA Crossover Strategy",
                                line=dict(color="#06b6d4", width=2.5),
                            ))
                            bt_fig.add_hline(y=1, line_dash="dot",
                                             line_color="rgba(100,116,139,0.3)",
                                             annotation_text="Baseline")
                            bt_fig.update_layout(
                                PLOTLY_LAYOUT,
                                title="Strategy vs Buy & Hold — Cumulative Returns",
                                height=420,
                                yaxis_title="Cumulative Return",
                                hovermode="x unified",
                            )
                            st.plotly_chart(bt_fig, use_container_width=True, key="backtest_chart")

                            # Backtest stats
                            bh_total = (bt["Buy_Hold_Cum"].iloc[-1] - 1) * 100
                            strat_total = (bt["Strategy_Cum"].iloc[-1] - 1) * 100
                            strat_daily = bt["Strategy_Return"].dropna()
                            strat_sharpe = (
                                (strat_daily.mean() * 252)
                                / (strat_daily.std() * np.sqrt(252))
                                if strat_daily.std() > 0
                                else 0
                            )
                            strat_cum = (1 + strat_daily).cumprod()
                            strat_dd = (
                                (strat_cum - strat_cum.cummax()) / strat_cum.cummax()
                            ).min() * 100
                            win_trades = (strat_daily > 0).sum()
                            total_trades = (strat_daily != 0).sum()
                            win_rate_bt = (win_trades / total_trades * 100) if total_trades > 0 else 0

                            bt_cols = st.columns(5)
                            with bt_cols[0]:
                                st.metric("Buy & Hold", f"{bh_total:+.2f}%")
                            with bt_cols[1]:
                                color = "normal" if strat_total > bh_total else "inverse"
                                st.metric(
                                    "Strategy Return",
                                    f"{strat_total:+.2f}%",
                                    f"{strat_total - bh_total:+.2f}% vs B&H",
                                    delta_color=color,
                                )
                            with bt_cols[2]:
                                st.metric("Strategy Sharpe", f"{strat_sharpe:.3f}")
                            with bt_cols[3]:
                                st.metric("Strategy Max DD", f"{strat_dd:.2f}%")
                            with bt_cols[4]:
                                st.metric("Win Rate", f"{win_rate_bt:.1f}%")
                        else:
                            st.info("SMA signals not available for backtesting.")
                else:
                    st.warning(f"No data available for {sym_label(signal_sym)}.")
else:
    st.info("👈 Select at least 2 instruments from the sidebar for quantitative analysis.")

st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#          SECTION 6: VOLATILITY SURFACE & TERM STRUCTURE
# ══════════════════════════════════════════════════════════════
st.markdown('<p class="section-title">Volatility Analysis</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="section-subtitle">Historical volatility comparison and regime detection across instruments</p>',
    unsafe_allow_html=True,
)

if selected_symbols:
    vol_tab1, vol_tab2 = st.tabs(["Volatility Comparison", "Regime Detection"])

    with vol_tab1:
        vol_window = st.select_slider(
            "Volatility Window (days)",
            options=[5, 10, 20, 30, 60, 90],
            value=20,
            key="vol_window",
        )

        vol_data = fetch_comparison_data(selected_symbols, period)
        if not vol_data.empty:
            vol_returns = vol_data.pct_change().dropna()
            rolling_vol = vol_returns.rolling(vol_window).std() * np.sqrt(252) * 100

            fig_vol = go.Figure()
            for i, col in enumerate(rolling_vol.columns):
                fig_vol.add_trace(go.Scatter(
                    x=rolling_vol.index,
                    y=rolling_vol[col],
                    name=sym_label(col),
                    line=dict(color=COLOR_SEQUENCE[i % len(COLOR_SEQUENCE)], width=2),
                    hovertemplate=(
                        f"<b>{sym_label(col)}</b><br>"
                        "Date: %{x|%Y-%m-%d}<br>"
                        "Vol: %{y:.2f}%<extra></extra>"
                    ),
                ))
            fig_vol.update_layout(
                PLOTLY_LAYOUT,
                title=f"Rolling {vol_window}-Day Annualized Volatility",
                height=480,
                yaxis_title="Annualized Volatility (%)",
                hovermode="x unified",
            )
            st.plotly_chart(fig_vol, use_container_width=True, key="vol_comparison")

            # Current volatility ranking
            st.markdown("**Current Volatility Ranking**")
            latest_vol = rolling_vol.iloc[-1].dropna().sort_values(ascending=False)
            rank_data = []
            for rank_idx, (sym, vol_val) in enumerate(latest_vol.items(), 1):
                avg_vol = rolling_vol[sym].mean()
                vol_percentile = (
                    (rolling_vol[sym] < vol_val).sum() / len(rolling_vol[sym].dropna()) * 100
                )
                regime = (
                    "🔴 HIGH" if vol_percentile > 80
                    else "🟡 ELEVATED" if vol_percentile > 60
                    else "🟢 NORMAL" if vol_percentile > 30
                    else "🔵 LOW"
                )
                rank_data.append({
                    "Rank": rank_idx,
                    "Instrument": sym_label(sym),
                    "Current Vol": f"{vol_val:.2f}%",
                    "Avg Vol": f"{avg_vol:.2f}%",
                    "Percentile": f"{vol_percentile:.0f}%",
                    "Regime": regime,
                })
            st.dataframe(pd.DataFrame(rank_data), use_container_width=True, hide_index=True)

    with vol_tab2:
        regime_sym = st.selectbox(
            "Instrument for Regime Detection",
            selected_symbols,
            format_func=sym_label,
            key="regime_sym",
        )
        if regime_sym:
            regime_data = fetch_single_ohlcv(regime_sym, period)
            if not regime_data.empty:
                r_close = regime_data["Close"]
                r_returns = r_close.pct_change().dropna()
                r_vol_20 = r_returns.rolling(20).std() * np.sqrt(252) * 100
                r_vol_60 = r_returns.rolling(60).std() * np.sqrt(252) * 100

                # Simple regime classification based on vol percentiles
                vol_median = r_vol_20.median()
                vol_75 = r_vol_20.quantile(0.75)
                vol_90 = r_vol_20.quantile(0.90)

                regimes = pd.Series("Normal", index=r_vol_20.index)
                regimes[r_vol_20 > vol_75] = "High Vol"
                regimes[r_vol_20 > vol_90] = "Crisis"
                regimes[r_vol_20 < r_vol_20.quantile(0.25)] = "Low Vol"

                fig_regime = make_subplots(
                    rows=2, cols=1,
                    shared_xaxes=True,
                    row_heights=[0.6, 0.4],
                    vertical_spacing=0.06,
                    subplot_titles=[
                        f"{sym_label(regime_sym)} Price with Volatility Regimes",
                        "Rolling Volatility & Regime Bands",
                    ],
                )

                # Price with regime background
                regime_colors_map = {
                    "Low Vol": "rgba(34,197,94,0.08)",
                    "Normal": "rgba(100,116,139,0.03)",
                    "High Vol": "rgba(245,158,11,0.08)",
                    "Crisis": "rgba(239,68,68,0.12)",
                }

                fig_regime.add_trace(go.Scatter(
                    x=r_close.index, y=r_close,
                    name="Price",
                    line=dict(color="#e2e8f0", width=2),
                ), row=1, col=1)

                # Volatility
                fig_regime.add_trace(go.Scatter(
                    x=r_vol_20.index, y=r_vol_20,
                    name="20D Vol",
                    line=dict(color="#06b6d4", width=2),
                    fill="tozeroy",
                    fillcolor="rgba(6,182,212,0.08)",
                ), row=2, col=1)
                fig_regime.add_trace(go.Scatter(
                    x=r_vol_60.index, y=r_vol_60,
                    name="60D Vol",
                    line=dict(color="#f59e0b", width=1.5, dash="dot"),
                ), row=2, col=1)

                # Regime bands
                fig_regime.add_hline(
                    y=vol_75, line_dash="dash",
                    line_color="rgba(245,158,11,0.5)",
                    row=2, col=1,
                    annotation_text="High Vol Threshold",
                )
                fig_regime.add_hline(
                    y=vol_90, line_dash="dash",
                    line_color="rgba(239,68,68,0.5)",
                    row=2, col=1,
                    annotation_text="Crisis Threshold",
                )

                fig_regime.update_layout(
                    PLOTLY_LAYOUT,
                    height=650,
                    showlegend=True,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1,
                        font=dict(size=10),
                    ),
                )
                fig_regime.update_xaxes(gridcolor="rgba(100,116,139,0.06)")
                fig_regime.update_yaxes(gridcolor="rgba(100,116,139,0.06)")
                st.plotly_chart(fig_regime, use_container_width=True, key="regime_chart")

                # Regime stats table
                st.markdown("**Performance by Volatility Regime**")
                aligned_returns = r_returns.reindex(regimes.index)
                regime_stats = []
                for regime_name in ["Low Vol", "Normal", "High Vol", "Crisis"]:
                    mask = regimes == regime_name
                    if mask.sum() > 0:
                        r_ret = aligned_returns[mask].dropna()
                        if len(r_ret) > 0:
                            regime_stats.append({
                                "Regime": regime_name,
                                "Days": mask.sum(),
                                "% of Time": f"{mask.sum() / len(regimes) * 100:.1f}%",
                                "Avg Daily Return": f"{r_ret.mean() * 100:+.3f}%",
                                "Ann. Return": f"{r_ret.mean() * 252 * 100:+.2f}%",
                                "Ann. Volatility": f"{r_ret.std() * np.sqrt(252) * 100:.2f}%",
                                "Sharpe": f"{(r_ret.mean() * 252) / (r_ret.std() * np.sqrt(252)):.3f}" if r_ret.std() > 0 else "N/A",
                                "Win Rate": f"{(r_ret > 0).sum() / len(r_ret) * 100:.1f}%",
                            })
                if regime_stats:
                    st.dataframe(
                        pd.DataFrame(regime_stats),
                        use_container_width=True,
                        hide_index=True,
                    )
            else:
                st.warning("No data available for regime detection.")

st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#         SECTION 7: MARKET BREADTH & SECTOR SNAPSHOT
# ══════════════════════════════════════════════════════════════
st.markdown('<p class="section-title">Market Snapshot</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="section-subtitle">A quick glance at the sectors and cross-asset performance</p>',
    unsafe_allow_html=True,
)

snapshot_symbols = {
    "NIFTY 50": "^NSEI",
    "SENSEX": "^BSESN",
    "BANK NIFTY": "^NSEBANK",
    "S&P 500": "^GSPC",
    "NASDAQ": "^IXIC",
    "DOW": "^DJI",
    "Gold": "GC=F",
    "Silver": "SI=F",
    "Crude Oil": "CL=F",
    "Bitcoin": "BTC-USD",
}

snap_period = st.radio(
    "Snapshot Period",
    ["1D", "1W", "1M", "3M", "6M", "1Y"],
    horizontal=True,
    index=3,
    key="snap_period",
)
snap_period_map = {
    "1D": "5d", "1W": "1mo", "1M": "1mo",
    "3M": "3mo", "6M": "6mo", "1Y": "1y",
}
snap_lookback = {"1D": 1, "1W": 5, "1M": 21, "3M": 63, "6M": 126, "1Y": 252}

snap_data = fetch_comparison_data(list(snapshot_symbols.values()), snap_period_map[snap_period])

if not snap_data.empty:
    perf_list = []
    for name, sym in snapshot_symbols.items():
        if sym in snap_data.columns:
            s = snap_data[sym].dropna()
            lookback = min(snap_lookback[snap_period], len(s) - 1)
            if lookback > 0 and len(s) > lookback:
                ret = (s.iloc[-1] / s.iloc[-lookback - 1] - 1) * 100
                perf_list.append({"name": name, "symbol": sym, "return": ret, "price": s.iloc[-1]})

    if perf_list:
        perf_list.sort(key=lambda x: x["return"], reverse=True)
        bar_colors = ["#22c55e" if p["return"] >= 0 else "#ef4444" for p in perf_list]

        fig_snap = go.Figure(go.Bar(
            x=[p["name"] for p in perf_list],
            y=[p["return"] for p in perf_list],
            marker_color=bar_colors,
            text=[f"{p['return']:+.2f}%" for p in perf_list],
            textposition="outside",
            textfont=dict(size=11, color="#e2e8f0"),
            hovertemplate="<b>%{x}</b><br>Return: %{y:+.2f}%<extra></extra>",
        ))
        fig_snap.add_hline(y=0, line_dash="dash", line_color="rgba(100,116,139,0.3)")
        fig_snap.update_layout(
            PLOTLY_LAYOUT,
            title=f"Cross-Asset Performance ({snap_period})",
            height=420,
            yaxis_title="Return (%)",
            xaxis_tickangle=-30,
        )
        st.plotly_chart(fig_snap, use_container_width=True, key="snapshot_bar")

        # Mini heatmap-style cards
        num_cards = len(perf_list)
        cols_per_row = 5
        for row in range(0, num_cards, cols_per_row):
            row_items = perf_list[row : row + cols_per_row]
            card_cols = st.columns(cols_per_row)
            for i, p in enumerate(row_items):
                with card_cols[i]:
                    delta_cls = "metric-delta-pos" if p["return"] >= 0 else "metric-delta-neg"
                    arrow = "▲" if p["return"] >= 0 else "▼"
                    st.markdown(
                        f"""
                        <div class="metric-box">
                            <div class="metric-label">{p['name']}</div>
                            <div class="metric-value">{get_currency_symbol(p['symbol'])}{p['price']:,.2f}</div>
                            <div class="{delta_cls}">{arrow} {p['return']:+.2f}%</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#                    FOOTER & LIVE TICKER BAR
# ══════════════════════════════════════════════════════════════

# Footer info
st.markdown(
    """
    <div style="text-align:center; padding:20px 0 60px; color:#475569; font-size:0.8rem;">
        <p><b>MoneyPal — Cross-Market Intelligence Platform</p>
        <p>This is a tool for educational & informational purposes only. Always do your own research before making investment decisions.</p>
    </div>
    """,
    unsafe_allow_html=True,
)



# ──── AUTO-REFRESH MECHANISM ────
# Add auto-refresh for live ticker (every 60 seconds)
st.markdown(
    """
    <script>
        // Auto-refresh ticker every 60 seconds
        // Note: Streamlit reruns on interaction; for true auto-refresh,
        // use st.rerun() with session_state timer in production
    </script>
    """,
    unsafe_allow_html=True,
)

# Session-based auto-refresh for ticker
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = time.time()

# Display refresh info in sidebar
with st.sidebar:
    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
    elapsed = int(time.time() - st.session_state.last_refresh)
    st.markdown(
        f'<p style="color:#475569; font-size:0.72rem; text-align:center;">'
        f'Last refreshed: {elapsed}s ago<br>'
        f'</p>',
        unsafe_allow_html=True,
    )
    if st.button("Refresh Data", key="global_refresh", use_container_width=True):
        st.session_state.last_refresh = time.time()
        st.cache_data.clear()
        st.rerun()
