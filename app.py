import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Quant AI Trading Desk",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- DESIGN ----------
st.markdown("""
<style>
.stApp {
    background: #0b1020;
    color: #eef2ff;
}
.block-container {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}
[data-testid="stSidebar"] {
    background: #11182b;
    border-right: 1px solid #27334d;
}
div[data-testid="stMetric"] {
    background: #151e33;
    border: 1px solid #273650;
    padding: 18px;
    border-radius: 14px;
}
h1, h2, h3 {
    color: #f1f5ff;
}
.small-muted {
    color: #9caac4;
    font-size: 14px;
}
.hero {
    background: linear-gradient(120deg, #162441, #102c35);
    padding: 25px;
    border-radius: 18px;
    border: 1px solid #30445d;
    margin-bottom: 20px;
}
div.stButton > button {
    border-radius: 10px;
    min-height: 42px;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# ---------- SIDEBAR ----------
st.sidebar.markdown("# 📈 QUANT DESK")
st.sidebar.caption("Your personal trading research workspace")

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "🏠 Home",
        "📊 Market Analyzer",
        "🛡️ Risk Manager",
        "🧪 Strategy Lab",
        "📓 Trading Journal",
        "📚 Learn Trading"
    ]
)

st.sidebar.divider()
st.sidebar.caption("MODE: RESEARCH / DEMO")
st.sidebar.caption("No live trades are placed by this app.")

# ---------- HEADER ----------
st.markdown(
    '<div class="hero"><h1>Quant AI Trading Desk</h1>'
    '<p>Understand the market. Test your rules. Manage risk.</p>'
    '<p class="small-muted">A beginner-friendly trading research platform</p>'
    '</div>',
    unsafe_allow_html=True
)

# ---------- HELPERS ----------
def calculate_rsi(series, period=14):
    delta = series.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.rolling(period).mean()
    avg_loss = losses.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def analyze_data(df):
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]

    if "close" not in df.columns:
        raise ValueError(
            "Your CSV needs a column named 'close'. "
            "Optional columns: open, high, low, volume, time."
        )

    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["close"]).copy()

    if len(df) < 30:
        raise ValueError("Upload at least 30 valid closing prices.")

    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["rsi"] = calculate_rsi(df["close"])
    df["macd"] = (
        df["close"].ewm(span=12, adjust=False).mean()
        - df["close"].ewm(span=26, adjust=False).mean()
    )
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

    return df

# ---------- HOME ----------
if page == "🏠 Home":
    st.subheader("Your overview")

    a, b, c = st.columns(3)
    a.metric("Workspace", "Ready")
    b.metric("Analysis mode", "Historical data")
    c.metric("Live trading", "Disabled")

    st.markdown("### What would you like to do?")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📊 Analyze a market")
        st.write(
            "Upload historical price data and review trend, "
            "momentum, and indicator readings."
        )

        st.markdown("#### 🛡️ Calculate trading risk")
        st.write(
            "Estimate risk amount, position size, and "
            "potential reward before planning a demo trade."
        )

    with col2:
        st.markdown("#### 🧪 Explore strategies")
        st.write(
            "Learn how trend, breakout, momentum, and range "
            "strategies work."
        )

        st.markdown("#### 📓 Keep a journal")
        st.write(
            "Record demo trades and review what you learned. "
            "Entries remain in this browser session."
        )

    st.info(
        "Start with demo trading and historical data. "
        "This app does not provide guaranteed predictions or "
        "execute trades."
    )

# ---------- MARKET ANALYZER ----------
elif page == "📊 Market Analyzer":
    st.subheader("Market Analyzer")
    st.write("Upload a CSV file containing historical price data.")

    st.caption(
        "Required column: close. Optional columns: open, high, low, "
        "volume, time. Use historical data from a source you trust."
    )

    uploaded = st.file_uploader("Choose your price-data CSV", type=["csv"])

    if uploaded:
        try:
            raw = pd.read_csv(uploaded)
            df = analyze_data(raw)

            latest = df.iloc[-1]
            previous = df.iloc[-2]

            x1, x2, x3 = st.columns(3)
            x1.metric(
                "Latest close",
                f"{latest['close']:.5f}",
                f"{latest['close'] - previous['close']:.5f}"
            )
            x2.metric("EMA 20", f"{latest['ema20']:.5f}")
            x3.metric(
                "RSI (14)",
                "Calculating" if pd.isna(latest["rsi"])
                else f"{latest['rsi']:.1f}"
            )

            st.markdown("### Price and trend")
            chart_data = df[["close", "ema20", "ema50"]].tail(200)
            st.line_chart(chart_data)

            st.markdown("### What the indicators mean")

            if latest["ema20"] > latest["ema50"]:
                st.success(
                    "Trend clue: EMA 20 is above EMA 50. "
                    "This suggests an upward trend in the selected data."
                )
            elif latest["ema20"] < latest["ema50"]:
                st.warning(
                    "Trend clue: EMA 20 is below EMA 50. "
                    "This suggests a downward trend in the selected data."
                )
            else:
                st.info("The two moving averages are currently equal.")

            if pd.isna(latest["rsi"]):
                st.info("RSI is unavailable for the latest candle.")
            elif latest["rsi"] >= 70:
                st.warning(
                    "RSI is 70 or higher. Momentum may be strong, "
                    "but the market may also be overextended."
                )
            elif latest["rsi"] <= 30:
                st.warning(
                    "RSI is 30 or lower. Selling pressure may be strong; "
                    "this does not guarantee a reversal."
                )
            else:
                st.info(
                    f"RSI is {latest['rsi']:.1f}, between 30 and 70."
                )

            if latest["macd"] > latest["macd_signal"]:
                st.write("MACD clue: MACD is above its signal line.")
            else:
                st.write("MACD clue: MACD is at or below its signal line.")

            st.markdown("### Latest data")
            st.dataframe(df.tail(20), use_container_width=True)

        except Exception as e:
            st.error(f"Could not analyze this file: {e}")

    else:
        st.info("Upload a CSV to begin. No live market feed is connected.")

# ---------- RISK MANAGER ----------
elif page == "🛡️ Risk Manager":
    st.subheader("Risk Manager")
    st.write("Plan risk for a hypothetical trade.")

    balance = st.number_input(
        "Account balance", min_value=0.0, value=10000.0, step=100.0
    )
    risk_percent = st.slider("Risk per trade (%)", 0.1, 2.0, 0.5, 0.1)
    stop_distance = st.number_input(
        "Stop Loss distance (points/price amount)", min_value=0.00001, value=10.0, step=1.0
    )
    
    risk_amount = balance * (risk_percent / 100.0)
    position_size = risk_amount / stop_distance if stop_distance > 0 else 0.0
    
    st.markdown("### Risk Calculation Results")
    r1, r2 = st.columns(2)
    r1.metric("Cash Amount at Risk", f"${risk_amount:.2f}")
    r2.metric("Suggested Position Size", f"{position_size:.4f}")

# ---------- OTHER PAGES ----------
else:
    st.subheader(page)
    st.write(f"Welcome to the {page} section! This feature area is ready for custom analysis layouts.")
    
