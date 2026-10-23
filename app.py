import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from plotly.subplots import make_subplots

# =========================================================
# CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Quant AI Trading Desk",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.stApp {
    background: #080d18;
    color: #e5eaf5;
}
.block-container {
    max-width: 1500px;
    padding-top: 1.3rem;
    padding-bottom: 3rem;
}
[data-testid="stSidebar"] {
    background: #0d1423;
    border-right: 1px solid #263249;
}
[data-testid="stMetric"] {
    background: #111b2d;
    border: 1px solid #27364e;
    border-radius: 12px;
    padding: 14px;
}
.panel {
    background: #111b2d;
    border: 1px solid #27364e;
    border-radius: 14px;
    padding: 18px;
    margin-bottom: 12px;
}
.muted {
    color: #9aa9c2;
    font-size: 13px;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# SAMPLE DATA
# =========================================================

@st.cache_data
def generate_sample_data():
    rng = np.random.default_rng(42)
    n = 500

    dates = pd.date_range(
        end=pd.Timestamp.now().floor("h"),
        periods=n,
        freq="h"
    )

    # Synthetic prices for testing only.
    returns = rng.normal(0.00015, 0.004, n)
    close = 1.0800 * np.exp(np.cumsum(returns))

    previous = np.concatenate([[close[0]], close[:-1]])

    open_price = previous
    high = np.maximum(open_price, close) * (
        1 + rng.uniform(0.0001, 0.002, n)
    )
    low = np.minimum(open_price, close) * (
        1 - rng.uniform(0.0001, 0.002, n)
    )
    volume = rng.integers(100, 2000, n)

    return pd.DataFrame({
        "time": dates,
        "open": open_price,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume
    })


def prepare_data(df):
    df = df.copy()
    df.columns = [
        str(c).strip().lower() for c in df.columns
    ]

    # Recognise common alternative column names.
    aliases = {
        "date": "time",
        "datetime": "time",
        "timestamp": "time",
        "tick_volume": "volume"
    }

    df = df.rename(columns={
        k: v for k, v in aliases.items()
        if k in df.columns and v not in df.columns
    })

    if "close" not in df.columns:
        raise ValueError(
            "Your CSV must contain a 'close' column."
        )

    df["close"] = pd.to_numeric(
        df["close"], errors="coerce"
    )

    # If only closing prices are supplied, construct
    # approximate candles. These are NOT actual OHLC prices.
    if not all(c in df.columns for c in ["open", "high", "low"]):
        df["open"] = df["close"].shift(1)
        df["open"] = df["open"].fillna(df["close"])
        df["high"] = df[["open", "close"]].max(axis=1)
        df["low"] = df[["open", "close"]].min(axis=1)

    for col in ["open", "high", "low"]:
        df[col] = pd.to_numeric(
            df[col], errors="coerce"
        )

    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(
            df["volume"], errors="coerce"
        ).fillna(0)
    else:
        df["volume"] = 0

    df = df.dropna(
        subset=["open", "high", "low", "close"]
    ).copy()

    if "time" in df.columns:
        df["time"] = pd.to_datetime(
            df["time"], errors="coerce"
        )
        df = df.dropna(subset=["time"])
        df = df.sort_values("time")
    else:
        df["time"] = np.arange(len(df))

    if len(df) < 60:
        raise ValueError(
            "Please provide at least 60 valid candles."
        )

    if (df["close"] <= 0).any():
        raise ValueError(
            "Closing prices must be positive."
        )

    return df.reset_index(drop=True)


# =========================================================
# INDICATORS
# =========================================================

def add_indicators(df):
    df = df.copy()

    close = df["close"]
    high = df["high"]
    low = df["low"]

    df["EMA20"] = close.ewm(
        span=20, adjust=False
    ).mean()

    df["EMA50"] = close.ewm(
        span=50, adjust=False
    ).mean()

    df["EMA200"] = close.ewm(
        span=200, adjust=False
    ).mean()

    df["SMA20"] = close.rolling(20).mean()

    std = close.rolling(20).std()

    df["BB_UPPER"] = df["SMA20"] + 2 * std
    df["BB_LOWER"] = df["SMA20"] - 2 * std

    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / 14, min_periods=14, adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / 14, min_periods=14, adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    df["RSI"] = 100 - (100 / (1 + rs))

    ema12 = close.ewm(
        span=12, adjust=False
    ).mean()

    ema26 = close.ewm(
        span=26, adjust=False
    ).mean()

    df["MACD"] = ema12 - ema26

    df["MACD_SIGNAL"] = df["MACD"].ewm(
        span=9, adjust=False
    ).mean()

    df["MACD_HIST"] = (
        df["MACD"] - df["MACD_SIGNAL"]
    )

    previous_close = close.shift(1)

    true_range = pd.concat([
        high - low,
        (high - previous_close).abs(),
        (low - previous_close).abs()
    ], axis=1).max(axis=1)

    df["ATR"] = true_range.ewm(
        alpha=1 / 14, min_periods=14, adjust=False
    ).mean()

    df["DONCHIAN_HIGH"] = high.shift(1).rolling(20).max()
    df["DONCHIAN_LOW"] = low.shift(1).rolling(20).min()

    return df


# =========================================================
# CHARTS
# =========================================================

def make_market_chart(df, show_ema, show_bollinger):
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.76, 0.24],
        subplot_titles=("Price Action", "Volume")
    )

    fig.add_trace(
        go.Candlestick(
            x=df["time"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="Candles",
            increasing_line_color="#00d6a3",
            decreasing_line_color="#ff5576"
        ),
        row=1,
        col=1
    )

    if show_ema:
        for col, color in [
            ("EMA20", "#f5c451"),
            ("EMA50", "#4ca7ff"),
            ("EMA200", "#bd8cff")
        ]:
            fig.add_trace(
                go.Scatter(
                    x=df["time"],
                    y=df[col],
                    name=col,
                    mode="lines",
                    line=dict(width=1.5, color=color)
                ),
                row=1,
                col=1
            )

    if show_bollinger:
        fig.add_trace(
            go.Scatter(
                x=df["time"],
                y=df["BB_UPPER"],
                name="Bollinger upper",
                line=dict(color="#8290a8", width=1, dash="dot")
            ),
            row=1,
            col=1
        )

        fig.add_trace(
            go.Scatter(
                x=df["time"],
                y=df["BB_LOWER"],
                name="Bollinger lower",
                line=dict(color="#8290a8", width=1, dash="dot")
            ),
            row=1,
            col=1
        )

    volume_colors = np.where(
        df["close"] >= df["open"],
        "#00b894",
        "#e74c68"
    )

    fig.add_trace(
        go.Bar(
            x=df["time"],
            y=df["volume"],
            name="Volume",
            marker_color=volume_colors
        ),
        row=2,
        col=1
    )

    fig.update_layout(
        template="plotly_dark",
        height=650,
        paper_bgcolor="#080d18",
        plot_bgcolor="#080d18",
        margin=dict(l=10, r=10, t=55, b=10),
        legend=dict(
            orientation="h",
            y=1.03,
            x=0
        ),
        xaxis_rangeslider_visible=False,
        hovermode="x unified"
    )

    fig.update_xaxes(
        showgrid=True,
        gridcolor="#1b2638"
    )

    fig.update_yaxes(
        showgrid=True,
        gridcolor="#1b2638"
    )

    return fig


def make_indicator_chart(df, indicator):
    fig = go.Figure()

    if indicator == "RSI":
        fig.add_trace(go.Scatter(
            x=df["time"],
            y=df["RSI"],
            name="RSI",
            line=dict(color="#bd8cff", width=2)
        ))

        fig.add_hline(
            y=70, line_dash="dash",
            line_color="#ff5576"
        )

        fig.add_hline(
            y=30, line_dash="dash",
            line_color="#00d6a3"
        )

    elif indicator == "MACD":
        fig.add_trace(go.Scatter(
            x=df["time"],
            y=df["MACD"],
            name="MACD",
            line=dict(color="#4ca7ff", width=2)
        ))

        fig.add_trace(go.Scatter(
            x=df["time"],
            y=df["MACD_SIGNAL"],
            name="Signal",
            line=dict(color="#f5c451", width=2)
        ))

        fig.add_trace(go.Bar(
            x=df["time"],
            y=df["MACD_HIST"],
            name="Histogram"
        ))

    else:
        fig.add_trace(go.Scatter(
            x=df["time"],
            y=df["ATR"],
            name="ATR",
            line=dict(color="#f5c451", width=2)
        ))

    fig.update_layout(
        template="plotly_dark",
        height=300,
        paper_bgcolor="#080d18",
        plot_bgcolor="#080d18",
        margin=dict(l=10, r=10, t=30, b=10),
        hovermode="x unified"
    )

    return fig


# =========================================================
# STRATEGY BACKTEST
# =========================================================

def run_backtest(df, strategy, fast, slow, fee_bps):
    data = df.copy()

    close = data["close"]

    if strategy == "EMA Crossover":
        fast_ema = close.ewm(
            span=fast, adjust=False
        ).mean()

        slow_ema = close.ewm(
            span=slow, adjust=False
        ).mean()

        signal = (fast_ema > slow_ema).astype(int)

    elif strategy == "Donchian Breakout":
        previous_high = data["high"].shift(1).rolling(
            fast
        ).max()

        previous_low = data["low"].shift(1).rolling(
            fast
        ).min()

        signal = pd.Series(0, index=data.index)

        signal[close > previous_high] = 1
        signal[close < previous_low] = 0

        signal = signal.ffill().fillna(0).astype(int)

    else:
        rsi = data["RSI"]

        signal = pd.Series(0, index=data.index)

        signal[rsi < 30] = 1
        signal[rsi > 50] = 0

        signal = signal.ffill().fillna(0).astype(int)

    # Signals are delayed one candle to avoid using
    # the current candle's close before it has closed.
    position = signal.shift(1).fillna(0)

    market_returns = close.pct_change().fillna(0)

    turnover = position.diff().abs().fillna(
        position.abs()
    )

    costs = turnover * (fee_bps / 10000)

    strategy_returns = (
        position * market_returns - costs
    )

    equity = (1 + strategy_returns).cumprod()

    total_return = (equity.iloc[-1] - 1) * 100

    running_peak = equity.cummax()

    drawdown = (
        equity / running_peak - 1
    ) * 100

    max_drawdown = drawdown.min()

    active = strategy_returns[position != 0]

    wins = active[active > 0]
    losses = active[active < 0]

    win_rate = (
        len(wins) / len(active) * 100
        if len(active) else 0
    )

    profit_factor = (
        wins.sum() / abs(losses.sum())
        if len(losses) and abs(losses.sum()) > 0
        else np.nan
    )

    periods_per_year = 24 * 365

    volatility = strategy_returns.std()

    sharpe = (
        strategy_returns.mean()
        / volatility
        * np.sqrt(periods_per_year)
        if volatility and volatility > 0
        else np.nan
    )

    data["Position"] = position
    data["Strategy Return"] = strategy_returns
    data["Equity"] = equity
    data["Drawdown"] = drawdown

    metrics = {
        "Return (%)": total_return,
        "Max drawdown (%)": max_drawdown,
        "Win rate (%)": win_rate,
        "Profit factor": profit_factor,
        "Sharpe (annualized)": sharpe
    }

    return data, metrics


# =========================================================
# NAVIGATION
# =========================================================

st.sidebar.markdown("# QUANT DESK")
st.sidebar.caption("Trading Research Terminal")

page = st.sidebar.radio(
    "WORKSPACE",
    [
        "Overview",
        "Market Analyzer",
        "Strategy Lab",
        "Risk Manager",
        "Trading Journal",
        "Learn"
    ]
)

st.sidebar.divider()

source = st.sidebar.radio(
    "MARKET DATA",
    ["Sample Data", "Upload CSV"]
)

uploaded_file = None

if source == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader(
        "Upload OHLC CSV",
        type=["csv"]
    )

st.sidebar.caption(
    "Sample data is synthetic. "
    "No live market connection is active."
)

# =========================================================
# LOAD ACTIVE DATA
# =========================================================

data_error = None
data_source_name = "Synthetic sample data"

try:
    if source == "Upload CSV" and uploaded_file is not None:
        raw = pd.read_csv(uploaded_file)
        market_df = prepare_data(raw)
        data_source_name = "Uploaded CSV"

    elif source == "Upload CSV":
        market_df = None

    else:
        market_df = generate_sample_data()
        market_df = prepare_data(market_df)

    if market_df is not None:
        market_df = add_indicators(market_df)

except Exception as exc:
    market_df = None
    data_error = str(exc)

# =========================================================
# HEADER
# =========================================================

st.markdown("## QUANT AI TRADING DESK")
st.caption(
    "Market research • Technical analysis • Strategy testing"
)

if source == "Sample Data":
    st.warning(
        "DEMO MODE — prices are synthetic, not live market prices."
    )

if data_error:
    st.error(data_error)

# =========================================================
# OVERVIEW
# =========================================================

if page == "Overview":
    st.markdown("### Market Overview")

    if market_df is not None:
        latest = market_df.iloc[-1]
        previous = market_df.iloc[-2]

        change_pct = (
            latest["close"] / previous["close"] - 1
        ) * 100

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Latest Close",
            f"{latest['close']:.5f}",
            f"{change_pct:+.2f}%"
        )

        c2.metric(
            "EMA 20",
            f"{latest['EMA20']:.5f}"
        )

        c3.metric(
            "RSI",
            f"{latest['RSI']:.1f}"
            if pd.notna(latest["RSI"]) else "N/A"
        )

        c4.metric(
            "ATR",
            f"{latest['ATR']:.5f}"
            if pd.notna(latest["ATR"]) else "N/A"
        )

        st.plotly_chart(
            make_market_chart(
                market_df.tail(200), True, True
            ),
            use_container_width=True,
            key="overview_chart"
        )

        st.markdown("### Market Conditions")

        if latest["EMA20"] > latest["EMA50"]:
            st.success(
                "Trend condition: EMA 20 is above EMA 50."
            )
        else:
            st.info(
                "Trend condition: EMA 20 is at or below EMA 50."
            )

        if latest["RSI"] >= 70:
            st.warning(
                "RSI is elevated. This does not automatically mean sell."
            )
        elif latest["RSI"] <= 30:
            st.warning(
                "RSI is low. This does not automatically mean buy."
            )
        else:
            st.info("RSI is between 30 and 70.")

    else:
        st.info("Select Sample Data or upload a valid CSV.")

    st.markdown("### Workspace")

    a, b, c = st.columns(3)

    a.markdown(
        '<div class="panel"><h3>📈 Analyze</h3>'
        '<p>Explore candles, trend, and momentum.</p></div>',
        unsafe_allow_html=True
    )

    b.markdown(
        '<div class="panel"><h3>🧪 Backtest</h3>'
        '<p>Test rule-based strategies on historical prices.</p></div>',
        unsafe_allow_html=True
    )

    c.markdown(
        '<div class="panel"><h3>🛡️ Risk</h3>'
        '<p>Calculate hypothetical risk and reward.</p></div>',
        unsafe_allow_html=True
    )


# =========================================================
# MARKET ANALYZER
# =========================================================

elif page == "Market Analyzer":
    st.markdown("### Technical Analysis Terminal")

    if market_df is None:
        st.info("Select Sample Data or upload a CSV.")
        st.stop()

    c1, c2, c3 = st.columns(3)

    with c1:
        bars = st.selectbox(
            "Candles to display",
            [50, 100, 200, 300, 500],
            index=2
        )

    with c2:
        show_ema = st.checkbox(
            "Show moving averages", value=True
        )

    with c3:
        show_bb = st.checkbox(
            "Show Bollinger Bands", value=True
        )

    visible = market_df.tail(bars)

    st.plotly_chart(
        make_market_chart(visible, show_ema, show_bb),
        use_container_width=True,
        key="analyzer_chart"
    )

    st.markdown("### Indicator Monitor")

    i1, i2, i3, i4 = st.columns(4)

    last = market_df.iloc[-1]

    i1.metric(
        "RSI (14)",
        f"{last['RSI']:.2f}"
        if pd.notna(last["RSI"]) else "N/A"
    )

    i2.metric(
        "MACD",
        f"{last['MACD']:.5f}"
        if pd.notna(last["MACD"]) else "N/A"
    )

    i3.metric(
        "ATR (14)",
        f"{last['ATR']:.5f}"
        if pd.notna(last["ATR"]) else "N/A"
    )

    i4.metric(
        "EMA 20 vs 50",
        "Above" if last["EMA20"] > last["EMA50"]
        else "Below"
    )

    indicator = st.selectbox(
        "Indicator chart",
        ["RSI", "MACD", "ATR"]
    )

    st.plotly_chart(
        make_indicator_chart(visible, indicator),
        use_container_width=True,
        key="indicator_chart"
    )

    with st.expander("View candle data"):
        st.dataframe(
            market_df.tail(50),
            use_container_width=True
        )

    st.download_button(
        "Export analyzed data",
        data=market_df.to_csv(index=False).encode("utf-8"),
        file_name="quant_analyzed_data.csv",
        mime="text/csv"
    )


# =========================================================
# STRATEGY LAB
# =========================================================

elif page == "Strategy Lab":
    st.markdown("### Strategy Backtesting Lab")

    if market_df is None:
        st.info("Select Sample Data or upload a CSV.")
        st.stop()

    st.caption(
        "Historical simulation only. Results are not predictions "
        "and do not guarantee future performance."
    )

    col1, col2 = st.columns(2)

    with col1:
        strategy = st.selectbox(
            "Strategy",
            [
                "EMA Crossover",
                "Donchian Breakout",
                "RSI Mean Reversion"
            ]
        )

        fast = st.slider(
            "Fast period / breakout period",
            5, 50, 20
        )

    with col2:
        slow = st.slider(
            "Slow EMA period",
            20, 150, 50
        )

        fee_bps = st.number_input(
            "Assumed cost per position change (basis points)",
            min_value=0.0,
            max_value=100.0,
            value=2.0,
            step=0.5
        )

    if strategy == "EMA Crossover" and fast >= slow:
        st.warning(
            "For an EMA crossover, choose a fast period smaller "
            "than the slow period."
        )

    else:
        resu
