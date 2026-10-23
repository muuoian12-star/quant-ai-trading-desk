import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Quant AI Trading Desk",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Quant AI Trading Desk")
st.caption("Multi-strategy market research and risk analysis")

st.warning(
    "Research mode only. Signals are rule-based, not guaranteed "
    "predictions. No live trading is enabled."
)

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.header("Market Settings")

symbol = st.sidebar.selectbox(
    "Instrument",
    ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD",
     "BTCUSD", "US30", "Volatility Index"]
)

timeframe = st.sidebar.selectbox(
    "Timeframe",
    ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
)

risk_percent = st.sidebar.number_input(
    "Risk per trade (%)",
    min_value=0.1,
    max_value=1.0,
    value=0.5,
    step=0.1
)

account_balance = st.sidebar.number_input(
    "Demo account balance",
    min_value=1.0,
    value=10000.0,
    step=100.0
)

st.sidebar.info(
    "The selected balance is for calculations only. "
    "It is not connected to a broker."
)

risk_cash = account_balance * risk_percent / 100

# -----------------------------
# DASHBOARD OVERVIEW
# -----------------------------
col1, col2, col3 = st.columns(3)

col1.metric("Selected Market", symbol)
col2.metric("Timeframe", timeframe)
col3.metric("Illustrative Risk Budget", f"{risk_cash:.2f}")

st.divider()

# -----------------------------
# STRATEGY FAMILIES
# -----------------------------
st.subheader("🧠 Strategy Centre")

strategies = {
    "Trend-following": "EMA, ADX",
    "Breakout": "Support/resistance, Donchian",
    "Momentum": "MACD, Awesome Oscillator",
    "Range trading": "RSI, Stochastic",
    "News trading": "Economic calendar, volatility",
    "Position trading": "Higher timeframes, macro data",
    "Carry trading": "Interest rates, swap costs",
    "Systematic / Grid": "Rule-based execution and risk limits"
}

for name, tools in strategies.items():
    with st.expander(name):
        st.write(f"**Core tools:** {tools}")

        if name in [
            "Trend-following",
            "Breakout",
            "Momentum",
            "Range trading"
        ]:
            st.write(
                "Technical rules can be evaluated using suitable "
                "historical OHLCV candle data."
            )
        else:
            st.write(
                "This strategy requires additional data or validated "
                "rules before the system can evaluate it."
            )

# -----------------------------
# CSV UPLOAD
# -----------------------------
st.divider()
st.subheader("📊 Analyse Historical Market Data")

st.write(
    "Upload a CSV containing candle data. Required columns: "
    "Open, High, Low, Close. Volume is optional."
)

uploaded_file = st.file_uploader(
    "Upload OHLCV CSV",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        df.columns = [
            str(c).strip().lower() for c in df.columns
        ]

        required = ["open", "high", "low", "close"]
        missing = [c for c in required if c not in df.columns]

        if missing:
            st.error(
                "Missing columns: " + ", ".join(missing)
            )
        else:
            for col in required:
                df[col] = pd.to_numeric(
                    df[col], errors="coerce"
                )

            df = df.dropna(subset=required).reset_index(drop=True)

            if len(df) < 30:
                st.error(
                    "Upload at least 30 valid candles for this "
                    "initial analysis."
                )
            else:
                # Indicators
                df["EMA20"] = df["close"].ewm(
                    span=20, adjust=False
                ).mean()

                df["EMA50"] = df["close"].ewm(
                    span=50, adjust=False
                ).mean()

                delta = df["close"].diff()
                gain = delta.clip(lower=0)
                loss = -delta.clip(upper=0)

                avg_gain = gain.ewm(
                    alpha=1/14, min_periods=14,
                    adjust=False
                ).mean()

                avg_loss = loss.ewm(
                    alpha=1/14, min_periods=14,
                    adjust=False
                ).mean()

                rs = avg_gain / avg_loss.replace(0, np.nan)
                df["RSI"] = 100 - (100 / (1 + rs))

                ema12 = df["close"].ewm(
                    span=12, adjust=False
                ).mean()

                ema26 = df["close"].ewm(
                    span=26, adjust=False
                ).mean()

                df["MACD"] = ema12 - ema26
                df["MACD_signal"] = df["MACD"].ewm(
                    span=9, adjust=False
                ).mean()

                previous_close = df["close"].shift(1)

                true_range = pd.concat(
                    [
                        df["high"] - df["low"],
                        (df["high"] - previous_close).abs(),
                        (df["low"] - previous_close).abs()
                    ],
                    axis=1
                ).max(axis=1)

                df["ATR"] = true_range.rolling(14).mean()

                latest = df.iloc[-1]

                st.success(
                    f"Loaded {len(df)} candles successfully."
                )

                a, b, c = st.columns(3)

                a.metric(
                    "Latest Close",
                    f"{latest['close']:.5f}"
                )

                b.metric(
                    "RSI (14)",
                    (
                        f"{latest['RSI']:.2f}"
                        if pd.notna(latest["RSI"])
                        else "Unavailable"
                    )
                )

                c.metric(
                    "ATR (14)",
                    (
                        f"{latest['ATR']:.5f}"
                        if pd.notna(latest["ATR"])
                        else "Unavailable"
                    )
                )

                # Basic rule-based verdicts
                trend = (
                    "BUY"
                    if latest["EMA20"] > latest["EMA50"]
                    else "SELL"
                    if latest["EMA20"] < latest["EMA50"]
                    else "WAIT"
                )

                momentum = (
                    "BUY"
                    if latest["MACD"] > latest["MACD_signal"]
                    else "SELL"
                    if latest["MACD"] < latest["MACD_signal"]
                    else "WAIT"
                )

                rsi = latest["RSI"]

                if pd.isna(rsi):
                    range_signal = "WAIT"
                elif rsi < 30:
                    range_signal = "POTENTIAL BUY SETUP"
                elif rsi > 70:
                    range_signal = "POTENTIAL SELL SETUP"
                else:
                    range_signal = "WAIT"

                previous_high = df["high"].iloc[-21:-1].max()
                previous_low = df["low"].iloc[-21:-1].min()

                if latest["close"] > previous_high:
                    breakout = "UPWARD BREAKOUT"
                elif latest["close"] < previous_low:
                    breakout = "DOWNWARD BREAKOUT"
                else:
                    breakout = "WAIT"

                st.subheader("Rule-Based Analysis")

                results = pd.DataFrame({
                    "Strategy": [
                        "Trend-following",
                        "Momentum",
                        "Range trading",
                        "Breakout"
                    ],
                    "Result": [
                        trend,
                        momentum,
                        range_signal,
                        breakout
                    ]
                })

                st.dataframe(
                    results,
                    use_container_width=True,
                    hide_index=True
                )

                st.line_chart(
                    df[["close", "EMA20", "EMA50"]].tail(150)
                )

                st.caption(
                    "These are preliminary technical rules, not "
                    "validated trading signals. The EMA rules do "
                    "not yet include ADX confirmation, and the "
                    "range rules do not confirm that the market "
                    "is actually ranging. News, carry, position "
                    "and grid strategies are not implemented."
                )

                st.download_button(
                    "Download Analysed Data",
                    df.to_csv(index=False),
                    file_name="quant_analysis.csv",
                    mime="text/csv"
                )

    except Exception as e:
        st.error(
            f"Could not analyse this CSV: {e}"
        )

# -----------------------------
# RISK CALCULATOR
# -----------------------------
st.divider()
st.subheader("🛡️ Risk Calculator")

r1, r2, r3 = st.columns(3)

with r1:
    entry = st.number_input(
        "Hypothetical entry price",
        min_value=0.00001,
        value=1.10000,
        format="%.5f"
    )

with r2:
    stop = st.number_input(
        "Hypothetical stop-loss price",
        min_value=0.00001,
        value=1.09500,
        format="%.5f"
    )

with r3:
    reward_ratio = st.selectbox(
        "Target risk-to-reward",
        [2, 3],
        format_func=lambda x: f"1:{x}"
    )

if entry > 0 and stop > 0 and entry != stop:
    distance = abs(entry - stop)
    direction = "BUY" if stop < entry else "SELL"

    target = (
        entry + distance * reward_ratio
        if direction == "BUY"
        else entry - distance * reward_ratio
    )

    st.write(f"**Hypothetical direction:** {direction}")
    st.write(f"**Entry:** {entry:.5f}")
    st.write(f"**Stop-loss:** {stop:.5f}")
    st.write(f"**Target:** {target:.5f}")
    st.write(f"**Price distance to stop:** {distance:.5f}")
    st.write(f"**Risk budget:** {risk_cash:.2f} account-currency units")

    st.info(
        "This calculator does not determine a valid market entry "
        "or calculate broker lot size. Lot sizing requires the "
        "instrument's contract specifications, tick value, "
        "account currency, spread and trading costs."
    )
else:
    st.error("Entry and stop-loss must be different positive prices.")

st.divider()

st.caption(
    "QUANT AI TRADING DESK | Research prototype | "
    "No broker connection | No guaranteed returns"
)
