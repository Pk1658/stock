import streamlit as st
import yfinance as yf
import pandas as pd

st.set_page_config(page_title="Nifty Stock Screener", layout="centered")
st.title("📈 Nifty Stock Screener")
st.caption("Educational tool. Not investment advice.")

NIFTY = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "HINDUNILVR", "ITC",
    "SBIN", "BHARTIARTL", "KOTAKBANK", "LT", "AXISBANK", "ASIANPAINT",
    "MARUTI", "SUNPHARMA", "TITAN", "ULTRACEMCO", "WIPRO", "NESTLEIND",
    "TATAMOTORS", "TATASTEEL", "POWERGRID", "NTPC", "ONGC", "ADANIENT",
]


def rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0).rolling(n).mean()
    down = (-d.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + up / down)


def analyse(sym):
    df = yf.download(sym + ".NS", period="1y", progress=False, auto_adjust=True)
    if df.empty or len(df) < 200:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    c, v = df["Close"], df["Volume"]
    score, why = 0, []

    r = rsi(c).iloc[-1]
    if r < 35:
        score += 2; why.append("RSI oversold")
    elif r < 60:
        score += 1; why.append("RSI healthy")
    elif r > 70:
        score -= 2; why.append("RSI overbought")

    dma50, dma200 = c.rolling(50).mean().iloc[-1], c.rolling(200).mean().iloc[-1]
    if dma50 > dma200:
        score += 2; why.append("50DMA > 200DMA")
    else:
        score -= 2; why.append("50DMA < 200DMA")
    if c.iloc[-1] > dma50:
        score += 1; why.append("Price > 50DMA")

    ema12, ema26 = c.ewm(span=12).mean(), c.ewm(span=26).mean()
    macd = ema12 - ema26
    if macd.iloc[-1] > macd.ewm(span=9).mean().iloc[-1]:
        score += 1; why.append("MACD bullish")

    if v.iloc[-1] > 1.5 * v.rolling(20).mean().iloc[-1] and c.iloc[-1] > c.iloc[-2]:
        score += 1; why.append("Volume breakout")

    signal = "BUY" if score >= 4 else "AVOID" if score <= 0 else "HOLD/WATCH"
    return {"Stock": sym, "Price": round(float(c.iloc[-1]), 2), "RSI": round(float(r), 1),
            "Score": score, "Signal": signal, "Reasons": ", ".join(why)}


@st.cache_data(ttl=3600)
def run_all():
    rows = [analyse(s) for s in NIFTY]
    return pd.DataFrame([r for r in rows if r]).sort_values("Score", ascending=False)


with st.spinner("Data load thai rahyo che..."):
    table = run_all()

pick = st.multiselect("Signal filter", ["BUY", "HOLD/WATCH", "AVOID"], default=["BUY", "HOLD/WATCH", "AVOID"])
st.dataframe(table[table["Signal"].isin(pick)], use_container_width=True, hide_index=True)

sym = st.selectbox("Chart jovu che?", table["Stock"])
hist = yf.download(sym + ".NS", period="1y", progress=False, auto_adjust=True)
if isinstance(hist.columns, pd.MultiIndex):
    hist.columns = hist.columns.get_level_values(0)
hist["50DMA"] = hist["Close"].rolling(50).mean()
hist["200DMA"] = hist["Close"].rolling(200).mean()
st.line_chart(hist[["Close", "50DMA", "200DMA"]])

st.warning("Aa tool educational che. Share market ma nuksan thai shake che. SEBI-registered advisor ni salah lo.")
