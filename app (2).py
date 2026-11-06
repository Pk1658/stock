import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

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

    price = float(c.iloc[-1])
    # 7 divas ~ 5 trading days: volatility aadharit expected range (~68% sambhavna)
    sigma = float(np.log(c / c.shift(1)).tail(60).std())
    move = sigma * (5 ** 0.5)
    low, high = price * np.exp(-move), price * np.exp(move)
    # ATR aadharit target / stoploss (sirf BUY mate)
    tr = pd.concat([df["High"] - df["Low"],
                    (df["High"] - c.shift()).abs(),
                    (df["Low"] - c.shift()).abs()], axis=1).max(axis=1)
    atr = float(tr.rolling(14).mean().iloc[-1])
    target = round(price + 2 * atr, 2) if signal == "BUY" else None
    stop = round(price - atr, 2) if signal == "BUY" else None

    return {"Stock": sym, "Price": round(price, 2), "RSI": round(float(r), 1),
            "Score": score, "Signal": signal,
            "7D Low": round(low, 2), "7D High": round(high, 2),
            "Target": target, "Stoploss": stop, "Reasons": ", ".join(why)}


@st.cache_data(ttl=3600)
def run(symbols):
    rows = [analyse(s) for s in symbols]
    rows = [r for r in rows if r]
    return pd.DataFrame(rows).sort_values("Score", ascending=False) if rows else pd.DataFrame()


txt = st.text_input("Stock na naam (NSE symbol, comma thi alag karo)", "RELIANCE, TCS, INFY")
add_nifty = st.checkbox("Nifty 25 stocks pan umero")

symbols = [s.strip().upper().replace(".NS", "") for s in txt.split(",") if s.strip()]
if add_nifty:
    symbols += [s for s in NIFTY if s not in symbols]

if not symbols:
    st.info("Upar koi stock no symbol lakho, jemake TATAMOTORS.")
    st.stop()

with st.spinner("Data load thai rahyo che..."):
    table = run(tuple(symbols))

missing = [s for s in symbols if table.empty or s not in set(table["Stock"])]
if missing:
    st.error("Aa stocks na malya (symbol check karo): " + ", ".join(missing))
if table.empty:
    st.stop()

st.dataframe(table, use_container_width=True, hide_index=True)

sym = st.selectbox("Chart jovu che?", table["Stock"])
hist = yf.download(sym + ".NS", period="1y", progress=False, auto_adjust=True)
if isinstance(hist.columns, pd.MultiIndex):
    hist.columns = hist.columns.get_level_values(0)
hist["50DMA"] = hist["Close"].rolling(50).mean()
hist["200DMA"] = hist["Close"].rolling(200).mean()
st.line_chart(hist[["Close", "50DMA", "200DMA"]])

# ---- Aavta 7 divas (future) no anuman ----
st.subheader(f"🔮 {sym}: aavta 7 divas no anuman")
cl = hist["Close"].dropna()
lr = np.log(cl / cl.shift(1)).dropna().tail(60)
mu, sd = float(lr.mean()) * 0.5, float(lr.std())  # drift ne aadho karyo (over-optimism taalva)
days = pd.bdate_range(pd.Timestamp.today().normalize() + pd.Timedelta(days=1), periods=5)  # 7 calendar days ~ 5 trading days
rng = np.random.default_rng(42)
sims = float(cl.iloc[-1]) * np.exp(np.cumsum(rng.normal(mu, sd, (5000, len(days))), axis=1))
p16, p50, p84 = np.percentile(sims, [16, 50, 84], axis=0)
fc = pd.DataFrame({"Date": days.strftime("%d %b"), "Low (16%)": p16.round(2),
                   "Expected": p50.round(2), "High (84%)": p84.round(2)})
st.dataframe(fc, use_container_width=True, hide_index=True)
chart = pd.concat([cl.tail(30).rename("Past"),
                   pd.Series(p50, index=days, name="Expected"),
                   pd.Series(p16, index=days, name="Low"),
                   pd.Series(p84, index=days, name="High")], axis=1)
st.line_chart(chart)
st.caption(f"7 divas ma Expected: {p50[-1]:.2f} (range {p16[-1]:.2f} - {p84[-1]:.2f}). Aa 5000 simulation par aadharit anuman che, sachi price nahi.")

st.info("7D Low/High = pachhla 60 divas ni volatility parthi avtna 7 divas no anumanit range (lagbhag 68% sambhavna). Target/Stoploss fakt BUY signal mate, ATR par aadharit. Aa guarantee nathi, anuman matra che.")
st.warning("Aa tool educational che. Share market ma nuksan thai shake che. SEBI-registered advisor ni salah lo.")
