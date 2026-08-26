import streamlit as st
import yfinance as yf
import pandas as pd
import ta
import plotly.graph_objects as go
from datetime import datetime
from streamlit_autorefresh import st_autorefresh
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Exnova Signals", page_icon="📱", layout="centered", initial_sidebar_state="collapsed")

# Actualización cada 12 segundos (más estable)
st_autorefresh(interval=12000, key="refresh")

st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    .signal-box {
        padding: 24px 10px;
        border-radius: 16px;
        text-align: center;
        margin: 10px 0;
        color: white;
        font-weight: bold;
    }
    .prob-box {
        padding: 16px 8px;
        border-radius: 12px;
        text-align: center;
        color: white;
        font-weight: bold;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h2 style='text-align:center; color:white;'>Exnova Signals</h2>", unsafe_allow_html=True)
st.caption(f"Actualizado: {datetime.now().strftime('%H:%M:%S')}")

# Selectores
c1, c2 = st.columns(2)
with c1:
    activo = st.selectbox("Activo", ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "BTC-USD", "ETH-USD"], index=0)
with c2:
    timeframe = st.selectbox("Timeframe", ["1m", "5m", "15m"], index=1)

@st.cache_data(ttl=10)
def get_data(ticker, interval):
    period = {"1m": "1d", "5m": "5d", "15m": "10d"}[interval]
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna()

df = get_data(activo, timeframe)

if len(df) < 25:
    st.warning("Pocos datos disponibles. Cambia el timeframe.")
    st.stop()

# Indicadores
df['RSI'] = ta.momentum.rsi(df['Close'], 7)
df['EMA9'] = ta.trend.ema_indicator(df['Close'], 9)
df['EMA21'] = ta.trend.ema_indicator(df['Close'], 21)
df['MACD'] = ta.trend.macd_diff(df['Close'], 12, 6)
df['Stoch'] = ta.momentum.stoch(df['High'], df['Low'], df['Close'], 8)
df = df.dropna()

ultimo = df.iloc[-1]
prev = df.iloc[-2]

# Score más sensible
score = 0

# RSI
if ultimo['RSI'] < 28: score += 2.5
elif ultimo['RSI'] < 40: score += 1.3
elif ultimo['RSI'] > 72: score -= 2.5
elif ultimo['RSI'] > 60: score -= 1.3

# EMAs
if ultimo['EMA9'] > ultimo['EMA21']: score += 1.8
else: score -= 1.8

# MACD
if ultimo['MACD'] > 0 and prev['MACD'] <= 0: score += 1.7
elif ultimo['MACD'] < 0 and prev['MACD'] >= 0: score -= 1.7
elif ultimo['MACD'] > 0: score += 0.6
else: score -= 0.6

# Stochastic
if ultimo['Stoch'] < 20: score += 1.2
elif ultimo['Stoch'] > 80: score -= 1.2

# Probabilidades más dinámicas
prob_call = max(10, min(90, 50 + score * 7.5))
prob_put = 100 - prob_call

if score >= 2.0:
    senal, color = "CALL", "#00C853"
elif score <= -2.0:
    senal, color = "PUT", "#FF1744"
else:
    senal, color = "NEUTRAL", "#616161"

# Señal grande
st.markdown(f"""
<div class="signal-box" style="background-color:{color};">
    <div style="font-size:15px; opacity:0.85;">SEÑAL</div>
    <div style="font-size:44px; margin:6px 0;">{senal}</div>
</div>
""", unsafe_allow_html=True)

# Probabilidades
p1, p2 = st.columns(2)
with p1:
    st.markdown(f"""
    <div class="prob-box" style="background-color:#00C853;">
        CALL<br><span style="font-size:28px;">{prob_call:.0f}%</span>
    </div>
    """, unsafe_allow_html=True)
with p2:
    st.markdown(f"""
    <div class="prob-box" style="background-color:#FF1744;">
        PUT<br><span style="font-size:28px;">{prob_put:.0f}%</span>
    </div>
    """, unsafe_allow_html=True)

# Métricas
m1, m2, m3 = st.columns(3)
precio = f"{ultimo['Close']:.5f}" if "USD=X" in activo else f"{ultimo['Close']:.2f}"
m1.metric("Precio", precio)
m2.metric("RSI", f"{ultimo['RSI']:.1f}")
m3.metric("Score", f"{score:.1f}")

# Botón de refresco manual
if st.button("🔄 Actualizar ahora", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

# Gráfico ligero
st.markdown("### Gráfico")
fig = go.Figure()

fig.add_trace(go.Candlestick(
    x=df.index[-60:],
    open=df['Open'][-60:],
    high=df['High'][-60:],
    low=df['Low'][-60:],
    close=df['Close'][-60:],
    increasing_line_color='#
