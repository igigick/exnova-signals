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

# Actualización cada 5 segundos
st_autorefresh(interval=5000, key="refresh")

# Estilos
st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    .signal-box {
        padding: 22px 10px;
        border-radius: 16px;
        text-align: center;
        margin: 12px 0 8px 0;
        color: white;
        font-weight: bold;
    }
    .prob-box {
        padding: 14px 8px;
        border-radius: 12px;
        text-align: center;
        color: white;
        font-weight: bold;
    }
    .metric-container { background-color: #1c1c1c; border-radius: 10px; padding: 8px; }
</style>
""", unsafe_allow_html=True)

st.markdown("<h2 style='text-align:center; color:white; margin-bottom:5px;'>Exnova Signals</h2>", unsafe_allow_html=True)
st.caption(f"Actualización automática • {datetime.now().strftime('%H:%M:%S')}")

# Selectores
c1, c2 = st.columns(2)
with c1:
    activo = st.selectbox("Activo", ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "BTC-USD", "ETH-USD", "AAPL"], index=0, label_visibility="collapsed")
with c2:
    timeframe = st.selectbox("Timeframe", ["1m", "5m", "15m"], index=1, label_visibility="collapsed")

# Datos
@st.cache_data(ttl=4)
def get_data(ticker, interval):
    period = {"1m": "1d", "5m": "5d", "15m": "10d"}[interval]
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna()

df = get_data(activo, timeframe)

if len(df) < 20:
    st.error("Pocos datos. Cambia de timeframe.")
    st.stop()

# Indicadores
df['RSI'] = ta.momentum.rsi(df['Close'], 7)
df['EMA9'] = ta.trend.ema_indicator(df['Close'], 9)
df['EMA21'] = ta.trend.ema_indicator(df['Close'], 21)
df['MACD'] = ta.trend.macd_diff(df['Close'], 16, 8)
df = df.dropna()

ultimo = df.iloc[-1]
prev = df.iloc[-2]

# Score
score = 0
if ultimo['RSI'] < 30: score += 2
elif ultimo['RSI'] > 70: score -= 2
elif ultimo['RSI'] < 45: score += 1
elif ultimo['RSI'] > 55: score -= 1

if ultimo['EMA9'] > ultimo['EMA21']: score += 1.5
else: score -= 1.5

if ultimo['MACD'] > 0 and prev['MACD'] <= 0: score += 1.5
elif ultimo['MACD'] < 0 and prev['MACD'] >= 0: score -= 1.5

# Probabilidades
prob_call = max(8, min(92, 50 + score * 9))
prob_put = 100 - prob_call

if score >= 2.2:
    senal, color = "CALL", "#00C853"
elif score <= -2.2:
    senal, color = "PUT", "#FF1744"
else:
    senal, color = "NEUTRAL", "#616161"

# Señal grande
st.markdown(f"""
<div class="signal-box" style="background-color:{color};">
    <div style="font-size:15px; opacity:0.9;">SEÑAL</div>
    <div style="font-size:42px; margin:4px 0;">{senal}</div>
</div>
""", unsafe_allow_html=True)

# Probabilidades
p1, p2 = st.columns(2)
with p1:
    st.markdown(f"""
    <div class="prob-box" style="background-color:#00C853;">
        CALL<br><span style="font-size:26px;">{prob_call:.0f}%</span>
    </div>
    """, unsafe_allow_html=True)
with p2:
    st.markdown(f"""
    <div class="prob-box" style="background-color:#FF1744;">
        PUT<br><span style="font-size:26px;">{prob_put:.0f}%</span>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# Precio y RSI
m1, m2, m3 = st.columns(3)
precio = f"{ultimo['Close']:.5f}" if "USD=X" in activo else f"{ultimo['Close']:.2f}"
m1.metric("Precio", precio)
m2.metric("RSI", f"{ultimo['RSI']:.1f}")
m3.metric("Score", f"{score:.1f}")

# Gráfico más ligero
st.markdown("### Gráfico")
fig = go.Figure()

fig.add_trace(go.Candlestick(
    x=df.index[-80:],
    open=df['Open'][-80:],
    high=df['High'][-80:],
    low=df['Low'][-80:],
    close=df['Close'][-80:],
    name="Precio",
    increasing_line_color='#00C853',
    decreasing_line_color='#FF1744'
))

fig.add_trace(go.Scatter(x=df.index[-80:], y=df['EMA9'][-80:], name="EMA9", line=dict(color='orange', width=1.2)))
fig.add_trace(go.Scatter(x=df.index[-80:], y=df['EMA21'][-80:], name="EMA21", line=dict(color='#2196F3', width=1.2)))

fig.update_layout(
    height=320,
    margin=dict(l=5, r=5, t=10, b=10),
    xaxis_rangeslider_visible=False,
    template="plotly_dark",
    showlegend=False,
    paper_bgcolor="#0e1117",
    plot_bgcolor="#0e1117"
)

st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

st.caption("Solo educativo • No es consejo financiero • Alto riesgo")
