import streamlit as st
import yfinance as yf
import pandas as pd
import ta
import plotly.graph_objects as go
from datetime import datetime
from streamlit_autorefresh import st_autorefresh
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Exnova Signals",
    page_icon="📱",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Actualización automática cada 15 segundos
st_autorefresh(interval=15 * 1000, key="auto_refresh")

st.markdown("""
<style>
    .stApp { background-color: #0b0e14; color: white; }
    .signal-box {
        padding: 26px 12px;
        border-radius: 18px;
        text-align: center;
        margin: 12px 0 10px 0;
        color: white;
        font-weight: 700;
    }
    .prob-box {
        padding: 16px 10px;
        border-radius: 14px;
        text-align: center;
        color: white;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h2 style='text-align:center; margin-bottom:5px;'>Exnova Signals</h2>", unsafe_allow_html=True)
st.caption(f"Actualización automática • {datetime.now().strftime('%H:%M:%S')}")

col1, col2 = st.columns(2)
with col1:
    activo = st.selectbox("Activo", ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "BTC-USD", "ETH-USD"], index=0)
with col2:
    timeframe = st.selectbox("Timeframe", ["1m", "5m", "15m"], index=1)

@st.cache_data(ttl=12, show_spinner=False)
def obtener_datos(ticker, interval):
    try:
        period = {"1m": "1d", "5m": "5d", "15m": "10d"}[interval]
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df.dropna()
    except:
        return pd.DataFrame()

df = obtener_datos(activo, timeframe)

if df.empty or len(df) < 30:
    st.error("No se pudieron cargar datos. Prueba otro activo o timeframe.")
    st.stop()

df['RSI'] = ta.momentum.rsi(df['Close'], window=7)
df['EMA9'] = ta.trend.ema_indicator(df['Close'], window=9)
df['EMA21'] = ta.trend.ema_indicator(df['Close'], window=21)
df['MACD'] = ta.trend.macd_diff(df['Close'], window_slow=16, window_fast=8)
df['Stoch'] = ta.momentum.stoch(df['High'], df['Low'], df['Close'], window=8)
df = df.dropna()

ultimo = df.iloc[-1]
anterior = df.iloc[-2]

score = 0.0
if ultimo['RSI'] < 25: score += 2.8
elif ultimo['RSI'] < 38: score += 1.5
elif ultimo['RSI'] > 75: score -= 2.8
elif ultimo['RSI'] > 62: score -= 1.5

if ultimo['EMA9'] > ultimo['EMA21']: score += 1.9
else: score -= 1.9

if ultimo['MACD'] > 0 and anterior['MACD'] <= 0: score += 1.8
elif ultimo['MACD'] < 0 and anterior['MACD'] >= 0: score -= 1.8
elif ultimo['MACD'] > 0: score += 0.7
else: score -= 0.7

if ultimo['Stoch'] < 18: score += 1.4
elif ultimo['Stoch'] > 82: score -= 1.4

prob_call = max(12, min(88, 50 + score * 7.2))
prob_put = 100 - prob_call

if score >= 2.1:
    senal, color = "CALL", "#00E676"
elif score <= -2.1:
    senal, color = "PUT", "#FF1744"
else:
    senal, color = "NEUTRAL", "#546E7A"

st.markdown(f"""
<div class="signal-box" style="background-color:{color};">
    <div style="font-size:14px; opacity:0.9;">SEÑAL ACTUAL</div>
    <div style="font-size:48px; margin:8px 0;">{senal}</div>
</div>
""", unsafe_allow_html=True)

p1, p2 = st.columns(2)
with p1:
    st.markdown(f"""
    <div class="prob-box" style="background-color:#00C853;">
        <div>CALL</div>
        <div style="font-size:30px;">{prob_call:.0f}%</div>
    </div>
    """, unsafe_allow_html=True)
with p2:
    st.markdown(f"""
    <div class="prob-box" style="background-color:#D50000;">
        <div>PUT</div>
        <div style="font-size:30px;">{prob_put:.0f}%</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

m1, m2, m3 = st.columns(3)
precio_txt = f"{ultimo['Close']:.5f}" if "USD=X" in activo else f"{ultimo['Close']:.2f}"
m1.metric("Precio", precio_txt)
m2.metric("RSI", f"{ultimo['RSI']:.1f}")
m3.metric("Score", f"{score:.1f}")

if st.button("🔄 Actualizar ahora", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

# Gráfico ligero
st.markdown("##### Gráfico")
df_plot = df.tail(40)

fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=df_plot.index,
    open=df_plot['Open'],
    high=df_plot['High'],
    low=df_plot['Low'],
    close=df_plot['Close'],
    increasing_line_color='#00E676',
    decreasing_line_color='#FF1744'
))
fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['EMA9'], line=dict(color='#FFB300', width=1.5)))
fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['EMA21'], line=dict(color='#42A5F5', width=1.5)))

fig.update_layout(
    height=280,
    margin=dict(l=0, r=0, t=10, b=10),
    xaxis_rangeslider_visible=False,
    template="plotly_dark",
    showlegend=False,
    paper_bgcolor="#0b0e14",
    plot_bgcolor="#0b0e14"
)

st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

st.caption("Herramienta educativa • No es consejo financiero • Alto riesgo de pérdida")