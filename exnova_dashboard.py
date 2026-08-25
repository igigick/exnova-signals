import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Exnova Signals",
    page_icon="📱",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .stButton>button {
        width: 100%;
        height: 3.2em;
        font-size: 17px;
        font-weight: bold;
    }
    .signal-box {
        padding: 18px;
        border-radius: 14px;
        text-align: center;
        margin: 12px 0;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

st.title("📱 Exnova Signals")
st.caption("Herramienta educativa de señales Call/Put • No es consejo financiero")

col1, col2 = st.columns(2)
with col1:
    activo = st.selectbox(
        "Activo",
        ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "BTC-USD", "ETH-USD", "AAPL", "TSLA"],
        index=0
    )
with col2:
    timeframe = st.selectbox("Timeframe", ["1m", "5m", "15m"], index=1)

periodo_map = {"1m": "5d", "5m": "30d", "15m": "60d"}
interval_map = {"1m": "1m", "5m": "5m", "15m": "15m"}

@st.cache_data(ttl=60)
def obtener_datos(ticker, interval, period):
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna()

df = obtener_datos(activo, interval_map[timeframe], periodo_map[timeframe])

if len(df) < 30:
    st.error("No hay suficientes datos. Prueba otro timeframe o activo.")
    st.stop()

df['RSI'] = ta.momentum.rsi(df['Close'], 7)
df['EMA_9'] = ta.trend.ema_indicator(df['Close'], 9)
df['EMA_21'] = ta.trend.ema_indicator(df['Close'], 21)
df['MACD'] = ta.trend.macd_diff(df['Close'], window_slow=16, window_fast=8)
df['Stoch'] = ta.momentum.stoch(df['High'], df['Low'], df['Close'], 8)
df['CCI'] = ta.trend.cci(df['High'], df['Low'], df['Close'], 10)

df = df.dropna()
ultimo = df.iloc[-1]
anterior = df.iloc[-2]

def generar_senal(row, prev):
    score = 0
    if row['RSI'] < 30: score += 2
    elif row['RSI'] > 70: score -= 2
    elif row['RSI'] < 45: score += 1
    elif row['RSI'] > 55: score -= 1

    if row['EMA_9'] > row['EMA_21']: score += 1.5
    else: score -= 1.5

    if row['MACD'] > 0 and prev['MACD'] <= 0: score += 1.5
    elif row['MACD'] < 0 and prev['MACD'] >= 0: score -= 1.5

    if row['Stoch'] < 25: score += 1
    elif row['Stoch'] > 75: score -= 1

    if row['CCI'] < -100: score += 1
    elif row['CCI'] > 100: score -= 1

    return score

score = generar_senal(ultimo, anterior)

if score >= 3:
    senal = "CALL"
    color = "#00C853"
    confianza = min(94, 58 + score * 5)
elif score <= -3:
    senal = "PUT"
    color = "#D50000"
    confianza = min(94, 58 + abs(score) * 5)
else:
    senal = "NEUTRAL"
    color = "#757575"
    confianza = 50

st.markdown(f"""
<div class="signal-box" style="background-color:{color};">
    <div style="font-size:17px;">SEÑAL ACTUAL</div>
    <div style="font-size:40px; font-weight:bold; margin:8px 0;">{senal}</div>
    <div style="font-size:19px;">Confianza: {confianza:.0f}%</div>
</div>
""", unsafe_allow_html=True)

m1, m2, m3 = st.columns(3)
precio = f"{ultimo['Close']:.5f}" if "USD=X" in activo else f"{ultimo['Close']:.2f}"
m1.metric("Precio", precio)
m2.metric("RSI (7)", f"{ultimo['RSI']:.1f}")
m3.metric("Score", f"{score:.1f}")

st.subheader("Gráfico")
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.07, row_heights=[0.7, 0.3])

fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Precio"), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['EMA_9'], name="EMA 9", line=dict(color='orange', width=1.5)), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['EMA_21'], name="EMA 21", line=dict(color='blue', width=1.5)), row=1, col=1)

fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name="RSI", line=dict(color='purple')), row=2, col=1)
fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

fig.update_layout(height=460, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=30, b=10), showlegend=False)
st.plotly_chart(fig, use_container_width=True)

with st.expander("Detalles de la señal"):
    st.write(f"**Activo:** {activo}")
    st.write(f"**Timeframe:** {timeframe}")
    st.write(f"**Score:** {score:.2f}")
    st.write(f"**RSI:** {ultimo['RSI']:.1f}")
    st.write(f"**Tendencia EMA:** {'Alcista' if ultimo['EMA_9'] > ultimo['EMA_21'] else 'Bajista'}")
    st.write(f"**Actualizado:** {datetime.now().strftime('%H:%M:%S')}")

st.markdown("---")
st.warning("Esta herramienta es solo educativa. Las señales no garantizan resultados. El trading de opciones binarias implica alto riesgo de pérdida del capital.")
