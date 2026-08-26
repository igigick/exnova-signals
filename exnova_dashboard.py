import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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

# Actualización automática cada 5 segundos
st_autorefresh(interval=5000, key="refresh")

st.markdown("""
<style>
    .signal-box {
        padding: 18px;
        border-radius: 14px;
        text-align: center;
        margin: 10px 0;
        color: white;
        font-weight: bold;
    }
    .prob-box {
        padding: 12px;
        border-radius: 10px;
        text-align: center;
        margin: 6px 0;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

st.title("📱 Exnova Signals")
st.caption("Actualización cada 5 segundos • Solo educativo")

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

@st.cache_data(ttl=4)
def obtener_datos(ticker, interval, period):
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna()

df = obtener_datos(activo, interval_map[timeframe], periodo_map[timeframe])

if len(df) < 30:
    st.error("No hay suficientes datos. Prueba otro timeframe.")
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

def calcular_score(row, prev):
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

score = calcular_score(ultimo, anterior)

prob_call = max(5, min(95, 50 + score * 8))
prob_put = 100 - prob_call

if score >= 2.5:
    senal = "CALL"
    color = "#00C853"
elif score <= -2.5:
    senal = "PUT"
    color = "#D50000"
else:
    senal = "NEUTRAL"
    color = "#757575"

st.markdown(f"""
<div class="signal-box" style="background-color:{color};">
    <div style="font-size:16px;">SEÑAL ACTUAL</div>
    <div style="font-size:38px; margin:6px 0;">{senal}</div>
</div>
""", unsafe_allow_html=True)

c1, c2 = st.columns(2)
with c1:
    st.markdown(f"""
    <div class="prob-box" style="background-color:#00C853;">
        <div>ALCISTA (CALL)</div>
        <div style="font-size:28px;">{prob_call:.0f}%</div>
    </div>
    """, unsafe_allow_html=True)
with c2:
    st.markdown(f"""
    <div class="prob-box" style="background-color:#D50000;">
        <div>BAJISTA (PUT)</div>
        <div style="font-size:28px;">{prob_put:.0f}%</div>
    </div>
    """, unsafe_allow_html=True)

m1, m2, m3 = st.columns(3)
precio = f"{ultimo['Close']:.5f}" if "USD=X" in activo else f"{ultimo['Close']:.2f}"
m1.metric("Precio", precio)
m2.metric("RSI", f"{ultimo['RSI']:.1f}")
m3.metric("Score", f"{score:.1f}")

st.caption(f"Última actualización: {datetime.now().strftime('%H:%M:%S')}")

st.subheader("Gráfico")
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.07, row_heights=[0.7, 0.3])

fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'],
                             low=df['Low'], close=df['Close'], name="Precio"), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['EMA_9'], name="EMA 9", line=dict(color='orange', width=1.5)), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['EMA_21'], name="EMA 21", line=dict(color='blue', width=1.5)), row=1, col=1)

fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name="RSI", line=dict(color='purple')), row=2, col=1)
fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

fig.update_layout(height=450, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=20, b=10), showlegend=False)
st.plotly_chart(fig, use_container_width=True)

with st.expander("Detalles"):
    st.write(f"**Activo:** {activo}")
    st.write(f"**Timeframe:** {timeframe}")
    st.write(f"**Score:** {score:.2f}")
    st.write(f"**Tendencia EMA:** {'Alcista' if ultimo['EMA_9'] > ultimo['EMA_21'] else 'Bajista'}")

st.warning("Herramienta educativa. No garantiza resultados. Alto riesgo de pérdida.")
