from datetime import datetime
import pandas as pd
import plotly.graph_objects as go
import ta
import warnings
import yfinance as yf
import streamlit as st

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Exnova Signals",
    page_icon="📱",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Estilos CSS optimizados
st.markdown(
    """
<style>
    .stApp {
        background-color: #0b0e14;
        color: white;
    }
    .signal-box {
        padding: 24px 12px;
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
""",
    unsafe_allow_html=True,
)

st.markdown(
    "<h2 style='text-align:center; margin-bottom:4px;'>Exnova Signals</h2>",
    unsafe_allow_html=True,
)

# Selectores globales (fuera del fragmento para que el usuario pueda cambiar de activo tranquilamente)
col1, col2 = st.columns(2)
with col1:
    activo = st.selectbox(
        "Activo",
        ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "BTC-USD", "ETH-USD"],
        index=0,
    )
with col2:
    timeframe = st.selectbox("Timeframe", ["1m", "5m", "15m"], index=1)


# Caché optimizada para los datos
@st.cache_data(ttl=4, show_spinner=False)
def obtener_datos(ticker, interval):
  try:
    period_map = {"1m": "1d", "5m": "5d", "15m": "10d"}
    df = yf.download(
        ticker, period=period_map[interval], interval=interval, progress=False
    )
    if isinstance(df.columns, pd.MultiIndex):
      df.columns = df.columns.get_level_values(0)
    return df.dropna()
  except Exception:
    return pd.DataFrame()


# FRAGMENTO AUTOMÁTICO: Solo se recarga esta parte cada 5 segundos sin afectar el scroll ni bloquear la página
@st.fragment(run_every=5)
def renderizar_panel_senales(ticker, tf):
  st.caption(
      f"Actualización automática cada 5s • {datetime.now().strftime('%H:%M:%S')}"
  )

  df = obtener_datos(ticker, tf)

  if df.empty or len(df) < 30:
    st.error("No se pudieron cargar los datos. Prueba otro activo o timeframe.")
    return

  # Indicadores técnicos
  df["RSI"] = ta.momentum.rsi(df["Close"], window=7)
  df["EMA9"] = ta.trend.ema_indicator(df["Close"], window=9)
  df["EMA21"] = ta.trend.ema_indicator(df["Close"], window=21)
  df["MACD"] = ta.trend.macd_diff(
      df["Close"], window_slow=16, window_fast=8
  )
  df["Stoch"] = ta.momentum.stoch(df["High"], df["Low"], df["Close"], window=8)
  df = df.dropna()

  if len(df) < 2:
    st.warning("Datos insuficientes tras calcular indicadores.")
    return

  ultimo = df.iloc[-1]
  anterior = df.iloc[-2]

  # Cálculo de Score
  score = 0.0

  if ultimo["RSI"] < 25:
    score += 2.8
  elif ultimo["RSI"] < 38:
    score += 1.5
  elif ultimo["RSI"] > 75:
    score -= 2.8
  elif ultimo["RSI"] > 62:
    score -= 1.5

  if ultimo["EMA9"] > ultimo["EMA21"]:
    score += 1.9
  else:
    score -= 1.9

  if ultimo["MACD"] > 0 and anterior["MACD"] <= 0:
    score += 1.8
  elif ultimo["MACD"] < 0 and anterior["MACD"] >= 0:
    score -= 1.8
  elif ultimo["MACD"] > 0:
    score += 0.7
  else:
    score -= 0.7

  if ultimo["Stoch"] < 18:
    score += 1.4
  elif ultimo["Stoch"] > 82:
    score -= 1.4

  prob_call = max(12, min(88, 50 + score * 7.2))
  prob_put = 100 - prob_call

  if score >= 2.1:
    senal = "CALL"
    color = "#00E676"
  elif score <= -2.1:
    senal = "PUT"
    color = "#FF1744"
  else:
    senal = "NEUTRAL"
    color = "#546E7A"

  # Mostrar Señal
  st.markdown(
      f"""
    <div class="signal-box" style="background-color:{color};">
        <div style="font-size:14px; opacity:0.9;">SEÑAL ACTUAL</div>
        <div style="font-size:48px; margin:8px 0;">{senal}</div>
    </div>
    """,
      unsafe_allow_html=True,
  )

  # Probabilidades
  p1, p2 = st.columns(2)
  with p1:
    st.markdown(
        f"""
        <div class="prob-box" style="background-color:#00C853;">
            <div>CALL</div>
            <div style="font-size:30px;">{prob_call:.0f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
  with p2:
    st.markdown(
        f"""
        <div class="prob-box" style="background-color:#D50000;">
            <div>PUT</div>
            <div style="font-size:30px;">{prob_put:.0f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

  st.write("")

  # Métricas
  m1, m2, m3 = st.columns(3)
  precio = (
      f"{ultimo['Close']:.5f}" if "USD=X" in ticker else f"{ultimo['Close']:.2f}"
  )
  m1.metric("Precio", precio)
  m2.metric("RSI", f"{ultimo['RSI']:.1f}")
  m3.metric("Score", f"{score:.1f}")

  # Gráfico optimizado
  st.markdown("##### Gráfico")

  df_plot = df.tail(40).copy()

  fig = go.Figure()

  fig.add_trace(
      go.Candlestick(
          x=df_plot.index,
          open=df_plot["Open"],
          high=df_plot["High"],
          low=df_plot["Low"],
          close=df_plot["Close"],
          increasing_line_color="#00E676",
          decreasing_line_color="#FF1744",
          line=dict(width=1),
          name="Precio",
      )
  )

  fig.add_trace(
      go.Scatter(
          x=df_plot.index,
          y=df_plot["EMA9"],
          mode="lines",
          line=dict(color="#FFB300", width=1.8),
          name="EMA 9",
      )
  )

  fig.add_trace(
      go.Scatter(
          x=df_plot.index,
          y=df_plot["EMA21"],
          mode="lines",
          line=dict(color="#42A5F5", width=1.8),
          name="EMA 21",
      )
  )

  fig.update_layout(
      height=300,
      margin=dict(l=5, r=5, t=15, b=10),
      xaxis_rangeslider_visible=False,
      template="plotly_dark",
      showlegend=False,
      paper_bgcolor="#0b0e14",
      plot_bgcolor="#0b0e14",
      xaxis=dict(
          showgrid=False, fixedrange=True, showticklabels=True, color="#888"
      ),
      yaxis=dict(
          showgrid=True,
          gridcolor="#1f1f1f",
          fixedrange=True,
          color="#888",
          side="right",
      ),
      font=dict(color="white", size=11),
  )

  st.plotly_chart(
      fig, use_container_width=True, config={"displayModeBar": False}
  )

  st.caption("Herramienta educativa • No es consejo financiero")


# Llamar al fragmento pasando el activo y el timeframe seleccionados
renderizar_panel_senales(activo, timeframe)
