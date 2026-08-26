import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
import plotly.graph_objects as go
from datetime import datetime
from streamlit_autorefresh import st_autorefresh
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Exnova Signals Pro",
    page_icon="📊",
    layout="centered",
    initial_sidebar_state="collapsed"
)

REFRESH_INTERVAL = 10
st_autorefresh(interval=REFRESH_INTERVAL * 1000, key="auto_refresh")

st.markdown("""
<style>
    .stApp { background-color: #0b0e14; color: #e0e0e0; }
    .signal-box {
        padding: 24px 12px; border-radius: 18px; text-align: center;
        margin: 12px 0 10px 0; color: white; font-weight: 700;
        box-shadow: 0 4px 20px rgba(0,0,0,0.4);
    }
    .strength-box {
        padding: 16px 10px; border-radius: 14px; text-align: center;
        color: white; font-weight: 600;
        box-shadow: 0 2px 10px rgba(0,0,0,0.3);
    }
    .metric-card {
        background-color: #151a25; padding: 16px; border-radius: 12px;
        text-align: center; border: 1px solid #1f2636;
    }
    .disclaimer {
        font-size: 11px; color: #666; text-align: center;
        padding: 10px; border-top: 1px solid #1f2636; margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h2 style='text-align:center; margin-bottom:4px;'>📊 Exnova Signals Pro</h2>", unsafe_allow_html=True)
st.caption(f"⏱ Actualización cada {REFRESH_INTERVAL}s • {datetime.now().strftime('%H:%M:%S')}")

col1, col2 = st.columns(2)
with col1:
    activo = st.selectbox(
        "Activo",
        ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "BTC-USD", "ETH-USD"],
        index=0,
    )
with col2:
    timeframe = st.selectbox(
        "Timeframe",
        ["1m", "5m", "15m", "1h"],
        index=1,
    )

@st.cache_data(ttl=REFRESH_INTERVAL, show_spinner=False)
def obtener_datos(ticker, interval):
    try:
        period_map = {"1m": "5d", "5m": "10d", "15m": "30d", "1h": "60d"}
        df = yf.download(
            ticker,
            period=period_map.get(interval, "10d"),
            interval=interval,
            progress=False,
            auto_adjust=True
        )
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        for col in required_cols:
            if col not in df.columns:
                return pd.DataFrame()
        return df.dropna()
    except Exception:
        return pd.DataFrame()

def calcular_indicadores(df):
    df = df.copy()
    df["RSI"] = ta.momentum.rsi(df["Close"], window=14)
    df["EMA12"] = ta.trend.ema_indicator(df["Close"], window=12)
    df["EMA26"] = ta.trend.ema_indicator(df["Close"], window=26)
    df["MACD"] = ta.trend.macd(df["Close"])
    df["MACD_Signal"] = ta.trend.macd_signal(df["Close"])
    df["MACD_Hist"] = ta.trend.macd_diff(df["Close"])
    df["Stoch_K"] = ta.momentum.stoch(df["High"], df["Low"], df["Close"], window=14, smooth_window=3)
    df["Stoch_D"] = ta.momentum.stoch_signal(df["High"], df["Low"], df["Close"], window=14, smooth1=3, smooth2=3)
    df["ATR"] = ta.volatility.average_true_range(df["High"], df["Low"], df["Close"], window=14)
    df["BB_Upper"] = ta.volatility.bollinger_hband(df["Close"], window=20, window_dev=2)
    df["BB_Lower"] = ta.volatility.bollinger_lband(df["Close"], window=20, window_dev=2)
    df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / df["Close"] * 100
    return df.dropna()

def calcular_score(df):
    ultimo = df.iloc[-1]
    anterior = df.iloc[-2]
    score = 0.0
    detalles = {}

    rsi = ultimo["RSI"]
    if rsi < 20:
        score += 1.0
        detalles["RSI"] = "+1.0 (Sobreventa extrema)"
    elif rsi < 35:
        score += 0.5
        detalles["RSI"] = "+0.5 (Sobreventa)"
    elif rsi > 80:
        score -= 1.0
        detalles["RSI"] = "-1.0 (Sobrecompra extrema)"
    elif rsi > 65:
        score -= 0.5
        detalles["RSI"] = "-0.5 (Sobrecompra)"
    else:
        detalles["RSI"] = "0.0 (Neutral)"

    if ultimo["EMA12"] > ultimo["EMA26"]:
        score += 0.8
        detalles["EMA"] = "+0.8 (Tendencia alcista)"
    else:
        score -= 0.8
        detalles["EMA"] = "-0.8 (Tendencia bajista)"

    macd = ultimo["MACD"]
    macd_sig = ultimo["MACD_Signal"]
    macd_hist = ultimo["MACD_Hist"]
    macd_hist_prev = anterior["MACD_Hist"]

    if macd > macd_sig and macd_hist > macd_hist_prev:
        score += 1.0
        detalles["MACD"] = "+1.0 (Cruce alcista + momentum creciente)"
    elif macd > macd_sig:
        score += 0.5
        detalles["MACD"] = "+0.5 (Cruce alcista)"
    elif macd < macd_sig and macd_hist < macd_hist_prev:
        score -= 1.0
        detalles["MACD"] = "-1.0 (Cruce bajista + momentum decreciente)"
    elif macd < macd_sig:
        score -= 0.5
        detalles["MACD"] = "-0.5 (Cruce bajista)"
    else:
        detalles["MACD"] = "0.0 (Neutral)"

    stoch_k = ultimo["Stoch_K"]
    stoch_d = ultimo["Stoch_D"]
    if stoch_k < 20 and stoch_k > stoch_d:
        score += 1.0
        detalles["Stoch"] = "+1.0 (Sobreventa + cruce alcista)"
    elif stoch_k < 30:
        score += 0.5
        detalles["Stoch"] = "+0.5 (Zona sobreventa)"
    elif stoch_k > 80 and stoch_k < stoch_d:
        score -= 1.0
        detalles["Stoch"] = "-1.0 (Sobrecompra + cruce bajista)"
    elif stoch_k > 70:
        score -= 0.5
        detalles["Stoch"] = "-0.5 (Zona sobrecompra)"
    else:
        detalles["Stoch"] = "0.0 (Neutral)"

    close = ultimo["Close"]
    bb_upper = ultimo["BB_Upper"]
    bb_lower = ultimo["BB_Lower"]
    if close < bb_lower * 1.005:
        score += 0.7
        detalles["BB"] = "+0.7 (Cerca banda inferior)"
    elif close > bb_upper * 0.995:
        score -= 0.7
        detalles["BB"] = "-0.7 (Cerca banda superior)"
    else:
        detalles["BB"] = "0.0 (Dentro de bandas)"

    score = max(-5, min(5, score))
    return score, detalles

df_raw = obtener_datos(activo, timeframe)

if df_raw.empty or len(df_raw) < 60:
    st.error("❌ No se pudieron cargar suficientes datos. Intenta con otro activo o timeframe.")
    st.stop()

df = calcular_indicadores(df_raw)

if len(df) < 30:
    st.error("❌ Datos insuficientes después del cálculo de indicadores.")
    st.stop()

ultimo = df.iloc[-1]
anterior = df.iloc[-2]
score, detalles = calcular_score(df)

THRESHOLD = 2.0

if score >= THRESHOLD:
    senal = "CALL"
    color = "#00E676"
    emoji = "📈"
elif score <= -THRESHOLD:
    senal = "PUT"
    color = "#FF1744"
    emoji = "📉"
else:
    senal = "NEUTRAL"
    color = "#546E7A"
    emoji = "➖"

fuerza = min(100, int((abs(score) / 5.0) * 100)) if abs(score) >= THRESHOLD else int((abs(score) / THRESHOLD) * 50)

atr = ultimo["ATR"]
precio_actual = ultimo["Close"]

if "USD=X" in activo or "JPY=X" in activo or "AUD" in activo:
    sl_pips = atr * 1.5
    tp_pips = atr * 3.0
    sl_dist = f"{sl_pips:.5f}"
    tp_dist = f"{tp_pips:.5f}"
    decimales = 5
    if senal == "CALL":
        sl = precio_actual - sl_pips
        tp = precio_actual + tp_pips
    elif senal == "PUT":
        sl = precio_actual + sl_pips
        tp = precio_actual - tp_pips
    else:
        sl = tp = None
else:
    sl_pips = atr * 2.0
    tp_pips = atr * 4.0
    sl_dist = f"{sl_pips:.2f}"
    tp_dist = f"{tp_pips:.2f}"
    decimales = 2
    if senal == "CALL":
        sl = precio_actual - sl_pips
        tp = precio_actual + tp_pips
    elif senal == "PUT":
        sl = precio_actual + sl_pips
        tp = precio_actual - tp_pips
    else:
        sl = tp = None

st.markdown(f"""
<div class="signal-box" style="background-color:{color};">
    <div style="font-size:13px; opacity:0.9; letter-spacing:1px;">SEÑAL ACTUAL</div>
    <div style="font-size:44px; margin:6px 0;">{emoji} {senal}</div>
    <div style="font-size:13px; opacity:0.85;">Fuerza: {fuerza}%</div>
</div>
""", unsafe_allow_html=True)

col_left, col_right = st.columns(2)
with col_left:
    st.markdown(f"""
    <div class="strength-box" style="background-color:#00C853;">
        <div style="font-size:12px; opacity:0.9;">FUERZA ALCISTA</div>
        <div style="font-size:28px;">{max(0, int(score * 10))}%</div>
    </div>
    """, unsafe_allow_html=True)
with col_right:
    st.markdown(f"""
    <div class="strength-box" style="background-color:#D50000;">
        <div style="font-size:12px; opacity:0.9;">FUERZA BAJISTA</div>
        <div style="font-size:28px;">{max(0, int(-score * 10))}%</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

precio_str = f"{precio_actual:.{decimales}f}"
sl_str = f"{sl:.{decimales}f}" if sl else "—"
tp_str = f"{tp:.{decimales}f}" if tp else "—"

m1, m2, m3, m4 = st.columns(4)
m1.metric("💰 Precio", precio_str)
m2.metric("📊 RSI", f"{ultimo['RSI']:.1f}")
m3.metric("📉 ATR", f"{atr:.{decimales}f}")
m4.metric("⚡ Score", f"{score:.1f}")

if senal != "NEUTRAL":
    st.markdown("##### 🛡 Gestión de Riesgo (Sugerencia)")
    r1, r2, r3 = st.columns(3)
    r1.metric("🛑 Stop Loss", sl_str, delta=f"-{sl_dist}", delta_color="inverse")
    r2.metric("🎯 Take Profit", tp_str, delta=f"+{tp_dist}", delta_color="normal")
    r3.metric("⚖ Ratio R:R", "1:2")
    st.caption("Basado en ATR(14) • Ajusta según tu estrategia personal")

with st.expander("🔍 Ver desglose del Score", expanded=False):
    st.markdown(f"**Score total:** `{score:.1f}/5.0`")
    st.markdown(f"**Umbral para señal:** `±{THRESHOLD}`")
    st.markdown("---")
    for indicador, desc in detalles.items():
        color_ind = "#00E676" if "+" in desc else "#FF1744" if "-" in desc else "#888"
        st.markdown(f"<span style='color:{color_ind}'>●</span> **{indicador}:** {desc}", unsafe_allow_html=True)

st.markdown("##### 📊 Gráfico de Precio")

df_plot = df.tail(60).copy()

fig = go.Figure()

fig.add_trace(go.Candlestick(
    x=df_plot.index,
    open=df_plot["Open"],
    high=df_plot["High"],
    low=df_plot["Low"],
    close=df_plot["Close"],
    increasing_line_color="#00E676",
    decreasing_line_color="#FF1744",
    increasing_fillcolor="rgba(0, 230, 118, 0.3)",
    decreasing_fillcolor="rgba(255, 23, 68, 0.3)",
    line=dict(width=1),
    name="Precio"
))

fig.add_trace(go.Scatter(
    x=df_plot.index, y=df_plot["EMA12"],
    mode="lines", line=dict(color="#FFB300", width=1.5), name="EMA 12"
))
fig.add_trace(go.Scatter(
    x=df_plot.index, y=df_plot["EMA26"],
    mode="lines", line=dict(color="#42A5F5", width=1.5), name="EMA 26"
))

fig.add_trace(go.Scatter(
    x=df_plot.index, y=df_plot["BB_Upper"],
    mode="lines", line=dict(color="rgba(255,255,255,0.15)", width=1, dash="dot"),
    showlegend=False
))
fig.add_trace(go.Scatter(
    x=df_plot.index, y=df_plot["BB_Lower"],
    mode="lines", line=dict(color="rgba(255,255,255,0.15)", width=1, dash="dot"),
    fill="tonexty", fillcolor="rgba(255,255,255,0.03)", showlegend=False
))

if senal != "NEUTRAL" and sl is not None and tp is not None:
    fig.add_hline(y=sl, line_dash="dash", line_color="#FF1744",
                  annotation_text="SL", annotation_position="right", annotation_font_color="#FF1744")
    fig.add_hline(y=tp, line_dash="dash", line_color="#00E676",
                  annotation_text="TP", annotation_position="right", annotation_font_color="#00E676")

fig.update_layout(
    height=350,
    margin=dict(l=5, r=5, t=15, b=10),
    xaxis_rangeslider_visible=False,
    template="plotly_dark",
    showlegend=True,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10)),
    paper_bgcolor="#0b0e14",
    plot_bgcolor="#0b0e14",
    xaxis=dict(showgrid=False, fixedrange=True, showticklabels=True, color="#888", tickfont=dict(size=10)),
    yaxis=dict(showgrid=True, gridcolor="#1f2636", fixedrange=True, color="#888", side="right", tickfont=dict(size=10)),
    font=dict(color="white", size=11)
)

st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key="main_chart")

st.markdown("##### 📋 Indicadores en tiempo real")

ind_cols = st.columns(5)
indicadores_data = [
    ("RSI(14)", f"{ultimo['RSI']:.1f}", "Sobreventa <30 | Sobrecompra >70"),
    ("MACD", f"{ultimo['MACD']:.{decimales}f}", "Cruce con señal"),
    ("Stoch %K", f"{ultimo['Stoch_K']:.1f}", "Sobreventa <20 | Sobrecompra >80"),
    ("ATR(14)", f"{ultimo['ATR']:.{decimales}f}", "Volatilidad actual"),
    ("BB Width", f"{ultimo['BB_Width']:.2f}%", "Ancho de bandas"),
]

for col, (nombre, valor, desc) in zip(ind_cols, indicadores_data):
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size:11px; color:#888;">{nombre}</div>
            <div style="font-size:18px; font-weight:600; margin:4px 0;">{valor}</div>
            <div style="font-size:9px; color:#555;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("""
<div class="disclaimer">
    ⚠️ <strong>Herramienta educativa únicamente</strong> • No constituye consejo de inversión • 
    El trading con CFDs, Forex y Criptomonedas conlleva un <strong>alto riesgo de pérdida</strong> • 
    Las señales son generadas por indicadores técnicos automáticos sin garantía de rentabilidad • 
    Realiza siempre tu propio análisis antes de operar.
</div>
""", unsafe_allow_html=True)
