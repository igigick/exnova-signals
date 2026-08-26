import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
from streamlit_autorefresh import st_autorefresh
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Exnova Signals Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

REFRESH_INTERVAL = 10
st_autorefresh(interval=REFRESH_INTERVAL * 1000, key="auto_refresh")

st.markdown("""
<style>
    .stApp { background-color: #0b0e14; color: #e0e0e0; }
    .block-container { padding: 0.3rem 1rem 0rem 1rem; max-width: 100%; }
    .signal-box {
        padding: 6px 4px; border-radius: 10px; text-align: center;
        margin: 2px 0; color: white; font-weight: 700;
        box-shadow: 0 2px 8px rgba(0,0,0,0.4);
    }
    .strength-box {
        padding: 4px 2px; border-radius: 8px; text-align: center;
        color: white; font-weight: 600; font-size: 11px;
    }
    .metric-card {
        background-color: #151a25; padding: 4px 2px; border-radius: 6px;
        text-align: center; border: 1px solid #1f2636; font-size: 10px;
    }
    .disclaimer {
        font-size: 9px; color: #555; text-align: center;
        padding: 2px; margin-top: 4px;
    }
    h1, h2, h3, h4, h5, h6 { margin: 0 !important; padding: 0 !important; }
    p { margin: 0 !important; padding: 0 !important; }
    .stSelectbox { margin-bottom: -10px !important; }
    .stSelectbox label { font-size: 11px !important; margin-bottom: 0 !important; }
    .stMetric { margin: 0 !important; padding: 0 !important; }
    .stMetric label { font-size: 10px !important; }
    .stMetric div { font-size: 14px !important; }
    .stExpander { margin: 2px 0 !important; }
    .stExpander button { font-size: 11px !important; padding: 2px !important; }
    iframe { height: 220px !important; }
</style>
""", unsafe_allow_html=True)

# HEADER compacto
c1, c2, c3 = st.columns([2, 3, 2])
with c1:
    st.markdown("<h4 style='margin:0;'>📊 Exnova Signals</h4>", unsafe_allow_html=True)
with c2:
    st.caption(f"⏱ {REFRESH_INTERVAL}s • {datetime.now().strftime('%H:%M:%S')}")
with c3:
    activo = st.selectbox("", ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "BTC", "ETH"], index=0, label_visibility="collapsed")
    activo_map = {"EURUSD":"EURUSD=X","GBPUSD":"GBPUSD=X","USDJPY":"USDJPY=X","AUDUSD":"AUDUSD=X","BTC":"BTC-USD","ETH":"ETH-USD"}
    activo = activo_map[activo]

# Timeframe en la misma línea del header si cabe, o debajo muy compacto
tf = st.selectbox("TF", ["1m", "5m", "15m", "1h"], index=1, label_visibility="collapsed", key="tf")

@st.cache_data(ttl=REFRESH_INTERVAL, show_spinner=False)
def obtener_datos(ticker, interval):
    try:
        period_map = {"1m": "5d", "5m": "10d", "15m": "30d", "1h": "60d"}
        df = yf.download(ticker, period=period_map.get(interval, "10d"), interval=interval, progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        for col in required_cols:
            if col not in df.columns:
                return pd.DataFrame()
        return df.dropna()
    except Exception:
        return pd.DataFrame()

def calcular_rsi(series, window=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calcular_ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def calcular_macd(series, fast=12, slow=26, signal=9):
    ema_fast = calcular_ema(series, fast)
    ema_slow = calcular_ema(series, slow)
    macd_line = ema_fast - ema_slow
    macd_signal = calcular_ema(macd_line, signal)
    return macd_line, macd_signal, macd_line - macd_signal

def calcular_atr(df, window=14):
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=window, min_periods=1).mean()

def calcular_bollinger(series, window=20, num_std=2):
    sma = series.rolling(window=window, min_periods=1).mean()
    std = series.rolling(window=window, min_periods=1).std()
    return sma + (std * num_std), sma - (std * num_std)

def calcular_stoch(df, k_window=14, d_window=3):
    lowest_low = df["Low"].rolling(window=k_window, min_periods=1).min()
    highest_high = df["High"].rolling(window=k_window, min_periods=1).max()
    stoch_k = 100 * (df["Close"] - lowest_low) / (highest_high - lowest_low)
    return stoch_k, stoch_k.rolling(window=d_window, min_periods=1).mean()

def calcular_indicadores(df):
    df = df.copy()
    df["RSI"] = calcular_rsi(df["Close"], 14)
    df["EMA12"] = calcular_ema(df["Close"], 12)
    df["EMA26"] = calcular_ema(df["Close"], 26)
    df["MACD"], df["MACD_Signal"], df["MACD_Hist"] = calcular_macd(df["Close"])
    df["Stoch_K"], df["Stoch_D"] = calcular_stoch(df)
    df["ATR"] = calcular_atr(df, 14)
    df["BB_Upper"], df["BB_Lower"] = calcular_bollinger(df["Close"])
    df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / df["Close"] * 100
    return df.dropna()

def calcular_score(df):
    ultimo = df.iloc[-1]
    anterior = df.iloc[-2]
    score = 0.0
    detalles = {}

    rsi = ultimo["RSI"]
    if rsi < 20: score += 1.0; detalles["RSI"] = "+1.0 (Sobreventa)"
    elif rsi < 35: score += 0.5; detalles["RSI"] = "+0.5"
    elif rsi > 80: score -= 1.0; detalles["RSI"] = "-1.0 (Sobrecompra)"
    elif rsi > 65: score -= 0.5; detalles["RSI"] = "-0.5"
    else: detalles["RSI"] = "0.0"

    if ultimo["EMA12"] > ultimo["EMA26"]: score += 0.8; detalles["EMA"] = "+0.8"
    else: score -= 0.8; detalles["EMA"] = "-0.8"

    macd = ultimo["MACD"]; macd_sig = ultimo["MACD_Signal"]; macd_hist = ultimo["MACD_Hist"]; macd_hist_prev = anterior["MACD_Hist"]
    if macd > macd_sig and macd_hist > macd_hist_prev: score += 1.0; detalles["MACD"] = "+1.0"
    elif macd > macd_sig: score += 0.5; detalles["MACD"] = "+0.5"
    elif macd < macd_sig and macd_hist < macd_hist_prev: score -= 1.0; detalles["MACD"] = "-1.0"
    elif macd < macd_sig: score -= 0.5; detalles["MACD"] = "-0.5"
    else: detalles["MACD"] = "0.0"

    stoch_k = ultimo["Stoch_K"]; stoch_d = ultimo["Stoch_D"]
    if stoch_k < 20 and stoch_k > stoch_d: score += 1.0; detalles["Stoch"] = "+1.0"
    elif stoch_k < 30: score += 0.5; detalles["Stoch"] = "+0.5"
    elif stoch_k > 80 and stoch_k < stoch_d: score -= 1.0; detalles["Stoch"] = "-1.0"
    elif stoch_k > 70: score -= 0.5; detalles["Stoch"] = "-0.5"
    else: detalles["Stoch"] = "0.0"

    close = ultimo["Close"]; bb_upper = ultimo["BB_Upper"]; bb_lower = ultimo["BB_Lower"]
    if close < bb_lower * 1.005: score += 0.7; detalles["BB"] = "+0.7"
    elif close > bb_upper * 0.995: score -= 0.7; detalles["BB"] = "-0.7"
    else: detalles["BB"] = "0.0"

    return max(-5, min(5, score)), detalles

df_raw = obtener_datos(activo, tf)

if df_raw.empty or len(df_raw) < 60:
    st.error("❌ Sin datos. Prueba otro activo/TF.")
    st.stop()

df = calcular_indicadores(df_raw)
if len(df) < 30:
    st.error("❌ Datos insuficientes.")
    st.stop()

ultimo = df.iloc[-1]
anterior = df.iloc[-2]
score, detalles = calcular_score(df)

THRESHOLD = 2.0
if score >= THRESHOLD:
    senal = "CALL"; color = "#00E676"; emoji = "📈"
elif score <= -THRESHOLD:
    senal = "PUT"; color = "#FF1744"; emoji = "📉"
else:
    senal = "NEUTRAL"; color = "#546E7A"; emoji = "➖"

fuerza = min(100, int((abs(score) / 5.0) * 100)) if abs(score) >= THRESHOLD else int((abs(score) / THRESHOLD) * 50)

atr = ultimo["ATR"]; precio_actual = ultimo["Close"]

if "USD=X" in activo or "JPY=X" in activo or "AUD" in activo:
    decimales = 5; sl_pips = atr * 1.5; tp_pips = atr * 3.0
else:
    decimales = 2; sl_pips = atr * 2.0; tp_pips = atr * 4.0

if senal == "CALL":
    sl = precio_actual - sl_pips; tp = precio_actual + tp_pips
elif senal == "PUT":
    sl = precio_actual + sl_pips; tp = precio_actual - tp_pips
else:
    sl = tp = None

# ======================== LAYOUT COMPACTO ========================
# FILA 1: Señal + Fuerzas
s1, s2, s3 = st.columns([2, 1, 1])
with s1:
    st.markdown(f"""
    <div class="signal-box" style="background-color:{color};">
        <div style="font-size:10px; opacity:0.9;">SEÑAL</div>
        <div style="font-size:28px; margin:0;">{emoji} {senal}</div>
        <div style="font-size:10px; opacity:0.85;">Fuerza {fuerza}%</div>
    </div>
    """, unsafe_allow_html=True)
with s2:
    st.markdown(f"""
    <div class="strength-box" style="background-color:#00C853;">
        <div>ALCISTA</div>
        <div style="font-size:18px;">{max(0, int(score * 10))}%</div>
    </div>
    """, unsafe_allow_html=True)
with s3:
    st.markdown(f"""
    <div class="strength-box" style="background-color:#D50000;">
        <div>BAJISTA</div>
        <div style="font-size:18px;">{max(0, int(-score * 10))}%</div>
    </div>
    """, unsafe_allow_html=True)

# FILA 2: Métricas + SL/TP en una sola fila
precio_str = f"{precio_actual:.{decimales}f}"
sl_str = f"{sl:.{decimales}f}" if sl else "—"
tp_str = f"{tp:.{decimales}f}" if tp else "—"

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Precio", precio_str)
m2.metric("RSI", f"{ultimo['RSI']:.0f}")
m3.metric("ATR", f"{atr:.{decimales}f}")
m4.metric("Score", f"{score:.1f}")
m5.metric("SL", sl_str)
m6.metric("TP", tp_str)

# FILA 3: Gráfico pequeño
st.markdown("<p style='font-size:11px; margin:2px 0;'>📊 Gráfico</p>", unsafe_allow_html=True)

df_plot = df.tail(40).copy()
fig = go.Figure()

fig.add_trace(go.Candlestick(
    x=df_plot.index, open=df_plot["Open"], high=df_plot["High"],
    low=df_plot["Low"], close=df_plot["Close"],
    increasing_line_color="#00E676", decreasing_line_color="#FF1744",
    increasing_fillcolor="rgba(0,230,118,0.2)", decreasing_fillcolor="rgba(255,23,68,0.2)",
    line=dict(width=1), name="P"
))
fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot["EMA12"], mode="lines", line=dict(color="#FFB300", width=1.2), name="E12"))
fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot["EMA26"], mode="lines", line=dict(color="#42A5F5", width=1.2), name="E26"))

if senal != "NEUTRAL" and sl is not None and tp is not None:
    fig.add_hline(y=sl, line_dash="dash", line_color="#FF1744", annotation_text="SL", annotation_position="right", annotation_font_size=9, annotation_font_color="#FF1744")
    fig.add_hline(y=tp, line_dash="dash", line_color="#00E676", annotation_text="TP", annotation_position="right", annotation_font_size=9, annotation_font_color="#00E676")

fig.update_layout(
    height=180,
    margin=dict(l=0, r=0, t=5, b=0),
    xaxis_rangeslider_visible=False,
    template="plotly_dark",
    showlegend=False,
    paper_bgcolor="#0b0e14",
    plot_bgcolor="#0b0e14",
    xaxis=dict(showgrid=False, fixedrange=True, showticklabels=True, color="#888", tickfont=dict(size=8)),
    yaxis=dict(showgrid=True, gridcolor="#1f2636", fixedrange=True, color="#888", side="right", tickfont=dict(size=8)),
    font=dict(color="white", size=9)
)

st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key="main_chart")

# FILA 4: Indicadores en cards pequeñas
st.markdown("<p style='font-size:11px; margin:2px 0;'>📋 Indicadores</p>", unsafe_allow_html=True)

i1, i2, i3, i4, i5 = st.columns(5)
indicadores = [
    ("RSI(14)", f"{ultimo['RSI']:.0f}", "<30|>70"),
    ("MACD", f"{ultimo['MACD']:.{decimales}f}", "Cruce"),
    ("Stoch", f"{ultimo['Stoch_K']:.0f}", "<20|>80"),
    ("ATR", f"{ultimo['ATR']:.{decimales}f}", "Vol"),
    ("BB", f"{ultimo['BB_Width']:.1f}%", "Ancho"),
]
for col, (nombre, valor, desc) in zip([i1, i2, i3, i4, i5], indicadores):
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size:9px; color:#888;">{nombre}</div>
            <div style="font-size:13px; font-weight:600;">{valor}</div>
            <div style="font-size:8px; color:#555;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

# Desglose compacto
with st.expander("🔍 Score", expanded=False):
    st.markdown(f"<p style='font-size:11px; margin:0;'><b>Total:</b> {score:.1f}/5.0 | <b>Umbral:</b> ±{THRESHOLD}</p>", unsafe_allow_html=True)
    for indicador, desc in detalles.items():
        color_ind = "#00E676" if "+" in desc else "#FF1744" if "-" in desc else "#888"
        st.markdown(f"<p style='font-size:10px; margin:0;'><span style='color:{color_ind}'>●</span> <b>{indicador}:</b> {desc}</p>", unsafe_allow_html=True)

st.markdown("""
<div class="disclaimer">
    ⚠️ Herramienta educativa • No es consejo financiero • Alto riesgo de pérdida
</div>
""", unsafe_allow_html=True)

