"""
📊 EXNOVA AI DASHBOARD v5.1 — Limpio & Rápido
Botón gigante + barras de probabilidad + gráfico. 
Procesamiento interno completo (métricas y ensemble ocultos).
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from streamlit_autorefresh import st_autorefresh
from ai_core import *

st.set_page_config(page_title="Exnova AI v5", page_icon="🧠", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');

.stApp{background-color:#0b0e14;color:#e2e8f0;font-family:'Inter',sans-serif}
.block-container{padding:0.5rem 1rem 1rem 1rem;max-width:100%}

.signal-btn{
    width:100%;border-radius:16px;padding:18px 8px;text-align:center;
    font-weight:800;font-size:28px;letter-spacing:1px;
    box-shadow:0 4px 20px rgba(0,0,0,0.5);transition:all 0.3s ease;
    border:2px solid rgba(255,255,255,0.1);margin-bottom:8px;
    animation:pulse 2s infinite;
}
.signal-btn.call{background:linear-gradient(135deg,#00c853,#00e676);color:#000}
.signal-btn.put{background:linear-gradient(135deg,#ff1744,#d50000);color:#fff}
.signal-btn.neutral{background:linear-gradient(135deg,#455a64,#78909c);color:#fff}
@keyframes pulse{0%{transform:scale(1)}50%{transform:scale(1.02)}100%{transform:scale(1)}}

.prob-bar-bg{width:100%;height:22px;background:#1a1f2e;border-radius:10px;overflow:hidden;margin:4px 0}
.prob-bar-fill{height:100%;border-radius:10px;display:flex;align-items:center;justify-content:flex-end;padding-right:8px;font-size:10px;font-weight:700;color:#fff;transition:width 0.8s ease}
.prob-label{font-size:11px;color:#8892a0;margin-top:2px}

.status-pill{display:inline-block;padding:3px 10px;border-radius:12px;font-size:9px;font-weight:600;margin-left:6px}
.status-pill.ok{background:#00c85320;color:#00e676;border:1px solid #00c85340}
.status-pill.warn{background:#ff910020;color:#ff9100;border:1px solid #ff910040}

.hist-row{display:flex;gap:5px;overflow-x:auto;padding:6px 0;scrollbar-width:none}
.hist-row::-webkit-scrollbar{display:none}
.hist-pill{padding:4px 10px;border-radius:14px;font-size:9px;font-weight:700;white-space:nowrap;border:1px solid}

.err-box{background:#1a0a0a;border:1px solid #ff1744;padding:12px;border-radius:8px;margin:8px 0;text-align:center}
.err-box h4{color:#ff1744;margin:0 0 4px 0;font-size:14px}
.err-box p{color:#b0bec5;margin:0;font-size:11px}

.stSelectbox>div>div{background:#11141d!important;border:1px solid #1a1f2e!important;border-radius:8px!important;color:#e2e8f0!important}
.stSelectbox label{color:#8892a0!important;font-size:11px!important}

.disclaimer{font-size:9px;color:#475569;text-align:center;padding:8px;margin-top:8px;border-top:1px solid #1a1f2e}

h1,h2,h3,h4,h5,h6,p{margin:0!important;padding:0!important}
</style>
""", unsafe_allow_html=True)

def _safe_hash(obj):
    return hashlib.md5(str(obj).encode("utf-8")).hexdigest()[:12]

def get_state_key(asset, tf):
    return f"ai_{asset}_{tf}"

def init_session():
    defaults = {
        "refresh": 30, "paused": False,
        "last_asset": None, "last_tf": None,
        "data_hash": None, "last_minute": None,
        "history": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()

c1, c2, c3, c4 = st.columns([2.5, 1.5, 1.5, 1.5])
with c1:
    st.markdown("<h3 style='color:#00d2ff;font-weight:800'>🧠 EXNOVA AI</h3>", unsafe_allow_html=True)
with c2:
    refresh_opt = st.selectbox("Refresh", [10, 30, 60, 120, 300], index=1,
                               label_visibility="collapsed", key="refresh_sel")
    st.session_state.refresh = refresh_opt
with c3:
    paused = st.toggle("⏸ Pausar", value=st.session_state.paused, key="pause_toggle")
    st.session_state.paused = paused
with c4:
    asset = st.selectbox("Activo", ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "BTC", "ETH", "XAUUSD"],
                         index=1, label_visibility="collapsed", key="asset_sel")

am = {"EURUSD": "EURUSD=X", "GBPUSD": "GBPUSD=X", "USDJPY": "USDJPY=X",
      "AUDUSD": "AUDUSD=X", "BTC": "BTC-USD", "ETH": "ETH-USD", "XAUUSD": "GC=F"}
ticker = am[asset]

tf = st.selectbox("Timeframe", ["1m", "5m", "15m", "1h"], index=1,
                  label_visibility="collapsed", key="tf_sel")

if not st.session_state.paused:
    st_autorefresh(interval=st.session_state.refresh * 1000, key="auto_refresh")

st.caption(f"⏱ {st.session_state.refresh}s • {datetime.now().strftime('%H:%M:%S')} • {asset} • {tf}")

state_key = get_state_key(asset, tf)
changed = (st.session_state.last_asset != asset or st.session_state.last_tf != tf)
if changed:
    st.session_state.last_asset = asset
    st.session_state.last_tf = tf
    st.session_state.data_hash = None

@st.cache_data(ttl=30, show_spinner=False)
def fetch_data(tkr, interval):
    try:
        return get_data(tkr, interval)
    except Exception as e:
        logger.error(f"Fetch error: {e}")
        return pd.DataFrame()

df = fetch_data(ticker, tf)

if df.empty or len(df) < 80:
    st.markdown(f"""
    <div class="err-box">
        <h4>📡 Esperando datos...</h4>
        <p>Descargando {asset} ({tf}). Esto puede tomar unos segundos.<br>
        Si persiste, verifica tu conexión o prueba otro timeframe.</p>
    </div>
    """, unsafe_allow_html=True)
    fig_empty = go.Figure()
    fig_empty.update_layout(template="plotly_dark", paper_bgcolor="#0b0e14", plot_bgcolor="#0b0e14",
                            height=200, margin=dict(l=0,r=0,t=20,b=0))
    fig_empty.add_annotation(text="Cargando datos...", xref="paper", yref="paper", showarrow=False,
                             font=dict(size=16, color="#475569"))
    st.plotly_chart(fig_empty, use_container_width=True, config={"displayModeBar": False})
    st.stop()

try:
    df = indis(df)
    f = feats(df)
except Exception as e:
    st.markdown(f"""
    <div class="err-box">
        <h4>⚠️ Error de cálculo</h4>
        <p>{str(e)}</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

ai = None
if state_key in st.session_state:
    ai = st.session_state[state_key]
else:
    ai = ExnovaAI()
    st.session_state[state_key] = ai

ai.asset = asset
ai.tf = tf

if not ai.ready:
    with st.spinner(f"🧠 Analizando {asset} {tf}..."):
        ai.load_or_train(asset, tf, df, f)

if ai.ready and not ai.fallback:
    current_hash = _safe_hash(str(df.index[-1])) if len(df) > 0 else ""
    ai.online_update(df, f, current_hash, st.session_state.data_hash)
    st.session_state.data_hash = current_hash

try:
    result = ai.predict(df, f)
except Exception as e:
    result = ai._neutral_result(str(e)) if ai else {"signal": "NEUTRAL", "pc": 2, "conf": 0, "strength": "NEUTRAL",
        "tech_score": 0, "tech_reasons": [], "votes": {}, "weights": {}, "components": {}, "error": str(e)}

sig = result.get("signal", "NEUTRAL")
pc = result.get("pc", 2)
conf = result.get("conf", 0)
strength = result.get("strength", "NEUTRAL")
tech_score = result.get("tech_score", 0)
tech_reasons = result.get("tech_reasons", [])
votes = result.get("votes", {})
weights = result.get("weights", {})
components = result.get("components", {})

u = df.iloc[-1]
pa = float(u.Close)
atr_val = float(u.ATR)

if "JPY" in asset: dec = 3
elif asset in ["EURUSD", "GBPUSD", "AUDUSD", "XAUUSD"]: dec = 5
else: dec = 2

sl, tpv, sp_raw, tp_raw = calculate_levels(pa, atr_val, sig, dec)

btn_class = "call" if sig == "CALL" else "put" if sig == "PUT" else "neutral"
btn_text = f"📈 {sig}" if sig == "CALL" else f"📉 {sig}" if sig == "PUT" else "➖ NEUTRAL"
btn_sub = f"{conf:.1f}% CONFIANZA • {strength}"

st.markdown(f"""
<div class="signal-btn {btn_class}">
    <div>{btn_text}</div>
    <div style="font-size:13px;font-weight:600;opacity:0.9;margin-top:4px">{btn_sub}</div>
</div>
""", unsafe_allow_html=True)

comp_nn = components.get("nn", {}).get("probs", {"put": 33.3, "call": 33.3, "neutral": 33.4})
put_pct = comp_nn.get("put", 0)
call_pct = comp_nn.get("call", 0)
neu_pct = comp_nn.get("neutral", 0)

st.markdown("<p style='font-size:11px;color:#8892a0;margin:8px 0 4px'>🎯 Probabilidades IA</p>", unsafe_allow_html=True)

st.markdown(f"""
<div class="prob-label">PUT</div>
<div class="prob-bar-bg">
    <div class="prob-bar-fill" style="width:{put_pct}%;background:linear-gradient(90deg,#d50000,#ff1744)">{put_pct:.1f}%</div>
</div>
<div class="prob-label">CALL</div>
<div class="prob-bar-bg">
    <div class="prob-bar-fill" style="width:{call_pct}%;background:linear-gradient(90deg,#00c853,#00e676)">{call_pct:.1f}%</div>
</div>
<div class="prob-label">NEUTRAL</div>
<div class="prob-bar-bg">
    <div class="prob-bar-fill" style="width:{neu_pct}%;background:linear-gradient(90deg,#455a64,#78909c)">{neu_pct:.1f}%</div>
</div>
""", unsafe_allow_html=True)

ps = f"{pa:.{dec}f}"
ss = f"{sl:.{dec}f}" if sl else "—"
ts = f"{tpv:.{dec}f}" if tpv else "—"
rr = f"1:{tp_raw/sp_raw:.1f}" if sig != "NEUTRAL" and sp_raw and sp_raw > 0 else "—"
st.markdown("<p style='font-size:11px;color:#8892a0;margin:8px 0 4px'>📊 Gráfico de precios</p>", unsafe_allow_html=True)

dp = df.tail(60).copy()
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02, row_heights=[0.75, 0.25])

fig.add_trace(go.Candlestick(
    x=dp.index, open=dp["Open"], high=dp["High"], low=dp["Low"], close=dp["Close"],
    increasing_line_color="#00e676", decreasing_line_color="#ff1744",
    increasing_fillcolor="rgba(0,230,118,0.3)", decreasing_fillcolor="rgba(255,23,68,0.3)",
    line=dict(width=1.5), name="Precio"
), row=1, col=1)

fig.add_trace(go.Scatter(x=dp.index, y=dp["E9"], mode="lines",
    line=dict(color="#ff9100", width=1.5), name="EMA 9", opacity=0.9), row=1, col=1)
fig.add_trace(go.Scatter(x=dp.index, y=dp["E26"], mode="lines",
    line=dict(color="#2979ff", width=1.5), name="EMA 26", opacity=0.9), row=1, col=1)

fig.add_trace(go.Scatter(x=dp.index, y=dp["BU"], mode="lines",
    line=dict(color="rgba(255,255,255,0.15)", width=1, dash="dot"), showlegend=False), row=1, col=1)
fig.add_trace(go.Scatter(x=dp.index, y=dp["BL"], mode="lines",
    line=dict(color="rgba(255,255,255,0.15)", width=1, dash="dot"), showlegend=False), row=1, col=1)

if sig != "NEUTRAL" and sl is not None and tpv is not None:
    fig.add_hline(y=sl, line_dash="dash", line_color="#ff1744", line_width=1.5,
        annotation_text=f"SL {ss}", annotation_position="right",
        annotation_font_size=10, annotation_font_color="#ff1744", row=1, col=1)
    fig.add_hline(y=tpv, line_dash="dash", line_color="#00e676", line_width=1.5,
        annotation_text=f"TP {ts}", annotation_position="right",
        annotation_font_size=10, annotation_font_color="#00e676", row=1, col=1)

vc = ["#ff1744" if dp["Close"].iloc[i] < dp["Open"].iloc[i] else "#00e676" for i in range(len(dp))]
fig.add_trace(go.Bar(x=dp.index, y=dp["Volume"], marker_color=vc, opacity=0.6, showlegend=False), row=2, col=1)

fig.update_layout(
    height=420, margin=dict(l=0, r=40, t=5, b=0),
    xaxis_rangeslider_visible=False,
    template="plotly_dark", paper_bgcolor="#0b0e14", plot_bgcolor="#0b0e14", showlegend=False,
    xaxis=dict(showgrid=False, fixedrange=True, showticklabels=False),
    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.03)", fixedrange=True, side="right", tickfont=dict(size=9, color="#64748b")),
    yaxis2=dict(showgrid=False, fixedrange=True, side="right", tickfont=dict(size=8, color="#64748b")),
    font=dict(color="#e2e8f0", size=10),
    hovermode="x unified"
)

st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key=f"chart_{asset}_{tf}")

info = ai.info if ai else {}
fb = ai.fallback if ai else True
status_txt = "✅ Modelo activo" if not fb else "⚠️ Fallback técnico"
status_cls = "ok" if not fb else "warn"
mode_txt = info.get("mode", "—")

st.markdown(f"""
<div style="background:#11141d;border:1px solid #1a1f2e;border-radius:8px;padding:8px 12px;margin:10px 0;font-size:10px;display:flex;align-items:center;justify-content:space-between">
    <span style="color:#8892a0">{status_txt} <span class="status-pill {status_cls}">{mode_txt}</span></span>
    <span style="color:#475569">{asset}/{tf}</span>
</div>
""", unsafe_allow_html=True)

now_key = datetime.now().strftime("%H:%M")
if st.session_state.get("last_minute") != now_key and sig != "NEUTRAL":
    st.session_state.history.append({
        "time": datetime.now().strftime("%H:%M"),
        "asset": asset, "tf": tf,
        "signal": sig, "conf": conf, "price": pa, "strength": strength,
    })
    st.session_state.last_minute = now_key
    st.session_state.history = [h for h in st.session_state.history if h["asset"] == asset and h["tf"] == tf][-30:]

hist_filtered = [h for h in st.session_state.history if h["asset"] == asset and h["tf"] == tf][-15:]
if hist_filtered:
    st.markdown("<p style='font-size:11px;color:#8892a0;margin:8px 0 4px'>🕐 Historial reciente</p>", unsafe_allow_html=True)
    pills = []
    for h in reversed(hist_filtered):
        sc = "#00e676" if h["signal"] == "CALL" else "#ff1744" if h["signal"] == "PUT" else "#78909c"
        pills.append(f'<span class="hist-pill" style="background:{sc}15;color:{sc};border-color:{sc}40">{h["time"]} {h["signal"]} {h["conf"]:.0f}%</span>')
    st.markdown(f'<div class="hist-row">{" ".join(pills)}</div>', unsafe_allow_html=True)

st.markdown('<div class="disclaimer">⚠️ Software educativo. Trading de alto riesgo. Las señales son predicciones estadísticas, no garantías. No es asesoría financiera.</div>', unsafe_allow_html=True)
