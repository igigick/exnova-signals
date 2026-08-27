"""
📊 EXNOVA AI DASHBOARD v3.1
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from streamlit_autorefresh import st_autorefresh
from ai_core import *

st.set_page_config(page_title="Exnova AI Pro", page_icon="🧠", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
.stApp{background-color:#080a0f;color:#e0e0e0}
.block-container{padding:0.3rem 0.8rem 0rem 0.8rem;max-width:100%}
.mini-box{padding:8px 4px;border-radius:10px;text-align:center;color:white;font-weight:700;font-size:12px;margin:2px;box-shadow:0 2px 10px rgba(0,0,0,0.4)}
.metric-mini{background-color:#11141d;padding:6px 4px;border-radius:6px;text-align:center;border:1px solid #1a1f2e;font-size:10px}
.disclaimer{font-size:9px;color:#444;text-align:center;padding:4px;margin-top:6px}
h1,h2,h3,h4,h5,h6,p{margin:0!important;padding:0!important}
.stSelectbox{margin-bottom:-10px!important}
.stSelectbox label{font-size:11px!important;margin-bottom:2px!important}
.metric-row{display:flex;justify-content:space-between;margin-top:6px;gap:4px}
.metric-item{flex:1;background:#11141d;border:1px solid #1a1f2e;border-radius:6px;padding:5px;text-align:center}
.metric-item .label{font-size:7px;color:#666;text-transform:uppercase;letter-spacing:0.5px}
.metric-item .value{font-size:13px;font-weight:700;color:#e0e0e0}
.history-row{display:flex;gap:5px;overflow-x:auto;padding:6px 0}
.history-pill{padding:4px 10px;border-radius:14px;font-size:9px;font-weight:700;white-space:nowrap}
.reason-tag{font-size:8px;padding:2px 6px;border-radius:4px;background:#0f1525;color:#64b5f6;display:inline-block;margin:1px;border:1px solid #1a2a40}
</style>
""", unsafe_allow_html=True)

defaults = {
    "ai": None, "done": False, "history": [],
    "refresh": 30, "paused": False, "last_asset": None,
    "last_tf": None, "data_hash": None, "last_minute": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

c1, c2, c3, c4 = st.columns([2, 2, 2, 2])
with c1:
    st.markdown("<h4 style='margin:0;color:#00d2ff'>🧠 Exnova AI Pro</h4>", unsafe_allow_html=True)
with c2:
    refresh_opt = st.selectbox("⏱ Refresh", [10, 30, 60, 120, 300], index=1,
                               label_visibility="collapsed", key="refresh_sel")
    st.session_state.refresh = refresh_opt
with c3:
    paused = st.toggle("⏸ Pausar", value=st.session_state.paused, key="pause_toggle")
    st.session_state.paused = paused
with c4:
    asset = st.selectbox("", ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "BTC", "ETH", "XAUUSD"],
                         index=0, label_visibility="collapsed", key="asset_sel")
    am = {
        "EURUSD": "EURUSD=X", "GBPUSD": "GBPUSD=X", "USDJPY": "USDJPY=X",
        "AUDUSD": "AUDUSD=X", "BTC": "BTC-USD", "ETH": "ETH-USD", "XAUUSD": "GC=F"
    }
    ticker = am[asset]

tf = st.selectbox("TF", ["1m", "5m", "15m", "1h"], index=1,
                  label_visibility="collapsed", key="tf_sel")

if not st.session_state.paused:
    st_autorefresh(interval=st.session_state.refresh * 1000, key="auto_refresh")

st.caption(f"⏱ {st.session_state.refresh}s • {datetime.now().strftime('%H:%M:%S')} • {asset} • {tf}")

if st.session_state.last_asset != asset or st.session_state.last_tf != tf:
    st.session_state.done = False
    st.session_state.ai = None
    st.session_state.history = []
    st.session_state.data_hash = None
    st.session_state.last_asset = asset
    st.session_state.last_tf = tf

df = get_data(ticker, tf)
if df.empty or len(df) < 100:
    st.error("❌ Sin datos suficientes.")
    st.stop()

df = indis(df)
if len(df) < 80:
    st.error("❌ Datos insuficientes tras indicadores.")
    st.stop()

f = feats(df)

if not st.session_state.done:
    with st.spinner("🧠 Entrenando IA Ensemble..."):
        ai = ExnovaAI()
        ai.asset = asset
        ai.tf = tf
        ai.load_or_train(asset, tf, df, f)
        st.session_state.ai = ai
        st.session_state.done = True

ai = st.session_state.ai
ai.asset = asset
ai.tf = tf

current_hash = hash(str(df.index[-1])) if len(df) > 0 else 0
ai.online_update(df, f, current_hash, st.session_state.data_hash)
st.session_state.data_hash = current_hash

result = ai.predict(df, f)
sig = result["signal"]
pc = result["pc"]
conf = result["conf"]
strength = result["strength"]
tech_score = result["tech_score"]
tech_reasons = result.get("tech_reasons", [])
votes = result.get("votes", {})
weights = result.get("weights", {})
components = result.get("components", {})

u = df.iloc[-1]
pa = u.Close
atr_val = u.ATR
dec = 5 if any(x in ticker for x in ["USD=X", "JPY=X", "AUD", "GC=F"]) else 2
sl, tpv, sp_raw, tp_raw = calculate_levels(pa, atr_val, sig, dec)

sig_colors = {"PUT": "#FF1744", "CALL": "#00E676", "NEUTRAL": "#546E7A"}
col_main = sig_colors.get(sig, "#546E7A")
em = ["📉", "📈", "➖"][pc]

b1, b2, b3 = st.columns(3)
comp = components.get("nn", {}).get("probs", {"put": 33, "call": 33, "neutral": 34})
with b1:
    st.markdown(f'<div class="mini-box" style="background-color:#D50000"><div style="font-size:10px">PUT</div><div style="font-size:20px">{comp.get("put", 0):.1f}%</div></div>', unsafe_allow_html=True)
with b2:
    glow = f"box-shadow:0 0 25px {col_main}70" if sig != "NEUTRAL" else ""
    st.markdown(f'<div class="mini-box" style="background-color:{col_main};{glow}"><div style="font-size:10px">IA → {sig}</div><div style="font-size:28px">{em}</div><div style="font-size:11px">{conf:.1f}% confianza</div><div style="font-size:9px;opacity:0.8">Fuerza: {strength}</div></div>', unsafe_allow_html=True)
with b3:
    st.markdown(f'<div class="mini-box" style="background-color:#00C853"><div style="font-size:10px">CALL</div><div style="font-size:20px">{comp.get("call", 0):.1f}%</div></div>', unsafe_allow_html=True)

ps = f"{pa:.{dec}f}"
ss = f"{sl:.{dec}f}" if sl is not None else "—"
ts = f"{tpv:.{dec}f}" if tpv is not None else "—"
rr = f"1:{tp_raw/sp_raw:.1f}" if sig != "NEUTRAL" and sp_raw > 0 else "—"

st.markdown(f"""
<div class="metric-row">
<div class="metric-item"><div class="label">Precio</div><div class="value">{ps}</div></div>
<div class="metric-item"><div class="label">RSI</div><div class="value">{u.RSI:.0f}</div></div>
<div class="metric-item"><div class="label">ADX</div><div class="value">{u.ADX:.0f}</div></div>
<div class="metric-item"><div class="label">ATR</div><div class="value">{atr_val:.{dec}f}</div></div>
<div class="metric-item"><div class="label">SL</div><div class="value" style="color:#FF1744">{ss}</div></div>
<div class="metric-item"><div class="label">TP</div><div class="value" style="color:#00E676">{ts}</div></div>
<div class="metric-item"><div class="label">R/R</div><div class="value" style="color:#FFD700">{rr}</div></div>
</div>
""", unsafe_allow_html=True)

if tech_reasons:
    tags = " ".join([f'<span class="reason-tag">✓ {r}</span>' for r in tech_reasons[:6]])
    st.markdown(f'<div style="text-align:center;margin:6px 0">{tags}</div>', unsafe_allow_html=True)

st.markdown("<p style='font-size:11px;margin:8px 0 4px 0'>📊 Gráfico de velas</p>", unsafe_allow_html=True)

dp = df.tail(80).copy()
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.78, 0.22])

fig.add_trace(go.Candlestick(
    x=dp.index, open=dp["Open"], high=dp["High"], low=dp["Low"], close=dp["Close"],
    increasing_line_color="#00E676", decreasing_line_color="#FF1744",
    increasing_fillcolor="#00E676", decreasing_fillcolor="#FF1744",
    line=dict(width=1), name="Precio"
), row=1, col=1)

fig.add_trace(go.Scatter(x=dp.index, y=dp["E9"], mode="lines",
    line=dict(color="#FF5722", width=1), name="EMA 9"), row=1, col=1)
fig.add_trace(go.Scatter(x=dp.index, y=dp["E26"], mode="lines",
    line=dict(color="#42A5F5", width=1.2), name="EMA 26"), row=1, col=1)

fig.add_trace(go.Scatter(x=dp.index, y=dp["BU"], mode="lines",
    line=dict(color="rgba(255,255,255,0.1)", width=1), showlegend=False), row=1, col=1)
fig.add_trace(go.Scatter(x=dp.index, y=dp["BL"], mode="lines",
    line=dict(color="rgba(255,255,255,0.1)", width=1), fill="tonexty",
    fillcolor="rgba(255,255,255,0.02)", showlegend=False), row=1, col=1)

if sig != "NEUTRAL" and sl is not None:
    fig.add_hline(y=sl, line_dash="dash", line_color="#FF1744",
        annotation_text="SL", annotation_position="right",
        annotation_font_size=9, annotation_font_color="#FF1744", row=1, col=1)
    fig.add_hline(y=tpv, line_dash="dash", line_color="#00E676",
        annotation_text="TP", annotation_position="right",
        annotation_font_size=9, annotation_font_color="#00E676", row=1, col=1)

vc = ["#FF1744" if dp["Close"].iloc[i] < dp["Open"].iloc[i] else "#00E676" for i in range(len(dp))]
fig.add_trace(go.Bar(x=dp.index, y=dp["Volume"], marker_color=vc, showlegend=False), row=2, col=1)

fig.update_layout(
    height=380, margin=dict(l=0, r=0, t=2, b=0),
    xaxis_rangeslider_visible=False,
    template="plotly_dark", paper_bgcolor="#080a0f", plot_bgcolor="#080a0f", showlegend=False,
    xaxis=dict(showgrid=False, fixedrange=True, showticklabels=True, color="#888", tickfont=dict(size=8)),
    yaxis=dict(showgrid=True, gridcolor="#1a1f2e", fixedrange=True, color="#888", side="right", tickfont=dict(size=8)),
    yaxis2=dict(showgrid=False, fixedrange=True, color="#888", side="right", tickfont=dict(size=8)),
    font=dict(color="white", size=9)
)

st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key="chart")

st.markdown("<p style='font-size:11px;margin:8px 0 4px 0'>📋 Indicadores técnicos</p>", unsafe_allow_html=True)

i1, i2, i3, i4, i5, i6 = st.columns(6)
inds = [
    ("RSI", f"{u.RSI:.0f}", "<30 sobreventa | >70 sobrecompra"),
    ("MACD", f"{u.MACD:.{dec}f}", f"Señal: {u.MS:.{dec}f}"),
    ("Stoch", f"{u.SK:.0f}", f"D: {u.SD:.0f}"),
    ("ADX", f"{u.ADX:.0f}", ">25 tendencia fuerte"),
    ("CCI", f"{u.CCI:.0f}", "±100 extremos"),
    ("MFI", f"{u.MFI:.0f}", "<20|>80 extremos"),
]
for col, (n, v, d) in zip([i1, i2, i3, i4, i5, i6], inds):
    with col:
        st.markdown(f'<div class="metric-mini"><div style="font-size:8px;color:#888">{n}</div><div style="font-size:13px;font-weight:600">{v}</div><div style="font-size:7px;color:#555">{d}</div></div>', unsafe_allow_html=True)

st.markdown("<p style='font-size:11px;margin:8px 0 4px 0'>🧬 Votos del Ensemble</p>", unsafe_allow_html=True)

if votes:
    cols = st.columns(3)
    vote_colors = {"PUT": "#FF1744", "CALL": "#00E676", "NEUTRAL": "#546E7A"}
    for i, (model, label) in enumerate([("nn", "Red Neuronal"), ("rf", "Random Forest"), ("tech", "Técnico")]):
        with cols[i]:
            w = weights.get(model, 0.33)
            comp_sig = components.get(model, {}).get("signal", "NEUTRAL")
            comp_conf = components.get(model, {}).get("conf", 0)
            c = vote_colors.get(comp_sig, "#546E7A")
            st.markdown(f"""
            <div style="background:#11141d;border:1px solid #1a1f2e;border-radius:6px;padding:6px;text-align:center">
                <div style="font-size:9px;color:#888">{label}</div>
                <div style="font-size:13px;font-weight:700;color:{c}">{comp_sig}</div>
                <div style="font-size:9px;color:#666">{comp_conf:.0f}% conf</div>
                <div style="font-size:8px;color:#444">Peso: {w:.0%}</div>
            </div>
            """, unsafe_allow_html=True)

info = ai.info
fb = ai.fallback
status_color = "#FF1744" if fb else "#00E676"
status_icon = "⚠️ Fallback" if fb else "✅ IA activa"
mode_text = info.get("mode", "")
last_up = info.get("last_update", "")

st.markdown(f"""
<div style="background:linear-gradient(135deg,#0a1520,#152030);border:1px solid {status_color};padding:8px;border-radius:8px;margin:8px 0;font-size:10px">
<p style="margin:0;color:{status_color}"><b>🧬 {status_icon}</b> 
| Score técnico: {tech_score}/100 | {mode_text} {last_up}</p>
</div>
""", unsafe_allow_html=True)

with st.expander("🔧 Detalles del modelo", expanded=False):
    if info.get("samples"):
        st.write(f"**Muestras:** {info['samples']}")
    if info.get("features"):
        st.write(f"**Features:** {info['features']}")
    if info.get("h1"):
        st.write(f"**Arquitectura NN:** {info['h1']}→{info.get('h2','?')}→{info.get('h3','?')}")
    st.write(f"**Sklearn:** {'✅' if info.get('sklearn') else '❌'}")
    if info.get("error"):
        st.write(f"❌ Error: {info['error']}")

now_key = datetime.now().strftime("%H:%M")
if st.session_state.get("last_minute") != now_key and sig != "NEUTRAL":
    st.session_state.history.append({
        "time": datetime.now().strftime("%H:%M:%S"),
        "signal": sig, "conf": conf, "price": pa, "strength": strength,
    })
    st.session_state.last_minute = now_key
    st.session_state.history = st.session_state.history[-50:]

if st.session_state.history:
    st.markdown("<p style='font-size:11px;margin:8px 0 4px 0'>🕐 Historial de señales</p>", unsafe_allow_html=True)
    pills = []
    for h in reversed(st.session_state.history[-15:]):
        sc = sig_colors[h["signal"]]
        pills.append(f'<span class="history-pill" style="background:{sc}20;color:{sc};border:1px solid {sc}60">{h["time"]} {h["signal"]} {h["conf"]:.0f}% [{h["strength"]}]</span>')
    st.markdown(f'<div class="history-row">{" ".join(pills)}</div>', unsafe_allow_html=True)

st.markdown('<div class="disclaimer">⚠️ Software educativo. IA Ensemble (NN+RF+Técnico) con pesos adaptativos. Trading de alto riesgo. No es asesoría financiera.</div>', unsafe_allow_html=True)
