import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import os
import warnings
warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════
# CONFIGURACIÓN DE PÁGINA
# ═══════════════════════════════════════════════
st.set_page_config(
    page_title="Exnova AI Pro",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

REFRESH_DEFAULT = 30
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# ═══════════════════════════════════════════════
# CSS CUSTOM
# ═══════════════════════════════════════════════
st.markdown("""
<style>
.stApp{background-color:#0b0e14;color:#e0e0e0}
.block-container{padding:0.3rem 0.8rem 0rem 0.8rem;max-width:100%}
.mini-box{padding:6px 4px;border-radius:8px;text-align:center;color:white;font-weight:700;font-size:11px;margin:2px;transition:all 0.3s ease}
.metric-mini{background-color:#151a25;padding:4px 2px;border-radius:5px;text-align:center;border:1px solid #1f2636;font-size:10px}
.disclaimer{font-size:9px;color:#555;text-align:center;padding:2px;margin-top:4px}
h1,h2,h3,h4,h5,h6,p{margin:0!important;padding:0!important}
.stSelectbox{margin-bottom:-10px!important}
.stSelectbox label{font-size:11px!important;margin-bottom:2px!important}
iframe{height:350px!important}
.metric-row{display:flex;justify-content:space-between;margin-top:6px;gap:4px}
.metric-item{flex:1;background:#151a25;border:1px solid #1f2636;border-radius:5px;padding:4px;text-align:center}
.metric-item .label{font-size:8px;color:#888}
.metric-item .value{font-size:12px;font-weight:600;color:#e0e0e0}
.metric-item .hint{font-size:7px;color:#555}
.history-row{display:flex;gap:4px;overflow-x:auto;padding:4px 0}
.history-pill{padding:3px 8px;border-radius:12px;font-size:9px;font-weight:600;white-space:nowrap}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════
# RED NEURONAL MEJORADA (L2, validación, persistencia)
# ═══════════════════════════════════════════════
class NN:
    def __init__(self, inp, hid, out, lr=0.01, l2=0.001):
        np.random.seed(42)
        self.lr = lr
        self.l2 = l2
        self.inp = inp
        self.hid = hid
        self.out = out
        self._init_weights()
        self.history = {"loss": [], "val_loss": [], "val_acc": []}

    def _init_weights(self):
        self.W1 = np.random.randn(self.inp, self.hid) * np.sqrt(2.0 / self.inp)
        self.b1 = np.zeros((1, self.hid))
        self.W2 = np.random.randn(self.hid, self.out) * np.sqrt(2.0 / self.hid)
        self.b2 = np.zeros((1, self.out))

    def relu(self, x): return np.maximum(0, x)
    def drelu(self, x): return (x > 0).astype(float)

    def softmax(self, x):
        e = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e / np.sum(e, axis=1, keepdims=True)

    def forward(self, X):
        z1 = np.dot(X, self.W1) + self.b1
        a1 = self.relu(z1)
        z2 = np.dot(a1, self.W2) + self.b2
        return self.softmax(z2), a1, z1

    def cross_entropy(self, pred, y):
        eps = 1e-9
        return -np.mean(np.sum(y * np.log(pred + eps), axis=1))

    def accuracy(self, pred, y):
        return np.mean(np.argmax(pred, axis=1) == np.argmax(y, axis=1))

    def train(self, X, y, epochs=50, val_split=0.2, patience=10):
        y = np.array(y)
        n = X.shape[0]
        
        if n > 20 and val_split > 0:
            split_idx = int(n * (1 - val_split))
            idx = np.random.permutation(n)
            X_tr, y_tr = X[idx[:split_idx]], y[idx[:split_idx]]
            X_val, y_val = X[idx[split_idx:]], y[idx[split_idx:]]
        else:
            X_tr, y_tr = X, y
            X_val, y_val = None, None

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(epochs):
            pred, a1, z1 = self.forward(X_tr)
            loss = self.cross_entropy(pred, y_tr) + self.l2 * (np.sum(self.W1**2) + np.sum(self.W2**2))

            dz2 = (pred - y_tr) / X_tr.shape[0]
            dW2 = np.dot(a1.T, dz2) + self.l2 * self.W2
            db2 = np.sum(dz2, axis=0, keepdims=True)
            dz1 = np.dot(dz2, self.W2.T) * self.drelu(z1)
            dW1 = np.dot(X_tr.T, dz1) + self.l2 * self.W1
            db1 = np.sum(dz1, axis=0, keepdims=True)

            self.W2 -= self.lr * dW2
            self.b2 -= self.lr * db2
            self.W1 -= self.lr * dW1
            self.b1 -= self.lr * db1

            if X_val is not None:
                val_pred, _, _ = self.forward(X_val)
                val_loss = self.cross_entropy(val_pred, y_val)
                val_acc = self.accuracy(val_pred, y_val)
                self.history["val_loss"].append(val_loss)
                self.history["val_acc"].append(val_acc)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break
            else:
                self.history["val_loss"].append(loss)
                self.history["val_acc"].append(0)

            self.history["loss"].append(loss)

        return self.history

    def predict(self, X):
        pred, _, _ = self.forward(X)
        return pred

    def save(self, path):
        np.savez(path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2,
                 inp=self.inp, hid=self.hid, out=self.out, lr=self.lr, l2=self.l2)

    @classmethod
    def load(cls, path):
        data = np.load(path)
        nn = cls(int(data["inp"]), int(data["hid"]), int(data["out"]),
                 float(data["lr"]), float(data["l2"]))
        nn.W1, nn.b1, nn.W2, nn.b2 = data["W1"], data["b1"], data["W2"], data["b2"]
        return nn

# ═══════════════════════════════════════════════
# INDICADORES TÉCNICOS
# ═══════════════════════════════════════════════
def rsi(s, w=14):
    d = s.diff()
    g = d.where(d > 0, 0)
    l = (-d).where(d < 0, 0)
    ag = g.rolling(w, min_periods=w).mean()
    al = l.rolling(w, min_periods=w).mean()
    rs = ag / al
    rs = rs.replace([np.inf, -np.inf], 1).fillna(1)
    return 100 - (100 / (1 + rs))

def ema(s, p): return s.ewm(span=p, adjust=False).mean()

def macd(s):
    m = ema(s, 12) - ema(s, 26)
    sig = ema(m, 9)
    return m, sig, m - sig

def atr(df):
    h, l, c = df["High"], df["Low"], df["Close"]
    tr1 = h - l
    tr2 = (h - c.shift()).abs()
    tr3 = (l - c.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(14, min_periods=1).mean()

def bb(s):
    m = s.rolling(20, min_periods=1).mean()
    sd = s.rolling(20, min_periods=1).std().replace(0, 0.001)
    return m + 2 * sd, m - 2 * sd

def stoch(df):
    ll = df["Low"].rolling(14, min_periods=1).min()
    hh = df["High"].rolling(14, min_periods=1).max()
    r = (hh - ll).replace(0, 0.001)
    k = 100 * (df["Close"] - ll) / r
    return k, k.rolling(3, min_periods=1).mean()

def chop(df):
    h, l, c = df["High"], df["Low"], df["Close"]
    tr1 = h - l
    tr2 = (h - c.shift()).abs()
    tr3 = (l - c.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    a = tr.rolling(14, min_periods=1).sum()
    mx = h.rolling(14, min_periods=1).max()
    mn = l.rolling(14, min_periods=1).min()
    r = (mx - mn).replace(0, 0.001)
    return (100 * np.log10(a / r) / np.log10(14)).fillna(50)

def indis(df):
    df = df.copy()
    df["RSI"] = rsi(df["Close"])
    df["E12"] = ema(df["Close"], 12)
    df["E26"] = ema(df["Close"], 26)
    df["MACD"], df["MS"], df["MH"] = macd(df["Close"])
    df["SK"], df["SD"] = stoch(df)
    df["ATR"] = atr(df)
    df["BU"], df["BL"] = bb(df["Close"])
    df["BW"] = (df["BU"] - df["BL"]) / df["Close"].replace(0, 0.001) * 100
    df["CH"] = chop(df)
    df["RET"] = df["Close"].pct_change()
    df["VOL"] = df["RET"].rolling(14, min_periods=1).std()
    return df.dropna()

# ═══════════════════════════════════════════════
# FEATURE ENGINEERING
# ═══════════════════════════════════════════════
def feats(df):
    f = pd.DataFrame(index=df.index)
    cs = df["Close"].std()
    cs = cs if cs > 0 else 0.001
    cm = df["Close"].mean()
    cm = cm if cm > 0 else 0.001
    
    f["r"] = df["RSI"] / 100
    f["m"] = np.tanh(df["MACD"] / cs)
    f["h"] = np.tanh(df["MH"] / cs)
    f["k"] = df["SK"] / 100
    f["d"] = df["SD"] / 100
    f["e12"] = (df["Close"] / df["E12"].replace(0, 0.001) - 1) * 10
    f["e26"] = (df["Close"] / df["E26"].replace(0, 0.001) - 1) * 10
    f["a"] = np.tanh(df["ATR"] / cm)
    bw = (df["BU"] - df["BL"]).replace(0, 0.001)
    f["bb"] = (df["Close"] - (df["BU"] + df["BL"]) / 2) / bw * 2
    f["c"] = df["CH"] / 100
    f["v"] = np.tanh(df["VOL"] * 10)
    f["rt"] = np.tanh(df["RET"] * 10)
    return f.fillna(0).replace([np.inf, -np.inf], 0)

# ═══════════════════════════════════════════════
# DATASET BUILDER (umbral adaptativo)
# ═══════════════════════════════════════════════
def dataset(df, f, lb=10, horizon=5):
    X, y = [], []
    fa = f.values
    c = df["Close"].values
    
    atr_mean = df["ATR"].mean()
    price_mean = df["Close"].mean()
    threshold = max(0.001, (atr_mean / price_mean) * 0.5) if price_mean > 0 else 0.003
    
    for i in range(lb, len(fa) - horizon):
        window = fa[i - lb:i].flatten()
        if len(window) == lb * f.shape[1] and np.isfinite(window).all():
            X.append(window)
            fr = (c[i + horizon] - c[i]) / c[i]
            if fr > threshold:
                y.append([0, 1, 0])
            elif fr < -threshold:
                y.append([1, 0, 0])
            else:
                y.append([0, 0, 1])
    return np.array(X), np.array(y), threshold

# ═══════════════════════════════════════════════
# DESCARGA DE DATOS
# ═══════════════════════════════════════════════
@st.cache_data(ttl=60, show_spinner=False)
def get_data(ticker, interval):
    try:
        pm = {"1m": "7d", "5m": "30d", "15m": "60d", "1h": "180d"}
        df = yf.download(
            ticker,
            period=pm.get(interval, "30d"),
            interval=interval,
            progress=False,
            auto_adjust=True,
            prepost=False
        )
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col not in df.columns:
                return pd.DataFrame()
        return df.dropna()
    except Exception as e:
        st.error(f"Error descargando datos: {e}")
        return pd.DataFrame()

# ═══════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════
def init_state():
    defaults = {
        "nn": None, "done": False, "fb": False,
        "last_asset": None, "last_tf": None,
        "history": [], "refresh": REFRESH_DEFAULT,
        "paused": False, "train_info": {}, "threshold": 0.003,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ═══════════════════════════════════════════════
# HEADER Y SELECTORES
# ═══════════════════════════════════════════════
c1, c2, c3, c4 = st.columns([2, 2, 2, 2])

with c1:
    st.markdown("<h4 style='margin:0;color:#00d2ff'>🧠 Exnova AI Pro</h4>", unsafe_allow_html=True)

with c2:
    refresh_opt = st.selectbox(
        "⏱ Refresh",
        [10, 30, 60, 120, 300],
        index=[10, 30, 60, 120, 300].index(st.session_state.refresh)
        if st.session_state.refresh in [10, 30, 60, 120, 300] else 1,
        label_visibility="collapsed",
        key="refresh_sel"
    )
    st.session_state.refresh = refresh_opt

with c3:
    paused = st.toggle("⏸ Pausar", value=st.session_state.paused, key="pause_toggle")
    st.session_state.paused = paused

with c4:
    asset = st.selectbox(
        "",
        ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "BTC", "ETH"],
        index=0,
        label_visibility="collapsed",
        key="asset_sel"
    )
    am = {
        "EURUSD": "EURUSD=X", "GBPUSD": "GBPUSD=X",
        "USDJPY": "USDJPY=X", "AUDUSD": "AUDUSD=X",
        "BTC": "BTC-USD", "ETH": "ETH-USD"
    }
    ticker = am[asset]

tf = st.selectbox(
    "Timeframe",
    ["1m", "5m", "15m", "1h"],
    index=1,
    label_visibility="collapsed",
    key="tf_sel"
)

if not st.session_state.paused:
    st_autorefresh(interval=st.session_state.refresh * 1000, key="auto_refresh")

st.caption(f"⏱ {st.session_state.refresh}s • {datetime.now().strftime('%H:%M:%S')} • {asset} • {tf}")

# ═══════════════════════════════════════════════
# DETECTAR CAMBIO DE ACTIVO/TF
# ═══════════════════════════════════════════════
if st.session_state.last_asset != asset or st.session_state.last_tf != tf:
    st.session_state.done = False
    st.session_state.fb = False
    st.session_state.nn = None
    st.session_state.train_info = {}
    st.session_state.history = []
    st.session_state.last_asset = asset
    st.session_state.last_tf = tf

# ═══════════════════════════════════════════════
# DESCARGAR Y PROCESAR DATOS
# ═══════════════════════════════════════════════
df = get_data(ticker, tf)

if df.empty or len(df) < 80:
    st.error("❌ Sin datos suficientes.")
    st.stop()

df = indis(df)
if len(df) < 60:
    st.error("❌ Datos insuficientes tras indicadores.")
    st.stop()

f = feats(df)

# ═══════════════════════════════════════════════
# ENTRENAMIENTO
# ═══════════════════════════════════════════════
model_path = os.path.join(MODEL_DIR, f"{asset}_{tf}.npz")

if not st.session_state.done:
    st.info("🧠 Entrenando modelo de IA...")
    try:
        X, y, threshold = dataset(df, f)
        st.session_state.threshold = threshold
        
        if len(X) > 30:
            n_features = X.shape[1]
            hidden = min(12, max(4, n_features // 8))
            
            nn = NN(n_features, hidden, 3, lr=0.01, l2=0.001)
            hist = nn.train(X, y, epochs=100, val_split=0.2, patience=15)
            
            st.session_state.nn = nn
            st.session_state.done = True
            st.session_state.fb = False
            st.session_state.train_info = {
                "samples": len(X),
                "features": n_features,
                "hidden": hidden,
                "epochs_trained": len(hist["loss"]),
                "final_loss": hist["loss"][-1] if hist["loss"] else None,
                "final_val_acc": hist["val_acc"][-1] if hist["val_acc"] else None,
                "threshold": threshold,
            }
            nn.save(model_path)
        else:
            if os.path.exists(model_path):
                st.session_state.nn = NN.load(model_path)
                st.session_state.done = True
                st.session_state.fb = False
                st.session_state.train_info = {"samples": len(X), "loaded": True}
            else:
                st.session_state.done = True
                st.session_state.fb = True
                st.session_state.train_info = {"samples": len(X), "error": "Muestras insuficientes"}
    except Exception as e:
        if os.path.exists(model_path):
            st.session_state.nn = NN.load(model_path)
            st.session_state.done = True
            st.session_state.fb = False
            st.session_state.train_info = {"loaded": True, "error": str(e)}
        else:
            st.session_state.done = True
            st.session_state.fb = True
            st.session_state.train_info = {"error": str(e)}

# ═══════════════════════════════════════════════
# PREDICCIÓN
# ═══════════════════════════════════════════════
fa = f.values

if len(fa) >= 10 and st.session_state.nn is not None and not st.session_state.fb:
    try:
        inp = fa[-10:].flatten().reshape(1, -1)
        expected = st.session_state.nn.inp
        
        if inp.shape[1] < expected:
            inp = np.pad(inp, ((0, 0), (0, expected - inp.shape[1])), constant_values=0)
        elif inp.shape[1] > expected:
            inp = inp[:, :expected]
            
        p = st.session_state.nn.predict(inp)
        pc = np.argmax(p)
        conf = np.max(p) * 100
    except Exception as e:
        p = [[0.33, 0.33, 0.34]]
        pc = 2
        conf = 34
        st.session_state.fb = True
else:
    p = [[0.33, 0.33, 0.34]]
    pc = 2
    conf = 34

sig = ["PUT", "CALL", "NEUTRAL"][pc]
col = ["#FF1744", "#00E676", "#546E7A"][pc]
em = ["📉", "📈", "➖"][pc]
put_pct = p[0][0] * 100
call_pct = p[0][1] * 100
neu_pct = p[0][2] * 100

u = df.iloc[-1]
atr_val = u.ATR
pa = u.Close

dec = 5 if "USD=X" in ticker or "JPY=X" in ticker or "AUD" in ticker else 2

sp = atr_val * 1.5 if dec == 5 else atr_val * 2
tp = atr_val * 3 if dec == 5 else atr_val * 4
if sig == "CALL":
    sl = pa - sp
    tpv = pa + tp
elif sig == "PUT":
    sl = pa + sp
    tpv = pa - tp
else:
    sl = tpv = None
# ═══════════════════════════════════════════════
# PARTE 2 — UI, GRÁFICO, INDICADORES, HISTORIAL
# ═══════════════════════════════════════════════

# ───────────────────────────────────────────────
# CAJAS DE SEÑAL
# ───────────────────────────────────────────────
b1, b2, b3 = st.columns(3)
with b1:
    st.markdown(
        f'<div class="mini-box" style="background-color:#D50000">'
        f'<div>PUT</div><div style="font-size:18px">{put_pct:.1f}%</div></div>',
        unsafe_allow_html=True
    )
with b2:
    st.markdown(
        f'<div class="mini-box" style="background-color:{col};box-shadow:0 0 15px {col}40">'
        f'<div>IA → {sig}</div><div style="font-size:24px">{em}</div>'
        f'<div style="font-size:10px">{conf:.1f}% confianza</div></div>',
        unsafe_allow_html=True
    )
with b3:
    st.markdown(
        f'<div class="mini-box" style="background-color:#00C853">'
        f'<div>CALL</div><div style="font-size:18px">{call_pct:.1f}%</div></div>',
        unsafe_allow_html=True
    )

# ───────────────────────────────────────────────
# MÉTRICAS EN FILA
# ───────────────────────────────────────────────
ps = f"{pa:.{dec}f}"
ss = f"{sl:.{dec}f}" if sl is not None else "—"
ts = f"{tpv:.{dec}f}" if tpv is not None else "—"

st.markdown(f"""
<div class="metric-row">
<div class="metric-item"><div class="label">Precio</div><div class="value">{ps}</div></div>
<div class="metric-item"><div class="label">RSI</div><div class="value">{u.RSI:.0f}</div></div>
<div class="metric-item"><div class="label">Chop</div><div class="value">{u.CH:.0f}</div></div>
<div class="metric-item"><div class="label">ATR</div><div class="value">{atr_val:.{dec}f}</div></div>
<div class="metric-item"><div class="label">SL</div><div class="value" style="color:#FF1744">{ss}</div></div>
<div class="metric-item"><div class="label">TP</div><div class="value" style="color:#00E676">{ts}</div></div>
</div>
""", unsafe_allow_html=True)

# ───────────────────────────────────────────────
# GRÁFICO DE VELAS
# ───────────────────────────────────────────────
st.markdown("<p style='font-size:11px;margin:8px 0 4px 0'>📊 Gráfico de velas</p>", unsafe_allow_html=True)

dp = df.tail(80).copy()
fig = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.03,
    row_heights=[0.78, 0.22]
)

fig.add_trace(go.Candlestick(
    x=dp.index,
    open=dp["Open"],
    high=dp["High"],
    low=dp["Low"],
    close=dp["Close"],
    increasing_line_color="#00E676",
    decreasing_line_color="#FF1744",
    increasing_fillcolor="#00E676",
    decreasing_fillcolor="#FF1744",
    line=dict(width=1),
    name="Precio"
), row=1, col=1)

fig.add_trace(go.Scatter(
    x=dp.index, y=dp["E12"],
    mode="lines",
    line=dict(color="#FFB300", width=1.2),
    name="EMA 12"
), row=1, col=1)

fig.add_trace(go.Scatter(
    x=dp.index, y=dp["E26"],
    mode="lines",
    line=dict(color="#42A5F5", width=1.2),
    name="EMA 26"
), row=1, col=1)

fig.add_trace(go.Scatter(
    x=dp.index, y=dp["BU"],
    mode="lines",
    line=dict(color="rgba(255,255,255,0.15)", width=1),
    showlegend=False,
    name="BB Sup"
), row=1, col=1)

fig.add_trace(go.Scatter(
    x=dp.index, y=dp["BL"],
    mode="lines",
    line=dict(color="rgba(255,255,255,0.15)", width=1),
    fill="tonexty",
    fillcolor="rgba(255,255,255,0.04)",
    showlegend=False,
    name="BB Inf"
), row=1, col=1)

if sig != "NEUTRAL" and sl is not None:
    fig.add_hline(
        y=sl, line_dash="dash", line_color="#FF1744",
        annotation_text="SL", annotation_position="right",
        annotation_font_size=9, annotation_font_color="#FF1744",
        row=1, col=1
    )
    fig.add_hline(
        y=tpv, line_dash="dash", line_color="#00E676",
        annotation_text="TP", annotation_position="right",
        annotation_font_size=9, annotation_font_color="#00E676",
        row=1, col=1
    )

vc = ["#FF1744" if dp["Close"].iloc[i] < dp["Open"].iloc[i] else "#00E676"
      for i in range(len(dp))]
fig.add_trace(go.Bar(
    x=dp.index, y=dp["Volume"],
    marker_color=vc,
    showlegend=False,
    name="Volumen"
), row=2, col=1)

fig.update_layout(
    height=380,
    margin=dict(l=0, r=0, t=2, b=0),
    xaxis_rangeslider_visible=False,
    template="plotly_dark",
    paper_bgcolor="#0b0e14",
    plot_bgcolor="#0b0e14",
    showlegend=False,
    xaxis=dict(showgrid=False, fixedrange=True, showticklabels=True,
               color="#888", tickfont=dict(size=8)),
    yaxis=dict(showgrid=True, gridcolor="#1f2636", fixedrange=True,
               color="#888", side="right", tickfont=dict(size=8)),
    yaxis2=dict(showgrid=False, fixedrange=True, color="#888",
                side="right", tickfont=dict(size=8)),
    font=dict(color="white", size=9)
)

st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key="chart")

# ───────────────────────────────────────────────
# INDICADORES EN FILA
# ───────────────────────────────────────────────
st.markdown("<p style='font-size:11px;margin:8px 0 4px 0'>📋 Indicadores técnicos</p>", unsafe_allow_html=True)

i1, i2, i3, i4, i5 = st.columns(5)
inds = [
    ("RSI", f"{u.RSI:.0f}", "<30 sobreventa | >70 sobrecompra"),
    ("MACD", f"{u.MACD:.{dec}f}", f"Señal: {u.MS:.{dec}f}"),
    ("Stoch", f"{u.SK:.0f}", f"D: {u.SD:.0f} | <20|>80"),
    ("Chop", f"{u.CH:.0f}", "<38 tendencia | >62 rango"),
    ("Vol", f"{u.VOL*100:.2f}%", "Desv. 14d")
]

for col, (n, v, d) in zip([i1, i2, i3, i4, i5], inds):
    with col:
        st.markdown(
            f'<div class="metric-mini">'
            f'<div style="font-size:8px;color:#888">{n}</div>'
            f'<div style="font-size:13px;font-weight:600">{v}</div>'
            f'<div style="font-size:7px;color:#555">{d}</div></div>',
            unsafe_allow_html=True
        )

# ───────────────────────────────────────────────
# INFO DEL MODELO
# ───────────────────────────────────────────────
info = st.session_state.train_info
fb = st.session_state.fb

status_color = "#FF1744" if fb else "#00E676"
status_icon = "⚠️ Fallback" if fb else "✅ Modelo activo"

st.markdown(f"""
<div style="background:linear-gradient(135deg,#0f2027,#203a43);border:1px solid {status_color};padding:6px;border-radius:6px;margin:8px 0;font-size:10px">
<p style="margin:0;color:{status_color}"><b>🧬 {status_icon}</b> 
| PUT {put_pct:.1f}% | CALL {call_pct:.1f}% | NEUTRAL {neu_pct:.1f}% 
| Umbral: {st.session_state.threshold*100:.2f}% | Chop: {u.CH:.0f}</p>
</div>
""", unsafe_allow_html=True)

with st.expander("🔧 Detalles del modelo", expanded=False):
    if info.get("samples"):
        st.write(f"**Muestras de entrenamiento:** {info['samples']}")
    if info.get("features"):
        st.write(f"**Features:** {info['features']}")
    if info.get("hidden"):
        st.write(f"**Neuronas ocultas:** {info['hidden']}")
    if info.get("epochs_trained"):
        st.write(f"**Épocas entrenadas:** {info['epochs_trained']}")
    if info.get("final_val_acc") is not None:
        st.write(f"**Accuracy de validación:** {info['final_val_acc']*100:.1f}%")
    if info.get("final_loss") is not None:
        st.write(f"**Loss final:** {info['final_loss']:.4f}")
    if info.get("loaded"):
        st.write("📦 Modelo cargado desde disco")
    if info.get("error"):
        st.write(f"❌ Error: {info['error']}")

# ───────────────────────────────────────────────
# HISTORIAL DE SEÑALES
# ───────────────────────────────────────────────
if "last_signal_ts" not in st.session_state or st.session_state.last_signal_ts != str(dp.index[-1]):
    st.session_state.history.append({
        "time": datetime.now().strftime("%H:%M:%S"),
        "signal": sig,
        "conf": conf,
        "price": pa,
        "rsi": u.RSI,
        "chop": u.CH,
    })
    st.session_state.last_signal_ts = str(dp.index[-1])
    st.session_state.history = st.session_state.history[-20:]

if st.session_state.history:
    st.markdown("<p style='font-size:11px;margin:8px 0 4px 0'>🕐 Historial de señales</p>", unsafe_allow_html=True)
    pills = []
    for h in reversed(st.session_state.history[-10:]):
        sig_color = {"PUT": "#FF1744", "CALL": "#00E676", "NEUTRAL": "#546E7A"}[h["signal"]]
        pills.append(
            f'<span class="history-pill" style="background:{sig_color}30;color:{sig_color};border:1px solid {sig_color}60">'
            f'{h["time"]} {h["signal"]} {h["conf"]:.0f}%</span>'
        )
    st.markdown(f'<div class="history-row">{" ".join(pills)}</div>', unsafe_allow_html=True)

# ───────────────────────────────────────────────
# DISCLAIMER
# ───────────────────────────────────────────────
st.markdown(
    '<div class="disclaimer">'
    '⚠️ Este software es de carácter educativo. La IA entrenada con datos históricos '
    'NO garantiza resultados futuros. El trading conlleva riesgo significativo de pérdida. '
    'No constituye asesoramiento financiero.</div>',
    unsafe_allow_html=True
)
