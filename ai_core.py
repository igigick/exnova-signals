"""
🧠 EXNOVA AI ENGINE v3.1 — Ensemble Híbrido
NN + RandomForest + Análisis Técnico + Meta-Clasificador
"""

import os
import requests
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from collections import deque
import warnings
warnings.filterwarnings("ignore")

try:
    from sklearn.ensemble import RandomForestClassifier
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False

# ═══════════════════════════════════════════════
# SUPABASE
# ═══════════════════════════════════════════════
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")
BUCKET = "models"

def sb_upload(data_bytes, filename, bucket=BUCKET):
    if not SUPABASE_URL or not SUPABASE_KEY:
        return False
    try:
        url = f"{SUPABASE_URL}/storage/v1/object/{bucket}/{filename}"
        headers = {
            "apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}",
            "Content-Type": "application/octet-stream", "x-upsert": "true"
        }
        r = requests.post(url, headers=headers, data=data_bytes, timeout=15)
        return r.status_code in [200, 201]
    except Exception:
        return False

def sb_download(filename, bucket=BUCKET):
    if not SUPABASE_URL or not SUPABASE_KEY:
        return None
    try:url = f"{SUPABASE_URL}/storage/v1/object/{bucket}/{filename}"
      tru:url = f"{SUPABASE_URL}/storage/v1/object/{bucket}/{filename}"
        headers = {"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}"}
        r = requests.get(url, headers=headers, timeout=15)
        return r.content if r.status_code == 200 else None
    except Exception:
        return None

# ═══════════════════════════════════════════════
# RED NEURONAL PROFUNDA
# ═══════════════════════════════════════════════
class DeepNN:
    def __init__(self, inp, h1=64, h2=32, h3=16, out=3, lr=0.003, l2=0.0001, dropout=0.25):
        np.random.seed(42)
        self.lr, self.l2, self.dropout_rate = lr, l2, dropout
        self.inp, self.h1, self.h2, self.h3, self.out = inp, h1, h2, h3, out
        
        self.W1 = np.random.randn(inp, h1) * np.sqrt(2.0 / inp)
        self.b1 = np.zeros((1, h1))
        self.W2 = np.random.randn(h1, h2) * np.sqrt(2.0 / h1)
        self.b2 = np.zeros((1, h2))
        self.W3 = np.random.randn(h2, h3) * np.sqrt(2.0 / h2)
        self.b3 = np.zeros((1, h3))
        self.W4 = np.random.randn(h3, out) * np.sqrt(2.0 / h3)
        self.b4 = np.zeros((1, out))
        
        self.bn_g1, self.bn_b1 = np.ones((1, h1)), np.zeros((1, h1))
        self.bn_g2, self.bn_b2 = np.ones((1, h2)), np.zeros((1, h2))
        self.bn_g3, self.bn_b3 = np.ones((1, h3)), np.zeros((1, h3))
        
        self.mu = 0.9
        self.vW1 = np.zeros_like(self.W1); self.vb1 = np.zeros_like(self.b1)
        self.vW2 = np.zeros_like(self.W2); self.vb2 = np.zeros_like(self.b2)
        self.vW3 = np.zeros_like(self.W3); self.vb3 = np.zeros_like(self.b3)
        self.vW4 = np.zeros_like(self.W4); self.vb4 = np.zeros_like(self.b4)

    def relu(self, x): return np.maximum(0, x)
    def drelu(self, x): return (x > 0).astype(float)
    
    def softmax(self, x):
        e = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e / np.sum(e, axis=1, keepdims=True)
    
    def batch_norm(self, x, gamma, beta, eps=1e-8):
        m = x.mean(axis=0, keepdims=True)
        v = x.var(axis=0, keepdims=True) + eps
        xn = (x - m) / np.sqrt(v)
        return gamma * xn + beta, m, v, xn

    def forward(self, X, training=False):
        self._z1 = np.dot(X, self.W1) + self.b1
        self._a1, self._bm1, self._bv1, self._bn1 = self.batch_norm(self._z1, self.bn_g1, self.bn_b1)
        self._a1 = self.relu(self._a1)
        if training and self.dropout_rate > 0:
            self._mask1 = (np.random.rand(*self._a1.shape) > self.dropout_rate).astype(float) / (1 - self.dropout_rate)
            self._a1 *= self._mask1
        
        self._z2 = np.dot(self._a1, self.W2) + self.b2
        self._a2, self._bm2, self._bv2, self._bn2 = self.batch_norm(self._z2, self.bn_g2, self.bn_b2)
        self._a2 = self.relu(self._a2)
        if training and self.dropout_rate > 0:
            self._mask2 = (np.random.rand(*self._a2.shape) > self.dropout_rate).astype(float) / (1 - self.dropout_rate)
            self._a2 *= self._mask2
        
        self._z3 = np.dot(self._a2, self.W3) + self.b3
        self._a3, self._bm3, self._bv3, self._bn3 = self.batch_norm(self._z3, self.bn_g3, self.bn_b3)
        self._a3 = self.relu(self._a3)
        
        z4 = np.dot(self._a3, self.W4) + self.b4
        return self.softmax(z4)

    def train(self, X, y, epochs=80, batch_size=64):
        y = np.array(y)
        n = X.shape[0]
        for epoch in range(epochs):
            idx = np.random.permutation(n)
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                b_idx = idx[start:end]
                bx, by = X[b_idx], y[b_idx]
                pred = self.forward(bx, training=True)
                
                dz4 = (pred - by) / bx.shape[0]
                dW4 = np.dot(self._a3.T, dz4) + self.l2 * self.W4
                db4 = np.sum(dz4, axis=0, keepdims=True)
                
                dz3 = np.dot(dz4, self.W4.T) * self.drelu(self._z3)
                dW3 = np.dot(self._a2.T, dz3) + self.l2 * self.W3
                db3 = np.sum(dz3, axis=0, keepdims=True)
                
                dz2 = np.dot(dz3, self.W3.T) * self.drelu(self._z2)
                if hasattr(self, '_mask2'): dz2 *= self._mask2
                dW2 = np.dot(self._a1.T, dz2) + self.l2 * self.W2
                db2 = np.sum(dz2, axis=0, keepdims=True)
                
                dz1 = np.dot(dz2, self.W2.T) * self.drelu(self._z1)
                if hasattr(self, '_mask1'): dz1 *= self._mask1
                dW1 = np.dot(bx.T, dz1) + self.l2 * self.W1
                db1 = np.sum(dz1, axis=0, keepdims=True)
                
                for W, b, dW, db, vW, vb in [
                    (self.W4, self.b4, dW4, db4, self.vW4, self.vb4),
                    (self.W3, self.b3, dW3, db3, self.vW3, self.vb3),
                    (self.W2, self.b2, dW2, db2, self.vW2, self.vb2),
                    (self.W1, self.b1, dW1, db1, self.vW1, self.vb1),
                ]:
                    vW[:] = self.mu * vW + self.lr * dW
                    vb[:] = self.mu * vb + self.lr * db
                    W -= vW
                    b -= vb
        return self

    def predict(self, X):
        return self.forward(X, training=False)

    def save(self, path):
        np.savez(path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2,
                 W3=self.W3, b3=self.b3, W4=self.W4, b4=self.b4,
                 inp=self.inp, h1=self.h1, h2=self.h2, h3=self.h3, out=self.out)

    @classmethod
    def load(cls, path):
        data = np.load(path)
        nn = cls(int(data["inp"]), int(data["h1"]), int(data["h2"]), int(data["h3"]), int(data["out"]))
        nn.W1, nn.b1 = data["W1"], data["b1"]
        nn.W2, nn.b2 = data["W2"], data["b2"]
        nn.W3, nn.b3 = data["W3"], data["b3"]
        nn.W4, nn.b4 = data["W4"], data["b4"]
        return nn
# ═══════════════════════════════════════════════
# INDICADORES TÉCNICOS
# ═══════════════════════════════════════════════
def rsi(s, w=14):
    d = s.diff()
    g, l = d.where(d > 0, 0), (-d).where(d < 0, 0)
    ag, al = g.rolling(w, min_periods=1).mean(), l.rolling(w, min_periods=1).mean()
    rs = ag / al.replace(0, 0.001)
    return 100 - (100 / (1 + rs.replace([np.inf, -np.inf], 1).fillna(1)))

def ema(s, p): return s.ewm(span=p, adjust=False).mean()
def sma(s, p): return s.rolling(p, min_periods=1).mean()

def macd(s):
    m = ema(s, 12) - ema(s, 26)
    return m, ema(m, 9), m - ema(m, 9)

def atr(df):
    h, l, c = df["High"], df["Low"], df["Close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(14, min_periods=1).mean()

def bb(s):
    m = sma(s, 20)
    sd = s.rolling(20, min_periods=1).std().replace(0, 0.001)
    return m + 2 * sd, m - 2 * sd, m

def stoch(df):
    ll = df["Low"].rolling(14, min_periods=1).min()
    hh = df["High"].rolling(14, min_periods=1).max()
    r = (hh - ll).replace(0, 0.001)
    k = 100 * (df["Close"] - ll) / r
    return k, k.rolling(3, min_periods=1).mean()

def chop(df):
    h, l, c = df["High"], df["Low"], df["Close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    a = tr.rolling(14, min_periods=1).sum()
    mx, mn = h.rolling(14, min_periods=1).max(), l.rolling(14, min_periods=1).min()
    r = (mx - mn).replace(0, 0.001)
    return (100 * np.log10(a / r) / np.log10(14)).fillna(50)

def adx(df):
    h, l, c = df["High"], df["Low"], df["Close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    plus_dm = (h - h.shift()).where((h - h.shift()) > (l.shift() - l), 0).clip(lower=0)
    minus_dm = (l.shift() - l).where((l.shift() - l) > (h - h.shift()), 0).clip(lower=0)
    atr14 = tr.rolling(14, min_periods=1).mean()
    plus_di = 100 * plus_dm.rolling(14, min_periods=1).mean() / atr14.replace(0, 0.001)
    minus_di = 100 * minus_dm.rolling(14, min_periods=1).mean() / atr14.replace(0, 0.001)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 0.001)) * 100
    return dx.rolling(14, min_periods=1).mean().fillna(25)

def williams_r(df):
    hh = df["High"].rolling(14, min_periods=1).max()
    ll = df["Low"].rolling(14, min_periods=1).min()
    r = (hh - ll).replace(0, 0.001)
    return -100 * (hh - df["Close"]) / r

def cci(df):
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    sma_tp = sma(tp, 20)
    md = (tp - sma_tp).abs().rolling(20, min_periods=1).mean()
    return (tp - sma_tp) / (0.015 * md.replace(0, 0.001))

def mfi(df):
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    rmf = tp * df["Volume"]
    d = tp.diff()
    pmf = rmf.where(d > 0, 0).rolling(14, min_periods=1).sum()
    nmf = rmf.where(d < 0, 0).rolling(14, min_periods=1).sum()
    return 100 - (100 / (1 + pmf / nmf.replace(0, 0.001)))

def candle_patterns(df):
    o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]
    body = (c - o).abs()
    rng = h - l
    upper = h - np.maximum(c, o)
    lower = np.minimum(c, o) - l
    patterns = pd.DataFrame(index=df.index)
    patterns["doji"] = (body / rng.replace(0, 0.001) < 0.1).astype(int)
    patterns["hammer"] = ((lower > body * 2) & (upper < body * 0.5)).astype(int)
    patterns["inv_hammer"] = ((upper > body * 2) & (lower < body * 0.5)).astype(int)
    patterns["engulfing_bull"] = ((o.shift() > c.shift()) & (o < c) & (o <= c.shift()) & (c >= o.shift())).astype(int)
    patterns["engulfing_bear"] = ((o.shift() < c.shift()) & (o > c) & (o >= c.shift()) & (c <= o.shift())).astype(int)
    return patterns

def indis(df):
    df = df.copy()
    df["RSI"] = rsi(df["Close"])
    df["WR"] = williams_r(df)
    df["CCI"] = cci(df)
    df["MFI"] = mfi(df)
    df["E9"] = ema(df["Close"], 9)
    df["E12"] = ema(df["Close"], 12)
    df["E26"] = ema(df["Close"], 26)
    df["E50"] = ema(df["Close"], 50)
    df["SMA20"] = sma(df["Close"], 20)
    df["SMA50"] = sma(df["Close"], 50)
    df["MACD"], df["MS"], df["MH"] = macd(df["Close"])
    df["SK"], df["SD"] = stoch(df)
    df["ATR"] = atr(df)
    df["BU"], df["BL"], df["BM"] = bb(df["Close"])
    df["BW"] = (df["BU"] - df["BL"]) / df["Close"].replace(0, 0.001) * 100
    df["CH"] = chop(df)
    df["ADX"] = adx(df)
    df["RET"] = df["Close"].pct_change()
    df["VOL"] = df["RET"].rolling(14, min_periods=1).std()
    df["OBV"] = (np.sign(df["Close"].diff()) * df["Volume"]).cumsum()
    pat = candle_patterns(df)
    for col in pat.columns:
        df[col] = pat[col]
    df["MOM10"] = df["Close"].diff(10) / df["Close"].shift(10).replace(0, 0.001)
    df["MOM20"] = df["Close"].diff(20) / df["Close"].shift(20).replace(0, 0.001)
    return df.dropna()

def feats(df):
    f = pd.DataFrame(index=df.index)
    cs = max(df["Close"].std(), 0.001)
    cm = max(df["Close"].mean(), 0.001)
    f["r"] = df["RSI"] / 100
    f["wr"] = df["WR"] / -100
    f["cci"] = np.tanh(df["CCI"] / 100)
    f["mfi"] = df["MFI"] / 100
    f["m"] = np.tanh(df["MACD"] / cs)
    f["h"] = np.tanh(df["MH"] / cs)
    f["k"] = df["SK"] / 100
    f["d"] = df["SD"] / 100
    f["e9"] = np.tanh((df["Close"] / df["E9"].replace(0, 0.001) - 1) * 10)
    f["e12"] = np.tanh((df["Close"] / df["E12"].replace(0, 0.001) - 1) * 10)
    f["e26"] = np.tanh((df["Close"] / df["E26"].replace(0, 0.001) - 1) * 10)
    f["e50"] = np.tanh((df["Close"] / df["E50"].replace(0, 0.001) - 1) * 10)
    f["sma20"] = np.tanh((df["Close"] / df["SMA20"].replace(0, 0.001) - 1) * 10)
    f["sma50"] = np.tanh((df["Close"] / df["SMA50"].replace(0, 0.001) - 1) * 10)
    f["a"] = np.tanh(df["ATR"] / cm)
    bw = (df["BU"] - df["BL"]).replace(0, 0.001)
    f["bb"] = (df["Close"] - df["BM"]) / bw * 2
    f["bw"] = np.tanh(df["BW"] / 10)
    f["c"] = df["CH"] / 100
    f["v"] = np.tanh(df["VOL"] * 10)
    f["rt"] = np.tanh(df["RET"] * 10)
    f["adx"] = df["ADX"] / 100
    f["obv"] = np.tanh(df["OBV"].diff() / (df["OBV"].abs().mean() + 0.001))
    f["mom10"] = np.tanh(df["MOM10"] * 5)
    f["mom20"] = np.tanh(df["MOM20"] * 5)
    for p in ["doji", "hammer", "inv_hammer", "engulfing_bull", "engulfing_bear"]:
        f[p] = df[p].astype(float)
    return f.fillna(0).replace([np.inf, -np.inf], 0)

def dataset(df, f, lb=15, horizon=5):
    X, y = [], []
    fa = f.values
    c = df["Close"].values
    atr_mean = df["ATR"].mean()
    price_mean = df["Close"].mean()
    threshold = max(0.0008, (atr_mean / price_mean) * 0.35) if price_mean > 0 else 0.002
    n_features = lb * f.shape[1]
    for i in range(lb, len(fa) - horizon):
        window = fa[i - lb:i].flatten()
        if len(window) == n_features and np.isfinite(window).all():
            X.append(window)
            fr = (c[i + horizon] - c[i]) / c[i]
            if fr > threshold:
                y.append([0, 1, 0])
            elif fr < -threshold:
                y.append([1, 0, 0])
            else:
                y.append([0, 0, 1])
    return np.array(X), np.array(y), threshold

def get_data(ticker, interval):
    try:
        pm = {"1m": "7d", "5m": "30d", "15m": "60d", "1h": "180d", "4h": "365d"}
        df = yf.download(ticker, period=pm.get(interval, "30d"), interval=interval,
                         progress=False, auto_adjust=True, prepost=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col not in df.columns:
                return pd.DataFrame()
        return df.dropna()
    except Exception:
        return pd.DataFrame()

# ═══════════════════════════════════════════════
# MODELO TÉCNICO
# ═══════════════════════════════════════════════
def technical_model(df):
    u = df.iloc[-1]
    scores = {"PUT": 0, "CALL": 0, "NEUTRAL": 0}
    reasons = {"PUT": [], "CALL": [], "NEUTRAL": []}
    
    if u.RSI < 30:
        scores["CALL"] += 20; reasons["CALL"].append("RSI sobreventa")
    elif u.RSI > 70:
        scores["PUT"] += 20; reasons["PUT"].append("RSI sobrecompra")
    elif 45 < u.RSI < 55:
        scores["NEUTRAL"] += 15; reasons["NEUTRAL"].append("RSI neutral")
    
    if u.MACD > u.MS and u.MH > 0:
        scores["CALL"] += 18; reasons["CALL"].append("MACD alcista")
    elif u.MACD < u.MS and u.MH < 0:
        scores["PUT"] += 18; reasons["PUT"].append("MACD bajista")
    
    if u.Close > u.E9 > u.E26:
        scores["CALL"] += 15; reasons["CALL"].append("Tendencia alcista EMA")
    elif u.Close < u.E9 < u.E26:
        scores["PUT"] += 15; reasons["PUT"].append("Tendencia bajista EMA")
    
    if u.SK > u.SD and u.SK < 80:
        scores["CALL"] += 12; reasons["CALL"].append("Stoch alcista")
    elif u.SK < u.SD and u.SK > 20:
        scores["PUT"] += 12; reasons["PUT"].append("Stoch bajista")
    
    if u.Close > u.BU:
        scores["PUT"] += 10; reasons["PUT"].append("Sobre BB superior")
    elif u.Close < u.BL:
        scores["CALL"] += 10; reasons["CALL"].append("Bajo BB inferior")
    
    if u.ADX > 25:
        if scores["CALL"] > scores["PUT"]:
            scores["CALL"] += 10; reasons["CALL"].append("ADX fuerte")
        elif scores["PUT"] > scores["CALL"]:
            scores["PUT"] += 10; reasons["PUT"].append("ADX fuerte")
    
    if u.WR < -80:
        scores["CALL"] += 8; reasons["CALL"].append("Williams sobreventa")
    elif u.WR > -20:
        scores["PUT"] += 8; reasons["PUT"].append("Williams sobrecompra")
    
    if u.CCI < -100:
        scores["CALL"] += 8; reasons["CALL"].append("CCI sobreventa")
    elif u.CCI > 100:
        scores["PUT"] += 8; reasons["PUT"].append("CCI sobrecompra")
    
    if u.MFI < 20:
        scores["CALL"] += 8; reasons["CALL"].append("MFI sobreventa")
    elif u.MFI > 80:
        scores["PUT"] += 8; reasons["PUT"].append("MFI sobrecompra")
    
    if u.engulfing_bull or u.hammer:
        scores["CALL"] += 12; reasons["CALL"].append("Patrón alcista")
    if u.engulfing_bear or u.inv_hammer:
        scores["PUT"] += 12; reasons["PUT"].append("Patrón bajista")
    
    total = sum(scores.values()) or 1
    probs = {k: v / total for k, v in scores.items()}
    pred = max(probs, key=probs.get)
    return pred, probs[pred] * 100, probs, reasons[pred]

# ═══════════════════════════════════════════════
# META-CLASIFICADOR
# ═══════════════════════════════════════════════
class MetaClassifier:
    def __init__(self):
        self.weights = {"nn": 0.4, "rf": 0.35, "tech": 0.25}
        self.history = deque(maxlen=100)
        self.accuracy = {"nn": 0.5, "rf": 0.5, "tech": 0.5}
    
    def update_weights(self):
        if len(self.history) < 10:
            return
        for model in ["nn", "rf", "tech"]:
            correct = sum(1 for h in self.history if h.get(model) == h["actual"])
            self.accuracy[model] = max(0.3, correct / len(self.history))
        total = sum(self.accuracy.values())
        self.weights = {k: v / total for k, v in self.accuracy.items()}
    
    def ensemble_predict(self, nn_pred, nn_conf, rf_pred, rf_conf, tech_pred, tech_conf):
        votes = {"PUT": 0, "CALL": 0, "NEUTRAL": 0}
        nn_w = self.weights["nn"] * (nn_conf / 100)
        rf_w = self.weights["rf"] * (rf_conf / 100)
        tech_w = self.weights["tech"] * (tech_conf / 100)
        votes[nn_pred] += nn_w
        votes[rf_pred] += rf_w
        votes[tech_pred] += tech_w
        final = max(votes, key=votes.get)
        total = sum(votes.values()) or 1
        conf = (votes[final] / total) * 100
        agree = sum(1 for p in [nn_pred, rf_pred, tech_pred] if p == final)
        if agree >= 2:
            conf = min(99, conf * 1.15)
        return final, conf, votes, self.weights
    
    def record(self, predictions, actual):
        self.history.append({**predictions, "actual": actual})
        if len(self.history) % 5 == 0:
            self.update_weights()

# ═══════════════════════════════════════════════
# IA PRINCIPAL
# ═══════════════════════════════════════════════
class ExnovaAI:
    def __init__(self):
        self.nn = None
        self.rf = None
        self.meta = MetaClassifier()
        self.ready = False
        self.fallback = False
        self.info = {}
        self.threshold = 0.002
        self.last_predictions = {}
    
    def _save_nn(self, asset, tf):
        if self.nn:
            path = f"/tmp/nn_{asset}_{tf}.npz"
            self.nn.save(path)
            with open(path, "rb") as f:
                sb_upload(f.read(), f"nn_{asset}_{tf}.npz")
    
    def _load_nn(self, asset, tf):
        data = sb_download(f"nn_{asset}_{tf}.npz")
        if data:
            path = f"/tmp/nn_{asset}_{tf}.npz"
            with open(path, "wb") as f:
                f.write(data)
            return DeepNN.load(path)
        return None
    
    def train(self, asset, tf, df, f):
        try:
            X, y, threshold = dataset(df, f)
            self.threshold = threshold
            n = len(X)
            if n < 40:
                self.ready = True
                self.fallback = True
                self.info = {"error": f"Muestras insuficientes ({n})"}
                return False
            split = int(n * 0.85)
            X_train, y_train = X[:split], y[:split]
            n_features = X.shape[1]
            h1 = min(64, max(32, n_features // 2))
            h2 = min(32, max(16, h1 // 2))
            h3 = min(16, max(8, h2 // 2))
            nn = DeepNN(n_features, h1, h2, h3, 3, lr=0.003, l2=0.0001, dropout=0.2)
            nn.train(X_train, y_train, epochs=60, batch_size=min(64, n))
            self.nn = nn
            if SKLEARN_OK:
                y_labels = np.argmax(y_train, axis=1)
                rf = RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_split=5, random_state=42)
                rf.fit(X_train, y_labels)
                self.rf = rf
            self.ready = True
            self.fallback = False
            self.info = {
                "samples": n, "features": n_features,
                "h1": h1, "h2": h2, "h3": h3,
                "threshold": threshold, "mode": "Entrenamiento ensemble",
                "sklearn": SKLEARN_OK
            }
            self._save_nn(asset, tf)
            return True
        except Exception as e:
            self.ready = True
            self.fallback = True
            self.info = {"error": str(e)}
            return False
    
    def load_or_train(self, asset, tf, df, f):
        loaded = self._load_nn(asset, tf)
        if loaded:
            self.nn = loaded
            self.ready = True
            self.fallback = False
            self.info = {"mode": "Cargado desde nube", "source": "Supabase"}
            return True
        return self.train(asset, tf, df, f)
    
    def online_update(self, df, f, current_hash, last_hash):
        if not self.ready or self.fallback or self.nn is None or current_hash == last_hash:
            return False
        try:
            X, y, _ = dataset(df, f)
            if len(X) > 5:
                batch = min(50, len(X))
                self.nn.train(X[-batch:], y[-batch:], epochs=3, batch_size=batch)
                self.info["mode"] = f"Online learning ({batch})"
                self.info["last_update"] = datetime.now().strftime("%H:%M:%S")
                self._save_nn(self.asset, self.tf)
                return True
        except Exception:
            pass
        return False
    
    def predict(self, df, f):
        fa = f.values
        if len(fa) < 15 or self.nn is None:
            return self._neutral_result("Modelo no listo")
        try:
            inp = fa[-15:].flatten().reshape(1, -1)
            expected = self.nn.inp
            if inp.shape[1] < expected:
                inp = np.pad(inp, ((0, 0), (0, expected - inp.shape[1])), constant_values=0)
            elif inp.shape[1] > expected:
                inp = inp[:, :expected]
            nn_p = self.nn.predict(inp)[0]
            nn_pc = np.argmax(nn_p)
            nn_conf = nn_p[nn_pc] * 100
            nn_sig = ["PUT", "CALL", "NEUTRAL"][nn_pc]
            if self.rf is not None and SKLEARN_OK:
                rf_pc = self.rf.predict(inp)[0]
                rf_proba = self.rf.predict_proba(inp)[0]
                rf_conf = rf_proba[rf_pc] * 100
                rf_sig = ["PUT", "CALL", "NEUTRAL"][rf_pc]
            else:
                rf_sig, rf_conf, rf_pc = nn_sig, nn_conf * 0.8, nn_pc
            tech_sig, tech_conf, tech_probs, tech_reasons = technical_model(df)
            final_sig, final_conf, votes, weights = self.meta.ensemble_predict(
                nn_sig, nn_conf, rf_sig, rf_conf, tech_sig, tech_conf
            )
            final_pc = ["PUT", "CALL", "NEUTRAL"].index(final_sig)
            tech_score = int(tech_conf)
            if final_conf >= 80 and tech_score >= 50:
                strength = "FUERTE"
            elif final_conf >= 60:
                strength = "MEDIA"
            elif final_conf >= 45:
                strength = "DÉBIL"
            else:
                strength = "NEUTRAL"
                final_sig = "NEUTRAL"
                final_pc = 2
            if final_sig != "NEUTRAL":
                agree = sum(1 for p in [nn_sig, rf_sig, tech_sig] if p == final_sig)
                if agree < 2 or tech_score < 40:
                    final_sig = "NEUTRAL"
                    final_pc = 2
                    strength = "NEUTRAL"
            self.last_predictions = {"nn": nn_sig, "rf": rf_sig, "tech": tech_sig, "actual": "PENDING"}
            return {
                "signal": final_sig, "pc": final_pc, "conf": final_conf,
                "strength": strength, "tech_score": tech_score,
                "tech_reasons": tech_reasons, "votes": votes, "weights": weights,
                "components": {
                    "nn": {"signal": nn_sig, "conf": nn_conf, "probs": {"put": nn_p[0]*100, "call": nn_p[1]*100, "neutral": nn_p[2]*100}},
                    "rf": {"signal": rf_sig, "conf": rf_conf},
                    "tech": {"signal": tech_sig, "conf": tech_conf, "probs": {k: v*100 for k, v in tech_probs.items()}}
                }
            }
        except Exception as e:
            return self._neutral_result(str(e))
    
    def _neutral_result(self, reason=""):
        return {
            "signal": "NEUTRAL", "pc": 2, "conf": 0,
            "strength": "NEUTRAL", "tech_score": 0,
            "tech_reasons": [], "votes": {}, "weights": self.meta.weights,
            "components": {}, "error": reason
        }

def calculate_levels(price, atr_val, signal, dec):
    if dec == 5:
        sp, tp = atr_val * 1.2, atr_val * 2.4
    else:
        sp, tp = atr_val * 1.5, atr_val * 3.0
    if signal == "CALL":
        return price - sp, price + tp, sp, tp
    elif signal == "PUT":
        return price + sp, price - tp, sp, tp
    return None, None, sp, tp
