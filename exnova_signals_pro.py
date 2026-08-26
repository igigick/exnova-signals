import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
from streamlit_autorefresh import st_autorefresh
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Exnova AI", page_icon="🧠", layout="wide", initial_sidebar_state="collapsed")
REFRESH_INTERVAL = 10
st_autorefresh(interval=REFRESH_INTERVAL * 1000, key="auto_refresh")

st.markdown("""
<style>
.stApp{background-color:#0b0e14;color:#e0e0e0}
.block-container{padding:0.3rem 1rem 0rem 1rem;max-width:100%}
.signal-box{padding:6px 4px;border-radius:10px;text-align:center;margin:2px 0;color:white;font-weight:700}
.strength-box{padding:4px 2px;border-radius:8px;text-align:center;color:white;font-weight:600;font-size:11px}
.metric-card{background-color:#151a25;padding:4px 2px;border-radius:6px;text-align:center;border:1px solid #1f2636;font-size:10px}
.disclaimer{font-size:9px;color:#555;text-align:center;padding:2px;margin-top:4px}
h1,h2,h3,h4,h5,h6,p{margin:0!important;padding:0!important}
.stSelectbox{margin-bottom:-10px!important}
.stSelectbox label{font-size:11px!important;margin-bottom:0!important}
.stMetric{margin:0!important;padding:0!important}
.stMetric label{font-size:10px!important}
.stMetric div{font-size:14px!important}
iframe{height:220px!important}
</style>
""", unsafe_allow_html=True)

class NN:
    def __init__(self,inp,hid,out,lr=0.01):
        np.random.seed(42)
        self.lr=lr
        self.W1=np.random.randn(inp,hid)*np.sqrt(2.0/inp)
        self.b1=np.zeros((1,hid))
        self.W2=np.random.randn(hid,out)*np.sqrt(2.0/hid)
        self.b2=np.zeros((1,out))
    def relu(self,x):return np.maximum(0,x)
    def drelu(self,x):return (x>0).astype(float)
    def softmax(self,x):
        e=np.exp(x-np.max(x,axis=1,keepdims=True))
        return e/np.sum(e,axis=1,keepdims=True)
    def forward(self,X):
        self.z1=np.dot(X,self.W1)+self.b1
        self.a1=self.relu(self.z1)
        self.z2=np.dot(self.a1,self.W2)+self.b2
        return self.softmax(self.z2)
    def train(self,X,y,epochs=100):
        for _ in range(epochs):
            pred=self.forward(X)
            dz2=(pred-y)/X.shape[0]
            dW2=np.dot(self.a1.T,dz2);db2=np.sum(dz2,axis=0,keepdims=True)
            dz1=np.dot(dz2,self.W2.T)*self.drelu(self.z1)
            dW1=np.dot(X.T,dz1);db1=np.sum(dz1,axis=0,keepdims=True)
            self.W2-=self.lr*dW2;self.b2-=self.lr*db2
            self.W1-=self.lr*dW1;self.b1-=self.lr*db1
        return pred

def rsi(s,w=14):
    d=s.diff();g=d.where(d>0,0);l=(-d).where(d<0,0)
    ag=g.rolling(w,w).mean();al=l.rolling(w,w).mean()
    return 100-(100/(1+ag/al))
def ema(s,p):return s.ewm(span=p,adjust=False).mean()
def macd(s):
    m=ema(s,12)-ema(s,26)
    return m,ema(m,9),m-ema(m,9)
def atr(df):
    tr=pd.concat([df["High"]-df["Low"],(df["High"]-df["Close"].shift()).abs(),(df["Low"]-df["Close"].shift()).abs()],axis=1).max(axis=1)
    return tr.rolling(14,1).mean()
def bb(s):m=s.rolling(20,1).mean();sd=s.rolling(20,1).std();return m+2*sd,m-2*sd
def stoch(df):
    ll=df["Low"].rolling(14,1).min();hh=df["High"].rolling(14,1).max()
    k=100*(df["Close"]-ll)/(hh-ll);return k,k.rolling(3,1).mean()
def chop(df):
    tr=pd.concat([df["High"]-df["Low"],(df["High"]-df["Close"].shift()).abs(),(df["Low"]-df["Close"].shift()).abs()],axis=1).max(axis=1)
    a=tr.rolling(14,1).sum();mx=df["High"].rolling(14,1).max();mn=df["Low"].rolling(14,1).min()
    r=mx-mn;r=r.replace(0,np.nan)
    return (100*np.log10(a/r)/np.log10(14)).fillna(50)

def indis(df):
    df=df.copy()
    df["RSI"]=rsi(df["Close"])
    df["E12"]=ema(df["Close"],12)
    df["E26"]=ema(df["Close"],26)
    df["MACD"],df["MS"],df["MH"]=macd(df["Close"])
    df["SK"],df["SD"]=stoch(df)
    df["ATR"]=atr(df)
    df["BU"],df["BL"]=bb(df["Close"])
    df["BW"]=(df["BU"]-df["BL"])/df["Close"]*100
    df["CH"]=chop(df)
    df["RET"]=df["Close"].pct_change()
    df["VOL"]=df["RET"].rolling(14,1).std()
    return df.dropna()

def feats(df):
    f=pd.DataFrame(index=df.index)
    f["r"]=df["RSI"]/100
    f["m"]=np.tanh(df["MACD"]/df["Close"].std())
    f["h"]=np.tanh(df["MH"]/df["Close"].std())
    f["k"]=df["SK"]/100
    f["d"]=df["SD"]/100
    f["e12"]=(df["Close"]/df["E12"]-1)*10
    f["e26"]=(df["Close"]/df["E26"]-1)*10
    f["a"]=np.tanh(df["ATR"]/df["Close"].mean())
    f["bb"]=(df["Close"]-(df["BU"]+df["BL"])/2)/(df["BU"]-df["BL"])*2
    f["c"]=df["CH"]/100
    f["v"]=np.tanh(df["VOL"]*10)
    f["rt"]=np.tanh(df["RET"]*10)
    return f.fillna(0)

def dataset(df,f,lb=10):
    X,y=[],[];fa=f.values;c=df["Close"].values
    for i in range(lb,len(fa)-5):
        X.append(fa[i-lb:i].flatten())
        fr=(c[i+5]-c[i])/c[i]
        y.append([0,1,0] if fr>0.003 else [1,0,0] if fr<-0.003 else [0,0,1])
    return np.array(X),np.array(y)

if "nn" not in st.session_state:st.session_state.nn=NN(120,24,3,0.005)
if "done" not in st.session_state:st.session_state.done=False

c1,c2,c3=st.columns([2,3,2])
with c1:st.markdown("<h4 style='margin:0'>🧠 Exnova AI</h4>",unsafe_allow_html=True)
with c2:st.caption(f"⏱ {REFRESH_INTERVAL}s • {datetime.now().strftime('%H:%M:%S')}")
with c3:
    a=st.selectbox("",["EURUSD","GBPUSD","USDJPY","AUDUSD","BTC","ETH"],0,label_visibility="collapsed")
    am={"EURUSD":"EURUSD=X","GBPUSD":"GBPUSD=X","USDJPY":"USDJPY=X","AUDUSD":"AUDUSD=X","BTC":"BTC-USD","ETH":"ETH-USD"}
    a=am[a]
tf=st.selectbox("TF",["1m","5m","15m","1h"],1,label_visibility="collapsed",key="tf")

@st.cache_data(ttl=REFRESH_INTERVAL,show_spinner=False)
def get(t,i):
    try:
        pm={"1m":"5d","5m":"10d","15m":"30d","1h":"60d"}
        df=yf.download(t,period=pm.get(i,"10d"),interval=i,progress=False,auto_adjust=True)
        if isinstance(df.columns,pd.MultiIndex):df.columns=df.columns.get_level_values(0)
        for c in ["Open","High","Low","Close","Volume"]:
            if c not in df.columns:return pd.DataFrame()
        return df.dropna()
    except:return pd.DataFrame()

df=get(a,tf)
if df.empty or len(df)<80:st.error("❌ Sin datos");st.stop()
df=indis(df)
if len(df)<60:st.error("❌ Insuficiente");st.stop()

f=feats(df)
if not st.session_state.done:
    with st.spinner("🧠 Entrenando IA..."):
        X,y=dataset(df,f)
        if len(X)>50:st.session_state.nn.train(X,y,200);st.session_state.done=True;st.session_state.n=len(X)
        else:st.session_state.done=True;st.session_state.n=0

fa=f.values
if len(fa)>=10:
    inp=fa[-10:].flatten().reshape(1,-1)
    if inp.shape[1]<120:inp=np.pad(inp,((0,0),(0,120-inp.shape[1])),'constant',constant_values=0)
    elif inp.shape[1]>120:inp=inp[:,:120]
    p=st.session_state.nn.forward(inp)
    pc=np.argmax(p);conf=np.max(p)*100
    sig=["PUT","CALL","NEUTRAL"][pc];col=["#FF1744","#00E676","#546E7A"][pc];em=["📉","📈","➖"][pc]
else:sig="NEUTRAL";col="#546E7A";em="➖";conf=0;p=[[0,0,1]]

u=df.iloc[-1];atr=u.ATR;pa=u.Close
dec=5 if "USD=X" in a or "JPY=X" in a or "AUD" in a else 2
sp=atr*1.5 if dec==5 else atr*2;tp=atr*3 if dec==5 else atr*4
if sig=="CALL":sl=pa-sp;tpv=pa+tp
elif sig=="PUT":sl=pa+sp;tpv=pa-tp
else:sl=tpv=None

s1,s2,s3=st.columns([2,1,1])
with s1:st.markdown(f'<div class="signal-box" style="background-color:{col}"><div style="font-size:10px">IA PREDICE</div><div style="font-size:28px;margin:0">{em} {sig}</div><div style="font-size:10px">Conf {conf:.0f}%</div></div>',unsafe_allow_html=True)
with s2:st.markdown(f'<div class="strength-box" style="background-color:#D50000"><div>BAJA</div><div style="font-size:18px">{p[0][0]*100:.0f}%</div></div>',unsafe_allow_html=True)
with s3:st.markdown(f'<div class="strength-box" style="background-color:#00C853"><div>SUBE</div><div style="font-size:18px">{p[0][1]*100:.0f}%</div></div>',unsafe_allow_html=True)

ps=f"{pa:.{dec}f}";ss=f"{sl:.{dec}f}" if sl else "—";ts=f"{tpv:.{dec}f}" if tpv else "—"
m1,m2,m3,m4,m5,m6=st.columns(6)
m1.metric("Precio",ps);m2.metric("RSI",f"{u.RSI:.0f}");m3.metric("Chop",f"{u.CH:.0f}");m4.metric("ATR",f"{atr:.{dec}f}");m5.metric("SL",ss);m6.metric("TP",ts)
st.markdown("<p style='font-size:11px;margin:2px 0'>📊 Gráfico</p>",unsafe_allow_html=True)
dp=df.tail(40).copy()
fig=go.Figure()
fig.add_trace(go.Candlestick(
    x=dp.index,open=dp["Open"],high=dp["High"],low=dp["Low"],close=dp["Close"],
    increasing_line_color="#00E676",decreasing_line_color="#FF1744",
    increasing_fillcolor="rgba(0,230,118,0.2)",decreasing_fillcolor="rgba(255,23,68,0.2)",
    line=dict(width=1),name="P"
))
fig.add_trace(go.Scatter(x=dp.index,y=dp["E12"],mode="lines",line=dict(color="#FFB300",width=1.2),name="E12"))
fig.add_trace(go.Scatter(x=dp.index,y=dp["E26"],mode="lines",line=dict(color="#42A5F5",width=1.2),name="E26"))
if sig!="NEUTRAL" and sl is not None:
    fig.add_hline(y=sl,line_dash="dash",line_color="#FF1744",annotation_text="SL",annotation_position="right",annotation_font_size=9,annotation_font_color="#FF1744")
    fig.add_hline(y=tpv,line_dash="dash",line_color="#00E676",annotation_text="TP",annotation_position="right",annotation_font_size=9,annotation_font_color="#00E676")
fig.update_layout(
    height=180,margin=dict(l=0,r=0,t=5,b=0),xaxis_rangeslider_visible=False,
    template="plotly_dark",showlegend=False,paper_bgcolor="#0b0e14",plot_bgcolor="#0b0e14",
    xaxis=dict(showgrid=False,fixedrange=True,showticklabels=True,color="#888",tickfont=dict(size=8)),
    yaxis=dict(showgrid=True,gridcolor="#1f2636",fixedrange=True,color="#888",side="right",tickfont=dict(size=8)),
    font=dict(color="white",size=9)
)
st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False},key="chart")

st.markdown("<p style='font-size:11px;margin:2px 0'>📋 Indicadores</p>",unsafe_allow_html=True)
i1,i2,i3,i4,i5=st.columns(5)
inds=[
    ("RSI",f"{u.RSI:.0f}","<30|>70"),
    ("MACD",f"{u.MACD:.{dec}f}","Cruce"),
    ("Stoch",f"{u.SK:.0f}","<20|>80"),
    ("Chop",f"{u.CH:.0f}","<38|>62"),
    ("Vol",f"{u.VOL*100:.2f}%","14d")
]
for col,(n,v,d) in zip([i1,i2,i3,i4,i5],inds):
    with col:st.markdown(f'<div class="metric-card"><div style="font-size:9px;color:#888">{n}</div><div style="font-size:13px;font-weight:600">{v}</div><div style="font-size:8px;color:#555">{d}</div></div>',unsafe_allow_html=True)

st.markdown(f'<div style="background:linear-gradient(135deg,#0f2027,#203a43);border:1px solid #00d2ff;padding:6px;border-radius:6px;margin:4px 0;font-size:10px"><p style="margin:0;color:#00d2ff"><b>🧬 IA Neural:</b> 12 indicadores → 24 neuronas → CALL/PUT/NEUTRAL</p><p style="margin:0;color:#888;font-size:9px">Entrena con RSI+EMA+MACD+Stoch+BB+ATR+Chop+Vol+Retornos | Choppiness: {u.CH:.1f} | {"TENDENCIA" if u.CH<38 else "LATERAL" if u.CH>62 else "TRANSICIÓN"}</p></div>',unsafe_allow_html=True)

st.markdown('<div class="disclaimer">⚠️ IA propia desde cero con Numpy • Sin ChatGPT • Alto riesgo</div>',unsafe_allow_html=True)
