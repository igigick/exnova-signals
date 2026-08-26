import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
.block-container{padding:0.2rem 0.5rem 0rem 0.5rem;max-width:100%}
.mini-box{padding:3px 2px;border-radius:6px;text-align:center;color:white;font-weight:700;font-size:10px;margin:1px}
.metric-mini{background-color:#151a25;padding:2px 1px;border-radius:4px;text-align:center;border:1px solid #1f2636;font-size:9px}
.disclaimer{font-size:8px;color:#555;text-align:center;padding:1px;margin-top:2px}
h1,h2,h3,h4,h5,h6,p{margin:0!important;padding:0!important}
.stSelectbox{margin-bottom:-12px!important}
.stSelectbox label{font-size:10px!important;margin-bottom:0!important}
iframe{height:200px!important}
</style>
""", unsafe_allow_html=True)

class NN:
    def __init__(self,inp,hid,out,lr=0.05):
        np.random.seed(42)
        self.lr=lr
        self.W1=np.random.randn(inp,hid)*0.1
        self.b1=np.zeros((1,hid))
        self.W2=np.random.randn(hid,out)*0.1
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
    def train(self,X,y,epochs=30):
        y=np.array(y)
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
    rs=ag/al;rs=rs.replace([np.inf,-np.inf],1).fillna(1)
    return 100-(100/(1+rs))
def ema(s,p):return s.ewm(span=p,adjust=False).mean()
def macd(s):
    m=ema(s,12)-ema(s,26)
    return m,ema(m,9),m-ema(m,9)
def atr(df):
    tr=pd.concat([df["High"]-df["Low"],(df["High"]-df["Close"].shift()).abs(),(df["Low"]-df["Close"].shift()).abs()],axis=1).max(axis=1)
    return tr.rolling(14,1).mean()
def bb(s):
    m=s.rolling(20,1).mean();sd=s.rolling(20,1).std().replace(0,0.001)
    return m+2*sd,m-2*sd
def stoch(df):
    ll=df["Low"].rolling(14,1).min();hh=df["High"].rolling(14,1).max()
    r=(hh-ll).replace(0,0.001)
    k=100*(df["Close"]-ll)/r
    return k,k.rolling(3,1).mean()
def chop(df):
    tr=pd.concat([df["High"]-df["Low"],(df["High"]-df["Close"].shift()).abs(),(df["Low"]-df["Close"].shift()).abs()],axis=1).max(axis=1)
    a=tr.rolling(14,1).sum();mx=df["High"].rolling(14,1).max();mn=df["Low"].rolling(14,1).min()
    r=(mx-mn).replace(0,0.001)
    return (100*np.log10(a/r)/np.log10(14)).fillna(50)

def indis(df):
    df=df.copy()
    df["RSI"]=rsi(df["Close"]);df["E12"]=ema(df["Close"],12);df["E26"]=ema(df["Close"],26)
    df["MACD"],df["MS"],df["MH"]=macd(df["Close"]);df["SK"],df["SD"]=stoch(df)
    df["ATR"]=atr(df);df["BU"],df["BL"]=bb(df["Close"])
    df["BW"]=(df["BU"]-df["BL"])/df["Close"].replace(0,0.001)*100;df["CH"]=chop(df)
    df["RET"]=df["Close"].pct_change();df["VOL"]=df["RET"].rolling(14,1).std()
    return df.dropna()

def feats(df):
    f=pd.DataFrame(index=df.index)
    cs=df["Close"].std();cs=cs if cs>0 else 0.001
    cm=df["Close"].mean();cm=cm if cm>0 else 0.001
    f["r"]=df["RSI"]/100
    f["m"]=np.tanh(df["MACD"]/cs)
    f["h"]=np.tanh(df["MH"]/cs)
    f["k"]=df["SK"]/100
    f["d"]=df["SD"]/100
    f["e12"]=(df["Close"]/df["E12"].replace(0,0.001)-1)*10
    f["e26"]=(df["Close"]/df["E26"].replace(0,0.001)-1)*10
    f["a"]=np.tanh(df["ATR"]/cm)
    bw=(df["BU"]-df["BL"]).replace(0,0.001)
    f["bb"]=(df["Close"]-(df["BU"]+df["BL"])/2)/bw*2
    f["c"]=df["CH"]/100
    f["v"]=np.tanh(df["VOL"]*10)
    f["rt"]=np.tanh(df["RET"]*10)
    return f.fillna(0).replace([np.inf,-np.inf],0)

def dataset(df,f,lb=10):
    X,y=[],[];fa=f.values;c=df["Close"].values
    for i in range(lb,len(fa)-5):
        window=fa[i-lb:i].flatten()
        if len(window)==120 and not np.isnan(window).any() and not np.isinf(window).any():
            X.append(window)
            fr=(c[i+5]-c[i])/c[i]
            y.append([0,1,0] if fr>0.003 else [1,0,0] if fr<-0.003 else [0,0,1])
    return np.array(X),np.array(y)

if "nn" not in st.session_state:st.session_state.nn=NN(120,16,3,0.05)
if "done" not in st.session_state:st.session_state.done=False
if "fb" not in st.session_state:st.session_state.fb=False

c1,c2,c3=st.columns([2,3,2])
with c1:st.markdown("<h5 style='margin:0'>🧠 Exnova AI</h5>",unsafe_allow_html=True)
with c2:st.caption(f"⏱{REFRESH_INTERVAL}s • {datetime.now().strftime('%H:%M:%S')}")
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
    st.info("🧠 Entrenando IA... (3 segundos)")
    try:
        X,y=dataset(df,f)
        if len(X)>20:
            st.session_state.nn.train(X,y,30)
            st.session_state.done=True;st.session_state.n=len(X);st.session_state.fb=False
        else:
            st.session_state.done=True;st.session_state.n=0;st.session_state.fb=True
    except:
        st.session_state.done=True;st.session_state.fb=True

fa=f.values
if len(fa)>=10:
    inp=fa[-10:].flatten().reshape(1,-1)
    if inp.shape[1]<120:inp=np.pad(inp,((0,0),(0,120-inp.shape[1])),'constant',constant_values=0)
    elif inp.shape[1]>120:inp=inp[:,:120]
    if not st.session_state.fb:
        try:
            p=st.session_state.nn.forward(inp)
            pc=np.argmax(p);conf=np.max(p)*100
        except:
            p=[[0.33,0.33,0.34]];pc=2;conf=34;st.session_state.fb=True
    else:
        p=[[0.33,0.33,0.34]];pc=2;conf=34
else:
    p=[[0.33,0.33,0.34]];pc=2;conf=34

sig=["PUT","CALL","NEUTRAL"][pc];col=["#FF1744","#00E676","#546E7A"][pc];em=["📉","📈","➖"][pc]
put_pct=p[0][0]*100;call_pct=p[0][1]*100;neu_pct=p[0][2]*100
u=df.iloc[-1];atr=u.ATR;pa=u.Close
dec=5 if "USD=X" in a or "JPY=X" in a or "AUD" in a else 2
sp=atr*1.5 if dec==5 else atr*2;tp=atr*3 if dec==5 else atr*4
if sig=="CALL":sl=pa-sp;tpv=pa+tp
elif sig=="PUT":sl=pa+sp;tpv=pa-tp
else:sl=tpv=None

b1,b2,b3=st.columns(3)
with b1:st.markdown(f'<div class="mini-box" style="background-color:#D50000"><div>PUT</div><div style="font-size:16px">{put_pct:.0f}%</div></div>',unsafe_allow_html=True)
with b2:st.markdown(f'<div class="mini-box" style="background-color:{col}"><div>IA → {sig}</div><div style="font-size:20px">{em}</div><div style="font-size:9px">{conf:.0f}% conf</div></div>',unsafe_allow_html=True)
with b3:st.markdown(f'<div class="mini-box" style="background-color:#00C853"><div>CALL</div><div style="font-size:16px">{call_pct:.0f}%</div></div>',unsafe_allow_html=True)

ps=f"{pa:.{dec}f}";ss=f"{sl:.{dec}f}" if sl else "—";ts=f"{tpv:.{dec}f}" if tpv else "—"
st.markdown(f"""
<div style="display:flex;justify-content:space-between;margin-top:4px">
<div class="metric-mini" style="flex:1;margin:0 1px"><div style="font-size:8px;color:#888">Precio</div><div style="font-size:11px;font-weight:600">{ps}</div></div>
<div class="metric-mini" style="flex:1;margin:0 1px"><div style="font-size:8px;color:#888">RSI</div><div style="font-size:11px;font-weight:600">{u.RSI:.0f}</div></div>
<div class="metric-mini" style="flex:1;margin:0 1px"><div style="font-size:8px;color:#888">Chop</div><div style="font-size:11px;font-weight:600">{u.CH:.0f}</div></div>
<div class="metric-mini" style="flex:1;margin:0 1px"><div style="font-size:8px;color:#888">ATR</div><div style="font-size:11px;font-weight:600">{atr:.{dec}f}</div></div>
<div class="metric-mini" style="flex:1;margin:0 1px"><div style="font-size:8px;color:#888">SL</div><div style="font-size:11px;font-weight:600">{ss}</div></div>
<div class="metric-mini" style="flex:1;margin:0 1px"><div style="font-size:8px;color:#888">TP</div><div style="font-size:11px;font-weight:600">{ts}</div></div>
</div>
""",unsafe_allow_html=True)
st.markdown("<p style='font-size:10px;margin:2px 0'>📊 Gráfico</p>",unsafe_allow_html=True)
dp=df.tail(50).copy()
fig=make_subplots(rows=2,cols=1,shared_xaxes=True,vertical_spacing=0.02,row_heights=[0.75,0.25])
fig.add_trace(go.Candlestick(
    x=dp.index,open=dp["Open"],high=dp["High"],low=dp["Low"],close=dp["Close"],
    increasing_line_color="#00E676",decreasing_line_color="#FF1744",
    increasing_fillcolor="#00E676",decreasing_fillcolor="#FF1744",
    line=dict(width=1),name="P"
),row=1,col=1)
fig.add_trace(go.Scatter(x=dp.index,y=dp["E12"],mode="lines",line=dict(color="#FFB300",width=1.2),name="E12"),row=1,col=1)
fig.add_trace(go.Scatter(x=dp.index,y=dp["E26"],mode="lines",line=dict(color="#42A5F5",width=1.2),name="E26"),row=1,col=1)
fig.add_trace(go.Scatter(x=dp.index,y=dp["BU"],mode="lines",line=dict(color="rgba(255,255,255,0.2)",width=1),showlegend=False),row=1,col=1)
fig.add_trace(go.Scatter(x=dp.index,y=dp["BL"],mode="lines",line=dict(color="rgba(255,255,255,0.2)",width=1),fill="tonexty",fillcolor="rgba(255,255,255,0.05)",showlegend=False),row=1,col=1)
if sig!="NEUTRAL" and sl is not None:
    fig.add_hline(y=sl,line_dash="dash",line_color="#FF1744",annotation_text="SL",annotation_position="right",annotation_font_size=8,annotation_font_color="#FF1744",row=1,col=1)
    fig.add_hline(y=tpv,line_dash="dash",line_color="#00E676",annotation_text="TP",annotation_position="right",annotation_font_size=8,annotation_font_color="#00E676",row=1,col=1)
vc=["#FF1744" if dp["Close"].iloc[i]<dp["Open"].iloc[i] else "#00E676" for i in range(len(dp))]
fig.add_trace(go.Bar(x=dp.index,y=dp["Volume"],marker_color=vc,showlegend=False),row=2,col=1)
fig.update_layout(
    height=200,margin=dict(l=0,r=0,t=2,b=0),xaxis_rangeslider_visible=False,
    template="plotly_dark",paper_bgcolor="#0b0e14",plot_bgcolor="#0b0e14",showlegend=False,
    xaxis=dict(showgrid=False,fixedrange=True,showticklabels=True,color="#888",tickfont=dict(size=8)),
    yaxis=dict(showgrid=True,gridcolor="#1f2636",fixedrange=True,color="#888",side="right",tickfont=dict(size=8)),
    yaxis2=dict(showgrid=False,fixedrange=True,color="#888",side="right",tickfont=dict(size=8)),
    font=dict(color="white",size=9)
)
st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False},key="chart")

st.markdown("<p style='font-size:10px;margin:2px 0'>📋 Indicadores</p>",unsafe_allow_html=True)
i1,i2,i3,i4,i5=st.columns(5)
inds=[("RSI",f"{u.RSI:.0f}","<30|>70"),("MACD",f"{u.MACD:.{dec}f}","Cruce"),("Stoch",f"{u.SK:.0f}","<20|>80"),("Chop",f"{u.CH:.0f}","<38|>62"),("Vol",f"{u.VOL*100:.2f}%","14d")]
for col,(n,v,d) in zip([i1,i2,i3,i4,i5],inds):
    with col:st.markdown(f'<div class="metric-mini"><div style="font-size:8px;color:#888">{n}</div><div style="font-size:12px;font-weight:600">{v}</div><div style="font-size:7px;color:#555">{d}</div></div>',unsafe_allow_html=True)

fb="⚠️ Fallback" if st.session_state.fb else "✅ IA lista"
st.markdown(f'<div style="background:linear-gradient(135deg,#0f2027,#203a43);border:1px solid #00d2ff;padding:4px;border-radius:4px;margin:2px 0;font-size:9px"><p style="margin:0;color:#00d2ff"><b>🧬 {fb}:</b> PUT {put_pct:.1f}% | CALL {call_pct:.1f}% | NEUTRAL {neu_pct:.1f}% | Chop {u.CH:.0f}</p></div>',unsafe_allow_html=True)
st.markdown('<div class="disclaimer">⚠️ IA propia desde cero con Numpy • Alto riesgo</div>',unsafe_allow_html=True)
