# Gráfico ligero y fluido
st.markdown("##### Gráfico")

# Solo últimos 40 datos para que vaya fluido
df_plot = df.tail(40)

fig = go.Figure()

# Velas más simples
fig.add_trace(go.Candlestick(
    x=df_plot.index,
    open=df_plot['Open'],
    high=df_plot['High'],
    low=df_plot['Low'],
    close=df_plot['Close'],
    increasing_line_color='#00E676',
    decreasing_line_color='#FF1744',
    increasing_fillcolor='#00E676',
    decreasing_fillcolor='#FF1744',
    name="Precio"
))

# EMAs
fig.add_trace(go.Scatter(
    x=df_plot.index, y=df_plot['EMA9'],
    line=dict(color='#FFB300', width=1.5),
    name="EMA9"
))
fig.add_trace(go.Scatter(
    x=df_plot.index, y=df_plot['EMA21'],
    line=dict(color='#42A5F5', width=1.5),
    name="EMA21"
))

fig.update_layout(
    height=280,
    margin=dict(l=0, r=0, t=10, b=10),
    xaxis_rangeslider_visible=False,
    template="plotly_dark",
    showlegend=False,
    paper_bgcolor="#0b0e14",
    plot_bgcolor="#0b0e14",
    xaxis=dict(showgrid=False),
    yaxis=dict(showgrid=True, gridcolor="#1e1e1e"),
    font=dict(color="white", size=11)
)

# Configuración para mejor rendimiento en móvil
st.plotly_chart(
    fig, 
    use_container_width=True, 
    config={
        "displayModeBar": False,
        "staticPlot": False,
        "scrollZoom": False
    }
)

st.caption("Herramienta educativa • No es consejo financiero • Alto riesgo de pérdida")