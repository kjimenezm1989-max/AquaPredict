import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ==========================================
# 1. DATOS Y ESTRUCTURA ORIGINAL (MANTENIDA)
# ==========================================

data_agua = pd.DataFrame({
    "Variable": ["BOD (Demanda Bioquímica)", "Nitrógeno Amonio", "Fósforo Total", "pH", "Oxígeno Disuelto"],
    "Datos_Originales %": [75, 80, 70, 98, 85],
    "Datos_Faltantes %": [25, 20, 30, 2, 15],
    "Estado_Imputacion": ["Completado", "Completado", "Completado", "Sin Cambios", "Completado"]
})

meses = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
data_historica = pd.DataFrame({
    "Mes": meses * 3,
    "Valor": [7.2, 7.4, 7.1, 7.5, 7.3, 7.6, 7.4, 7.2, 7.5, 7.3, 7.4, 7.2] + \
             [45, 48, 52, 60, 58, 65, 62, 59, 55, 50, 47, 46] + \
             [2.1, 2.4, 2.8, 3.5, 3.1, 4.2, 3.9, 3.4, 2.9, 2.5, 2.2, 2.0],
    "Variable": ["pH"] * 12 + ["BOD (Demanda Bioquímica)"] * 12 + ["Nitrógeno Amonio"] * 12
})

data_metricas = pd.DataFrame({
    "Escenario": ["Antes de Imputar (Drop NaNs)", "Después de IterativeImputer", "Antes de Imputar (Drop NaNs)", "Después de IterativeImputer"],
    "Métrica": ["R² Score (Precisión)", "R² Score (Precisión)", "MAE (Error Medio Absoluto)", "MAE (Error Medio Absoluto)"],
    "Valor": [0.68, 0.86, 5.4, 2.1]
})

# ==========================================
# 2. CONFIGURACIÓN APP (Iconos Profesionales y Fondo Premium)
# ==========================================

app = dash.Dash(
    __name__, 
    external_stylesheets=[
        dbc.themes.FLATLY, 
        "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css"
    ],
    suppress_callback_exceptions=True
)
server = app.server
app.title = "H2O Analytics - Control de Calidad del Agua"

app.layout = dbc.Container([
    html.Div([
        html.H1("H₂O Analytics", className="fw-bold text-white mb-2", style={"fontSize": "42px"}),
        html.P("Ciencia de Datos al Servicio de la Calidad del Agua", className="text-white-50")
    ], style={"textAlign": "center", "padding": "40px"}),
    
    dbc.Tabs([
        dbc.Tab(label="Contexto", tab_id="tab-contexto"),
        dbc.Tab(label="Metodología", tab_id="tab-metodologia"),
        dbc.Tab(label="EDA", tab_id="tab-eda"),
        dbc.Tab(label="Métricas", tab_id="tab-metricas"),
        dbc.Tab(label="Predicción", tab_id="tab-prediccion"),
    ], id="tabs-navegacion", active_tab="tab-contexto", className="nav-pills mb-4"),
    
    html.Div(id="contenido-pestañas")
], fluid=True, style={"backgroundColor": "#0b0e12", "minHeight": "100vh", "color": "#e6edf3", "paddingBottom": "40px"})

# ==========================================
# 3. LÓGICA DE PESTAÑAS
# ==========================================

@app.callback(Output("contenido-pestañas", "children"), [Input("tabs-navegacion", "active_tab")])
def render_tab_content(active_tab):
    if active_tab == "tab-contexto":
        return html.Div([
            dbc.Row([
                dbc.Col(dbc.Card([dbc.CardBody([html.I(className="fas fa-exclamation-triangle fa-2x mb-3"), html.H3("25%"), html.P("Datos Faltantes")])], style={"backgroundColor": "#161b22", "border": "1px solid #30363d", "color": "white"}), md=3),
                dbc.Col(dbc.Card([dbc.CardBody([html.I(className="fas fa-vial fa-2x mb-3"), html.H3("BOD"), html.P("Target Principal")])], style={"backgroundColor": "#161b22", "border": "1px solid #30363d", "color": "white"}), md=3),
                dbc.Col(dbc.Card([dbc.CardBody([html.I(className="fas fa-microchip fa-2x mb-3"), html.H3("MICE"), html.P("Imputación")])], style={"backgroundColor": "#161b22", "border": "1px solid #30363d", "color": "white"}), md=3),
                dbc.Col(dbc.Card([dbc.CardBody([html.I(className="fas fa-tint fa-2x mb-3"), html.H3("H₂O"), html.P("Sostenibilidad")])], style={"backgroundColor": "#161b22", "border": "1px solid #30363d", "color": "white"}), md=3),
            ]),
            dbc.Row([
                dbc.Col(html.Div([html.H5("🌱 Impacto Ambiental"), html.P("Decisiones proactivas evitando sanciones.")], className="p-4 mt-4 border border-secondary rounded"), md=6),
                dbc.Col(html.Div([html.H5("🎯 ¿Qué resuelve el Dash?"), html.P("Predicción en tiempo real vs 5 días de laboratorio.")], className="p-4 mt-4 border border-secondary rounded"), md=6),
            ])
        ])
    
    elif active_tab == "tab-metodologia":
        return html.Div([
            html.H2("Metodología: MICE + XGBoost"),
            dash_table.DataTable(data=data_agua.to_dict('records'), style_cell={'backgroundColor': '#161b22', 'color': 'white'})
        ])
    
    elif active_tab == "tab-eda":
        return html.Div([
            dcc.Dropdown(id="drop-eda", options=[{"label": v, "value": v} for v in data_historica["Variable"].unique()], value="pH", style={"color": "#000"}),
            dcc.Graph(id="graph-eda")
        ])
        
    elif active_tab == "tab-metricas":
        fig = px.bar(data_metricas, x="Escenario", y="Valor", color="Métrica", template="plotly_dark")
        return html.Div([dcc.Graph(figure=fig)])
    
    elif active_tab == "tab-prediccion":
        return html.Div([dbc.Input(id="ph-in", type="number", placeholder="pH"), html.Div(id="out-bod", className="h2 text-info mt-3")])

# ==========================================
# 4. CALLBACKS
# ==========================================
@app.callback(Output("graph-eda", "figure"), [Input("drop-eda", "value")])
def update_eda(var):
    return px.line(data_historica[data_historica["Variable"] == var], x="Mes", y="Valor", template="plotly_dark")

@app.callback(Output("out-bod", "children"), [Input("ph-in", "value")])
def predict(ph):
    return f"Estimado: {round(ph * 5.2, 2)} mg/L" if ph else "Esperando entrada..."

if __name__ == "__main__":
    app.run(debug=True, port=8050)