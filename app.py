import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

# ==========================================
# 1. CARGA Y SIMULACIÓN DE DATOS (MÉTODOS OPTIMIZADOS)
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

# METRICAS DEL MODELO XGBOOST (Sustentación académica)
data_metricas = pd.DataFrame({
    "Escenario": ["Antes de Imputar", "Después de IterativeImputer", "Antes de Imputar", "Después de IterativeImputer"],
    "Métrica": ["R² Score", "R² Score", "MAE (mg/L)", "MAE (mg/L)"],
    "Valor": [0.68, 0.86, 5.4, 2.1]
})

# ==========================================
# 2. CONFIGURACIÓN DE LA APP & ESTILOS
# ==========================================

# Definimos la aplicación Dash
app = dash.Dash(
    __name__, 
    external_stylesheets=[
        dbc.themes.FLATLY, 
        "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css"
    ],
    suppress_callback_exceptions=True
)

# Servidor necesario para despliegue en plataformas como Render
server = app.server

# Título de la pestaña del navegador
app.title = "H2O Analytics - Control de Calidad del Agua"

BADGE_STYLE = {
    "backgroundColor": "#00d2d3", "color": "#1a252f", "fontWeight": "bold",
    "padding": "6px 12px", "borderRadius": "20px", "display": "inline-block",
    "marginTop": "8px", "fontSize": "13px", "boxShadow": "0 4px 10px rgba(0,0,0,0.3)"
}

tabs_menu = dbc.Tabs([
    dbc.Tab(label="Contexto", tab_id="tab-contexto"),
    dbc.Tab(label="Metodología", tab_id="tab-metodologia"),
    dbc.Tab(label="EDA", tab_id="tab-eda"),
    dbc.Tab(label="Métricas", tab_id="tab-metricas"),
    dbc.Tab(label="Predicción", tab_id="tab-prediccion"),
], id="tabs-navegacion", active_tab="tab-contexto", className="nav-pills justify-content-start mb-4")

HEADER_CONTAINER_STYLE = {
    "height": "180px", "backgroundColor": "#0b0e12", "borderRadius": "12px",
    "marginTop": "15px", "marginBottom": "25px", "boxShadow": "0 4px 15px rgba(0,0,0,0.5)",
    "display": "flex", "flexDirection": "column", "justifyContent": "center",
    "alignItems": "center", "textAlign": "center"
}

app.layout = dbc.Container([
    html.Div([
        html.H1("H₂O Analytics", className="fw-bold text-white mb-2", style={"fontSize": "42px", "letterSpacing": "1px"}),
        html.P("Ciencia de Datos al Servicio de la Calidad del Agua", className="text-white-50 mb-2", style={"fontSize": "16px"}),
        html.Span("IterativeImputer + ML", style=BADGE_STYLE)
    ], style=HEADER_CONTAINER_STYLE),
    tabs_menu,
    html.Div(id="contenido-pestañas")
], fluid=True, style={"backgroundColor": "#0b0e12", "minHeight": "100vh", "color": "#e6edf3", "paddingBottom": "40px"})

# ==========================================
# 3. CALLBACK PRINCIPAL
# ==========================================
@app.callback(Output("contenido-pestañas", "children"), [Input("tabs-navegacion", "active_tab")])
def render_tab_content(active_tab):
    if active_tab == "tab-contexto":
        return html.Div([
            html.H2("Diagnóstico del Proyecto: Calidad Hídrica Industrial", className="fw-bold mb-4", style={"color": "#ffffff"}),
            html.P("Este proyecto aborda la problemática de la medición in-situ de la Demanda Bioquímica de Oxígeno (BOD). Los métodos de laboratorio tradicionales requieren hasta 5 días para generar resultados, lo que impide una gestión reactiva ante picos de contaminación.", className="text-secondary mb-4"),
            
            dbc.Row([
                dbc.Col(dbc.Card([dbc.CardBody([html.I(className="fas fa-microchip fa-2x mb-3 text-info"), html.H4("Modelo de Inferencia"), html.P("Sustitución de ensayos físicos por modelos predictivos XGBoost.")])], style={"backgroundColor": "#161b22", "border": "1px solid #30363d", "color": "white"}), md=4),
                dbc.Col(dbc.Card([dbc.CardBody([html.I(className="fas fa-database fa-2x mb-3 text-warning"), html.H4("Integridad de Datos"), html.P("Uso de MICE para imputar valores faltantes sin perder varianza.")])], style={"backgroundColor": "#161b22", "border": "1px solid #30363d", "color": "white"}), md=4),
                dbc.Col(dbc.Card([dbc.CardBody([html.I(className="fas fa-shield-alt fa-2x mb-3 text-success"), html.H4("Impacto Operativo"), html.P("Reducción de tiempos de decisión de 120 horas a tiempo real.")])], style={"backgroundColor": "#161b22", "border": "1px solid #30363d", "color": "white"}), md=4),
            ], className="mb-4"),
            
            html.Div([
                html.H5("🌱 El Problema de Negocio", className="text-white"),
                html.P("El dataset (train/test) presenta variables altamente correlacionadas (pH, OD, Conductividad) con un 25% de datos faltantes. La metodología implementada permite que, ante la ausencia de un sensor específico, el sistema pueda estimar el estado de la cuenca con alta confianza estadística.")
            ], className="p-4 border border-secondary rounded mt-3")
        ])
    
    elif active_tab == "tab-metodologia":
        return html.Div([
            html.H2("Metodología: Validación y Selección de Modelo", className="fw-bold mb-4", style={"color": "#ffffff"}),
            dbc.Row([
                dbc.Col(html.Div([
                    html.H5("🧪 Pruebas Realizadas", className="text-info"),
                    html.Ul([
                        html.Li("Regresión Lineal Simple: Establecida como baseline para medir la mejora base."),
                        html.Li("Regresión Lasso/Ridge: Aplicadas para penalizar coeficientes y evitar sobreajuste (overfitting)."),
                        html.Li("XGBoost (Modelo Final): Prueba de ensamble mediante Gradient Boosting."),
                    ], className="text-white"),
                    html.H5("¿Por qué XGBoost fue el ganador?", className="text-info mt-3"),
                    html.P("XGBoost superó a los modelos lineales al capturar dependencias no lineales entre el BOD y las variables físico-químicas. Mientras que la Regresión Lineal obtuvo un R² de 0.68, el modelo XGBoost alcanzó 0.86, demostrando que la contaminación orgánica en esta cuenca no tiene una relación constante, sino dependiente de umbrales críticos de pH y OD.", className="text-white")
                ], className="p-4 border border-secondary rounded", style={"backgroundColor": "#161b22"}), md=6),
                
                dbc.Col(html.Div([
                    html.H5("📋 Estado de Calidad de las Variables", className="text-info mb-3"),
                    dash_table.DataTable(
                        data=data_agua.to_dict('records'), 
                        style_cell={'backgroundColor': '#161b22', 'color': 'white', 'textAlign': 'center'},
                        style_header={'backgroundColor': '#0b0e12', 'color': '#00d2d3', 'fontWeight': 'bold'}
                    )
                ], className="p-4 border border-secondary rounded", style={"backgroundColor": "#161b22"}), md=6)
            ])
        ])

    elif active_tab == "tab-eda":
        return html.Div([
            dcc.Dropdown(id="dropdown-variable-eda", options=[{"label": var, "value": var} for var in data_historica["Variable"].unique()], value="BOD (Demanda Bioquímica)", style={"color": "#000"}),
            dcc.Graph(id="grafico-dinamico-eda")
        ])

    elif active_tab == "tab-metricas":
        fig = px.bar(data_metricas, x="Escenario", y="Valor", color="Métrica", template="plotly_dark")
        return html.Div([dcc.Graph(figure=fig), html.P("El R² de 0.86 valida que el modelo explica la mayor parte de la contaminación orgánica.", className="text-white")])

    elif active_tab == "tab-prediccion":
        return html.Div([
            html.H2("Módulo de Predicción en Tiempo Real", className="fw-bold mb-4", style={"color": "#f5f4f2"}),
            dbc.Row([
                dbc.Col(html.Div([
                    html.H5("🎛️ Variables de Entrada", className="fw-bold mb-3", style={"color": "#f5f4f2"}),
                    html.P("Ajuste los valores medidos en campo para calcular la demanda biológica proyectada:", style={"fontSize": "13px", "color": "#f5f4f2"}),
                    html.Label("Nivel de pH:", className="fw-bold mb-1", style={"fontSize": "13px", "color": "#f5f4f2"}),
                    dbc.Input(id="input-ph", type="number", value=7.4, min=0, max=14, step=0.1, className="mb-3", style={"borderRadius": "8px"}),
                    html.Label("Nitrógeno Amonio (mg/L):", className="fw-bold mb-1", style={"fontSize": "13px", "color": "#f5f4f2"}),
                    dbc.Input(id="input-nitrogeno", type="number", value=2.5, min=0, step=0.1, className="mb-3", style={"borderRadius": "8px"}),
                    html.Label("Oxígeno Disuelto (mg/L):", className="fw-bold mb-1", style={"fontSize": "13px", "color": "#f5f4f2"}),                    
                    dbc.Input(id="input-oxigeno", type="number", value=5.0, min=0, step=0.1, className="mb-2", style={"borderRadius": "8px"}),
                    html.Label("Fósforo Total (mg/L):", className="fw-bold mb-1", style={"fontSize": "13px", "color": "#f5f4f2"}),
                    dbc.Input(id="input-fosforo", type="number", value=0.5, min=0, step=0.1, className="mb-3", style={"borderRadius": "8px"}),
                ], style={"backgroundColor": "black", "padding": "25px", "borderRadius": "15px", "boxShadow": "0 4px 6px rgba(0,0,0,0.15)"}), md=5, className="mb-3"),
                dbc.Col(html.Div([
                    html.H5("🔮 Estimación Proyectada del Target", className="fw-bold mb-4", style={"color": "#f5f4f2"}),
                    html.Div([
                        html.H6("BOD (DEMANDA BIOQUÍMICA DE OXÍGENO) CALCULADA:", className="text-muted fw-bold mb-2", style={"fontSize": "12px", "letterSpacing": "1px"}),
                        html.Div(id="resultado-prediccion-bod", className="fw-bold mb-2", style={"fontSize": "56px", "letterSpacing": "1px"}),
                        html.Div(id="status-badge-bod")
                    ], style={"padding": "40px 30px", "borderRadius": "15px", "textAlign": "center", "backgroundColor": "#f5f4f2", "border": "1px solid #e9ecef"}),
                    html.P(
                        ["⚠️ ", html.Strong("Nota de Ingeniería:"), " Un valor de BOD superior a 50 mg/L se considera un indicador de alta carga orgánica residual."], 
                        className="mt-3", # Quitamos la clase text-muted
                        style={"fontSize": "12px", "color": "#f5f4f2"} # Tu color original ahora sí funcionará
                    )
                ], style={"backgroundColor": "black", "padding": "25px", "borderRadius": "15px", "boxShadow": "0 4px 6px rgba(0,0,0,0.15)", "height": "100%"}), md=7, className="mb-3")
            ])
        ], className="p-2")

# ==========================================
# 4. CALLBACKS
# ==========================================
@app.callback(Output("grafico-dinamico-eda", "figure"), [Input("dropdown-variable-eda", "value")])
def update_eda(var):
    df_f = data_historica[data_historica["Variable"] == var]
    return px.line(df_f, x="Mes", y="Valor", markers=True, template="plotly_dark")

@app.callback(Output("out-bod", "children"), [Input("ph-in", "value")])
def predict(ph):
    return f"Predicción: {round(ph * 5.2, 2)} mg/L" if ph else "Esperando entrada..."

@app.callback(
    [Output("resultado-prediccion-bod", "children"),
     Output("resultado-prediccion-bod", "style"),
     Output("status-badge-bod", "children")],
    [Input("input-ph", "value"),
     Input("input-nitrogeno", "value"),     
     Input("input-oxigeno", "value"),
     Input("input-fosforo", "value")]
)

def calcular_prediccion_h2o(ph, nitrogeno, oxigeno, fosforo):
    if ph is None or nitrogeno is None or oxigeno is None or fosforo is None:
        return "---", {"color": "#f5f4f2"}, dbc.Badge("Esperando datos", color="secondary")
    
    bod_estimado = (nitrogeno * 15.4) + (ph * 3.2) - (oxigeno * 1.5) - (fosforo * 8.7) + 12.3
    bod_estimado = max(0, round(bod_estimado, 2))
    
    if bod_estimado < 40:
        color_texto = "#2ecc71"
        badge = dbc.Badge("Calidad Estable (Baja Carga Orgánica)", color="success", style={"padding": "8px 12px", "borderRadius": "10px"})
    elif 40 <= bod_estimado <= 60:
        color_texto = "#f39c12"
        badge = dbc.Badge("Alerta: Carga Orgánica Moderada", color="warning", style={"padding": "8px 12px", "borderRadius": "10px", "color": "white"})
    else:
        color_texto = "#e74c3c"
        badge = dbc.Badge("CRÍTICO: Alta Contaminación Detectada", color="danger", style={"padding": "8px 12px", "borderRadius": "10px"})
        
    return f"{bod_estimado} mg/L", {"color": color_texto, "fontSize": "56px"}, badge


if __name__ == "__main__":
    app.run(debug=True, port=8050)