import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.express as px

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

data_metricas = pd.DataFrame({
    "Escenario": ["Antes de Imputar (Drop NaNs)", "Después de IterativeImputer", "Antes de Imputar (Drop NaNs)", "Después de IterativeImputer"],
    "Métrica": ["R² Score (Precisión)", "R² Score (Precisión)", "MAE (Error Medio Absoluto)", "MAE (Error Medio Absoluto)"],
    "Valor": [0.68, 0.86, 5.4, 2.1]
})

# ==========================================
# 2. CONFIGURACIÓN DE LA APP & ESTILOS DASH
# ==========================================

app = dash.Dash(
    __name__, 
    external_stylesheets=[dbc.themes.FLATLY],
    suppress_callback_exceptions=True
)
server = app.server

app.title = "H2O Analytics - Control de Calidad del Agua"

BADGE_STYLE = {
    "backgroundColor": "#00d2d3",
    "color": "#1a252f",
    "fontWeight": "bold",
    "padding": "6px 12px",
    "borderRadius": "20px",
    "display": "inline-block",
    "marginTop": "8px",
    "fontSize": "13px",
    "boxShadow": "0 4px 10px rgba(0,0,0,0.3)"
}

tabs_menu = dbc.Tabs(
    [
        dbc.Tab(label="Contexto", tab_id="tab-contexto"),
        dbc.Tab(label="Metodología", tab_id="tab-metodologia"),
        dbc.Tab(label="EDA", tab_id="tab-eda"),
        dbc.Tab(label="Métricas", tab_id="tab-metricas"),
        dbc.Tab(label="Predicción", tab_id="tab-prediccion"),
    ],
    id="tabs-navegacion",
    active_tab="tab-contexto",
    className="nav-pills justify-content-start mb-4",
)

# Estilo para el contenedor del encabezado (Centrado Absoluto y Minimalista)
HEADER_CONTAINER_STYLE = {
    "height": "180px",
    "backgroundColor": "#0b0e12",  # Fondo negro mate premium
    "borderRadius": "12px",
    "marginTop": "15px",
    "marginBottom": "25px",
    "boxShadow": "0 4px 15px rgba(0,0,0,0.5)",
    "display": "flex",
    "flexDirection": "column",
    "justifyContent": "center",  # Centrado vertical
    "alignItems": "center",      # Centrado horizontal
    "textAlign": "center"        # Alineación del texto
}

app.layout = dbc.Container(
    [
        # Encabezado limpio y perfectamente centrado
        html.Div(
            [
                html.H1("H₂O Analytics", className="fw-bold text-white mb-2", style={"fontSize": "42px", "letterSpacing": "1px"}),
                html.P("Ciencia de Datos al Servicio de la Calidad del Agua", className="text-white-50 mb-2", style={"fontSize": "16px"}),
                html.Span("IterativeImputer + ML", style=BADGE_STYLE)
            ],
            style=HEADER_CONTAINER_STYLE
        ),
        
        tabs_menu,
        html.Div(id="contenido-pestañas")
    ],
    fluid=True,
    style={
        "backgroundColor": "#0b0e12",  # Mantiene la consistencia oscura en toda la app
        "minHeight": "100vh", 
        "color": "#e6edf3",
        "paddingBottom": "40px"
    }
)

# ==========================================
# 3. CALLBACK PRINCIPAL: ENRUTAMIENTO DE PESTAÑAS
# ==========================================

@app.callback(
    Output("contenido-pestañas", "children"),
    [Input("tabs-navegacion", "active_tab")]
)
def render_tab_content(active_tab):
    if active_tab == "tab-contexto":
        return html.Div([
            html.H2("¿Por qué importa el monitoreo de la calidad del agua?", className="fw-bold mb-4", style={"color": "#ffffff"}),
            dbc.Row([
                dbc.Col(dbc.Card([dbc.CardBody([html.Div("⚠️", style={"fontSize": "30px", "marginBottom": "10px"}), html.H3("25%", className="fw-bold text-white mb-1"), html.P("Datos Faltantes", className="fw-bold text-white mb-1", style={"fontSize": "14px"}), html.P("Registros incompletos en variables críticas.", style={"fontSize": "12px", "opacity": "0.9"})])], style={"backgroundColor": "#ff6b6b", "color": "white", "borderRadius": "15px", "border": "none", "height": "100%"}), md=3, className="mb-3"),
                dbc.Col(dbc.Card([dbc.CardBody([html.Div("🧪", style={"fontSize": "30px", "marginBottom": "10px"}), html.H3("BOD", className="fw-bold text-white mb-1"), html.P("Demanda Bioquímica", className="fw-bold text-white mb-1", style={"fontSize": "14px"}), html.P("Target principal del análisis de contaminación.", style={"fontSize": "12px", "opacity": "0.9"})])], style={"backgroundColor": "#ff9f43", "color": "white", "borderRadius": "15px", "border": "none", "height": "100%"}), md=3, className="mb-3"),
                dbc.Col(dbc.Card([dbc.CardBody([html.Div("🤖", style={"fontSize": "30px", "marginBottom": "10px"}), html.H3("Imputer", className="fw-bold text-white mb-1"), html.P("IterativeImputer", className="fw-bold text-white mb-1", style={"fontSize": "14px"}), html.P("Algoritmo seleccionado para rescatar datos.", style={"fontSize": "12px", "opacity": "0.9"})])], style={"backgroundColor": "#54a0ff", "color": "white", "borderRadius": "15px", "border": "none", "height": "100%"}), md=3, className="mb-3"),
                dbc.Col(dbc.Card([dbc.CardBody([html.Div("💧", style={"fontSize": "30px", "marginBottom": "10px"}), html.H3("H₂O", className="fw-bold text-white mb-1"), html.P("Sostenibilidad", className="fw-bold text-white mb-1", style={"fontSize": "14px"}), html.P("Modelado orientado a la estabilidad hídrica.", style={"fontSize": "12px", "opacity": "0.9"})])], style={"backgroundColor": "#a55eea", "color": "white", "borderRadius": "15px", "border": "none", "height": "100%"}), md=3, className="mb-3"),
            ], className="mb-4"),
            dbc.Row([
                dbc.Col(html.Div([html.H5("🌱 Impacto Ambiental y Operativo", className="fw-bold mb-3", style={"color": "#1a252f"}), html.P("La falta de continuidad en las mediciones de parámetros hídricos como el BOD o el Nitrógeno Amonio impide que las plantas de tratamiento y las entidades de control tomen decisiones proactivas. Esto puede derivar en sanciones legales, daños ecológicos severos e ineficiencia en la dosificación de químicos.", style={"fontSize": "14px", "color": "#57606f"})], style={"backgroundColor": "white", "padding": "25px", "borderRadius": "15px", "boxShadow": "0 4px 6px rgba(0,0,0,0.15)", "height": "100%"}), md=6, className="mb-3"),
                dbc.Col(html.Div([html.H5("🎯 ¿Qué resuelve este Dashboard?", className="fw-bold mb-3", style={"color": "#1a252f"}), html.P("Este sistema transforma datos crudos e incompletos en una herramienta de analítica avanzada. A través de técnicas de Imputación Multivariada y modelos de Machine Learning, el dashboard permite visualizar tendencias históricas limpias y predecir niveles de contaminación antes de que afecten la calidad del recurso hídrico.", style={"fontSize": "14px", "color": "#57606f"})], style={"backgroundColor": "white", "padding": "25px", "borderRadius": "15px", "boxShadow": "0 4px 6px rgba(0,0,0,0.15)", "height": "100%"}), md=6, className="mb-3"),
            ])
        ], className="p-2")
        
    elif active_tab == "tab-metodologia":
        return html.Div([
            html.H2("Metodología del Proyecto", className="fw-bold mb-4", style={"color": "#ffffff"}),
            dbc.Row([
                dbc.Col(html.Div([html.H5("🔄 Pipeline de Datos (H₂O Workflow)", className="fw-bold mb-4", style={"color": "#1a252f"}), html.Div([html.H6("1. Ingesta & Análisis Exploratorio (EDA)", className="fw-bold text-primary"), html.P("Evaluación del dataset original. Identificación de un volumen crítico de datos faltantes.", style={"fontSize": "13px", "color": "#57606f"})], className="border-left pl-3 mb-3", style={"borderLeft": "3px solid #007bff"}), html.Div([html.H6("2. Tratamiento con IterativeImputer", className="fw-bold text-warning"), html.P("Uso de imputación multivariada por cadenas de ecuaciones (MICE).", style={"fontSize": "13px", "color": "#57606f"})], className="border-left pl-3 mb-3", style={"borderLeft": "3px solid #ff9f43"}), html.Div([html.H6("3. Modelado Predictivo (Machine Learning)", className="fw-bold text-success"), html.P("Entrenamiento de algoritmos de regresión.", style={"fontSize": "13px", "color": "#57606f"})], className="border-left pl-3", style={"borderLeft": "3px solid #28a745"})], style={"backgroundColor": "white", "padding": "25px", "borderRadius": "15px", "boxShadow": "0 4px 6px rgba(0,0,0,0.15)", "height": "100%"}), md=5, className="mb-3"),
                dbc.Col(html.Div([html.H5("📋 Estado de Calidad de las Variables", className="fw-bold mb-3", style={"color": "#1a252f"}), html.P("Resumen del estado del dataset antes y después de aplicar las técnicas de imputación:", style={"fontSize": "13px", "color": "#6c757d"}), dbc.Table.from_dataframe(data_agua, striped=True, bordered=True, hover=True, className="align-middle text-center", style={"fontSize": "13px", "backgroundColor": "white"})], style={"backgroundColor": "white", "padding": "25px", "borderRadius": "15px", "boxShadow": "0 4px 6px rgba(0,0,0,0.15)", "height": "100%"}), md=7, className="mb-3"),
            ])
        ], className="p-2")
        
    elif active_tab == "tab-eda":
        return html.Div([
            html.H2("Análisis Exploratorio de Datos (EDA)", className="fw-bold mb-4", style={"color": "#ffffff"}),
            dbc.Row([
                dbc.Col(html.Div([html.H5("📊 Parámetros de Control", className="fw-bold mb-3", style={"color": "#1a252f"}), html.Label("Seleccione la variable a visualizar:", className="text-muted fw-bold", style={"fontSize": "13px"}), dcc.Dropdown(id="dropdown-variable-eda", options=[{"label": var, "value": var} for var in data_historica["Variable"].unique()], value="BOD (Demanda Bioquímica)", clearable=False, style={"borderRadius": "8px", "fontSize": "14px", "color": "#000"}), html.Hr(), html.P(["💡 ", html.Strong("Observación del EDA:"), " El comportamiento del ", html.Strong("BOD"), " muestra una correlación indirecta con los niveles extremos de pH."], style={"fontSize": "13px", "color": "#57606f", "backgroundColor": "#f1f2f6", "padding": "15px", "borderRadius": "10px"})], style={"backgroundColor": "white", "padding": "25px", "borderRadius": "15px", "boxShadow": "0 4px 6px rgba(0,0,0,0.15)", "height": "100%"}), md=4, className="mb-3"),
                dbc.Col(html.Div([html.H5("📈 Comportamiento de Tendencia Temporal", className="fw-bold mb-2", style={"color": "#1a252f"}), dcc.Graph(id="grafico-dinamico-eda")], style={"backgroundColor": "white", "padding": "25px", "borderRadius": "15px", "boxShadow": "0 4px 6px rgba(0,0,0,0.15)", "height": "100%"}), md=8, className="mb-3"),
            ])
        ], className="p-2")

    elif active_tab == "tab-metricas":
        df_r2 = data_metricas[data_metricas["Métrica"] == "R² Score (Precisión)"]
        fig_metricas = px.bar(df_r2, x="Escenario", y="Valor", color="Escenario", color_discrete_map={"Antes de Imputar (Drop NaNs)": "#95a5a6", "Después de IterativeImputer": "#2ecc71"}, text_auto=True)
        fig_metricas.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", showlegend=False, margin={"l": 40, "r": 20, "t": 20, "b": 40}, yaxis={"title": "Coeficiente de Determinación (R²)", "gridcolor": "#f1f2f6"})
        
        return html.Div([
            html.H2("Métricas de Rendimiento del Modelo", className="fw-bold mb-4", style={"color": "#ffffff"}),
            dbc.Row([
                dbc.Col([
                    html.Div([html.H6("📈 R² SCORE LOGRADO", className="text-muted fw-bold mb-1", style={"fontSize": "12px"}), html.H2("0.86", className="fw-bold text-success mb-2"), html.P("El modelo explica el 86% de la variabilidad del BOD tras balancear y rellenar nulos con MICE.", style={"fontSize": "13px", "color": "#57606f"})], style={"backgroundColor": "white", "padding": "20px", "borderRadius": "15px", "boxShadow": "0 4px 6px rgba(0,0,0,0.15)", "marginBottom": "15px"}),
                    html.Div([html.H6("📉 ERROR MEDIO ABSOLUTO (MAE)", className="text-muted fw-bold mb-1", style={"fontSize": "12px"}), html.H2("2.1 mg/L", className="fw-bold text-info mb-2"), html.P("Reducción drástica del error de predicción en comparación al modelo entrenado omitiendo nulos.", style={"fontSize": "13px", "color": "#57606f"})], style={"backgroundColor": "white", "padding": "20px", "borderRadius": "15px", "boxShadow": "0 4px 6px rgba(0,0,0,0.15)"})
                ], md=4, className="mb-3"),
                dbc.Col(html.Div([html.H5("📊 Impacto de la Imputación en la Precisión del Modelo", className="fw-bold mb-3", style={"color": "#1a252f"}), dcc.Graph(figure=fig_metricas)], style={"backgroundColor": "white", "padding": "25px", "borderRadius": "15px", "boxShadow": "0 4px 6px rgba(0,0,0,0.15)"}), md=8, className="mb-3")
            ])
        ], className="p-2")

    elif active_tab == "tab-prediccion":
        return html.Div([
            html.H2("Módulo de Predicción en Tiempo Real", className="fw-bold mb-4", style={"color": "#ffffff"}),
            dbc.Row([
                dbc.Col(html.Div([
                    html.H5("🎛️ Variables de Entrada", className="fw-bold mb-3", style={"color": "#1a252f"}),
                    html.P("Ajuste los valores medidos en campo para calcular la demanda biológica proyectada:", style={"fontSize": "13px", "color": "#6c757d"}),
                    html.Label("Nivel de pH:", className="fw-bold mb-1", style={"fontSize": "13px", "color": "#000"}),
                    dbc.Input(id="input-ph", type="number", value=7.4, min=0, max=14, step=0.1, className="mb-3", style={"borderRadius": "8px"}),
                    html.Label("Nitrógeno Amonio (mg/L):", className="fw-bold mb-1", style={"fontSize": "13px", "color": "#000"}),
                    dbc.Input(id="input-nitrogeno", type="number", value=2.5, min=0, step=0.1, className="mb-3", style={"borderRadius": "8px"}),
                    html.Label("Oxígeno Disuelto (mg/L):", className="fw-bold mb-1", style={"fontSize": "13px", "color": "#000"}),
                    dbc.Input(id="input-oxigeno", type="number", value=5.0, min=0, step=0.1, className="mb-2", style={"borderRadius": "8px"}),
                ], style={"backgroundColor": "white", "padding": "25px", "borderRadius": "15px", "boxShadow": "0 4px 6px rgba(0,0,0,0.15)"}), md=5, className="mb-3"),
                dbc.Col(html.Div([
                    html.H5("🔮 Estimación Proyectada del Target", className="fw-bold mb-4", style={"color": "#1a252f"}),
                    html.Div([
                        html.H6("BOD (DEMANDA BIOQUÍMICA DE OXÍGENO) CALCULADA:", className="text-muted fw-bold mb-2", style={"fontSize": "12px", "letterSpacing": "1px"}),
                        html.Div(id="resultado-prediccion-bod", className="fw-bold mb-2", style={"fontSize": "56px"}),
                        html.Div(id="status-badge-bod")
                    ], style={"padding": "40px 30px", "borderRadius": "15px", "textAlign": "center", "backgroundColor": "#f8f9fa", "border": "1px solid #e9ecef"}),
                    html.P(["⚠️ ", html.Strong("Nota de Ingeniería:"), " Un valor de BOD superior a 50 mg/L se considera un indicador de alta carga orgánica residual."], className="text-muted mt-3", style={"fontSize": "12px"})
                ], style={"backgroundColor": "white", "padding": "25px", "borderRadius": "15px", "boxShadow": "0 4px 6px rgba(0,0,0,0.15)", "height": "100%"}), md=7, className="mb-3")
            ])
        ], className="p-2")
    else:
        return html.Div([html.H3("Pestaña no encontrada", className="text-danger")])

# ==========================================
# 4. CALLBACKS DE INTERACTIVIDAD INTERNA
# ==========================================

@app.callback(
    Output("grafico-dinamico-eda", "figure"),
    [Input("dropdown-variable-eda", "value")]
)
def update_eda_graph(variable_seleccionada):
    df_filtrado = data_historica[data_historica["Variable"] == variable_seleccionada]
    fig = px.line(
        df_filtrado, 
        x="Mes", 
        y="Valor", 
        markers=True,
        color_discrete_sequence=["#00d2d3" if "BOD" in variable_seleccionada else "#54a0ff"]
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin={"l": 40, "r": 20, "t": 20, "b": 40},
        xaxis={"gridcolor": "#f1f2f6", "title": "Meses del Período"},
        yaxis={"gridcolor": "#f1f2f6", "title": "Valor de Medición"},
        hovermode="x unified"
    )
    return fig


@app.callback(
    [Output("resultado-prediccion-bod", "children"),
     Output("resultado-prediccion-bod", "style"),
     Output("status-badge-bod", "children")],
    [Input("input-ph", "value"),
     Input("input-nitrogeno", "value"),
     Input("input-oxigeno", "value")]
)
def calcular_prediccion_h2o(ph, nitrogeno, oxigeno):
    if ph is None or nitrogeno is None or oxigeno is None:
        return "---", {"color": "#7f8c8d"}, dbc.Badge("Esperando datos", color="secondary")
    
    bod_estimado = (nitrogeno * 15.4) + (ph * 3.2) - (oxigeno * 1.5) + 12.3
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

# ==========================================
# 5. EJECUCIÓN DEL SERVIDOR
# ==========================================
if __name__ == "__main__":
    app.run(debug=True, port=8050)