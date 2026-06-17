# ============================================================
# tabs/prediccionmodelo.py
# Tab 5 – Predicción Interactiva en Tiempo Real
# Carga model.pkl y predice el riesgo de abandono de un empleado
# ============================================================

import os
import sys
import joblib
import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, callback
import plotly.graph_objects as go
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model.train_model import preprocess, get_feature_columns


# ── Carga el modelo una sola vez al importar el módulo ───────
_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "model.pkl")
_pipeline   = joblib.load(_MODEL_PATH)

CARD_STYLE   = {"borderRadius": "16px", "border": "none",
                "boxShadow": "0 2px 12px rgba(0,0,0,0.07)"}
DEPARTAMENTOS = ["Finanzas", "Marketing", "Operaciones", "RRHH", "TI", "Ventas"]


def _input_row(label: str, component, help_text: str = "") -> dbc.Row:
    """Fila de formulario con label + control + texto de ayuda opcional."""
    children = [
        dbc.Label(label, width=5, className="fw-semibold text-end pe-3"),
        dbc.Col(component, width=7),
    ]
    if help_text:
        children.append(
            dbc.Col(html.Small(help_text, className="text-muted"),
                    width={"offset": 5, "size": 7})
        )
    return dbc.Row(children, className="mb-3 align-items-center")


def layout() -> html.Div:
    """Retorna el layout del Tab de Predicción Interactiva."""
    return html.Div([

        dbc.Row(dbc.Col(html.Div([
            html.H2("Predicción de Riesgo de Abandono", className="fw-bold mb-2"),
            html.P("Ingresa el perfil de un empleado para obtener una predicción en tiempo real.",
                   className="lead text-muted"),
        ], className="py-4"))),

        dbc.Row([
            # ── Formulario de entrada ─────────────────────────
            dbc.Col(dbc.Card([
                dbc.CardHeader(html.H5("👤 Perfil del Empleado",
                                       className="mb-0 fw-semibold")),
                dbc.CardBody([
                    _input_row("Edad", dbc.Input(
                        id="pred-edad", type="number", value=35, min=22, max=59,
                        className="rounded-3")),

                    _input_row("Salario (COP)", dbc.Input(
                        id="pred-salario", type="number", value=4000000,
                        min=1800000, max=12000000, step=100000,
                        className="rounded-3"),
                        "Entre $1.800.000 y $12.000.000"),

                    _input_row("Años en la empresa", dbc.Input(
                        id="pred-años", type="number", value=3, min=0, max=19,
                        className="rounded-3")),

                    _input_row("Departamento", dcc.Dropdown(
                        id="pred-depto",
                        options=[{"label": d, "value": d} for d in DEPARTAMENTOS],
                        value="TI", clearable=False,
                        style={"borderRadius": "12px"})),

                    _input_row("Satisfacción", dcc.Slider(
                        id="pred-satisfaccion", min=1, max=5, step=1, value=3,
                        marks={i: str(i) for i in range(1, 6)},
                        tooltip={"placement": "bottom"})),

                    _input_row("Horas / semana", dbc.Input(
                        id="pred-horas", type="number", value=45, min=35, max=69,
                        className="rounded-3")),

                    _input_row("Promociones recibidas", dbc.Input(
                        id="pred-promociones", type="number", value=1, min=0, max=4,
                        className="rounded-3")),

                    _input_row("Viajes de negocio", dcc.Slider(
                        id="pred-viajes", min=0, max=2, step=1, value=0,
                        marks={0: "Ninguno", 1: "Ocasional", 2: "Frecuente"},
                        tooltip={"placement": "bottom"})),

                    _input_row("Distancia al hogar (km)", dbc.Input(
                        id="pred-distancia", type="number", value=10, min=1, max=49,
                        className="rounded-3")),

                    _input_row("Balance vida-trabajo", dcc.Slider(
                        id="pred-balance", min=1, max=4, step=1, value=3,
                        marks={1: "Malo", 2: "Regular", 3: "Bueno", 4: "Excelente"},
                        tooltip={"placement": "bottom"})),

                    dbc.Button(
                        "🔍 Predecir Riesgo", id="btn-predecir",
                        color="primary", className="w-100 mt-2 rounded-3",
                        style={
                            "backgroundColor": "#A0C4FF",
                            "border": "none",
                            "color": "#2d3436",
                            "fontWeight": "600",
                        },
                    ),
                ]),
            ], style=CARD_STYLE), md=6, className="mb-3"),

            # ── Panel de resultado ────────────────────────────
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("📊 Resultado de la Predicción",
                                           className="mb-0 fw-semibold")),
                    dbc.CardBody(html.Div(id="pred-resultado")),
                ], style=CARD_STYLE, className="mb-3"),

                dbc.Card([
                    dbc.CardHeader(html.H5("🎯 Probabilidad de Abandono",
                                           className="mb-0 fw-semibold")),
                    dbc.CardBody(dcc.Graph(
                        id="pred-gauge",
                        config={"displayModeBar": False},
                        style={"height": "260px"},
                    )),
                ], style=CARD_STYLE),
            ], md=6),
        ]),

    ], className="px-2 py-3")


# ── Callback de predicción ───────────────────────────────────
@callback(
    Output("pred-resultado", "children"),
    Output("pred-gauge",     "figure"),
    Input("btn-predecir",     "n_clicks"),
    Input("pred-edad",        "value"),
    Input("pred-salario",     "value"),
    Input("pred-años",        "value"),
    Input("pred-depto",       "value"),
    Input("pred-satisfaccion","value"),
    Input("pred-horas",       "value"),
    Input("pred-promociones", "value"),
    Input("pred-viajes",      "value"),
    Input("pred-distancia",   "value"),
    Input("pred-balance",     "value"),
    prevent_initial_call=False,
)
def predecir(n_clicks, edad, salario, años, depto, satisfaccion,
             horas, promociones, viajes, distancia, balance):
    """
    Construye el vector de entrada, llama al pipeline y
    devuelve la tarjeta de resultado + gauge de probabilidad.
    """
    # Valores por defecto si algún input es None
    edad         = edad         or 35
    salario      = salario      or 4_000_000
    años         = años         or 3
    depto        = depto        or "TI"
    satisfaccion = satisfaccion or 3
    horas        = horas        or 45
    promociones  = promociones  or 1
    viajes       = viajes       or 0
    distancia    = distancia    or 10
    balance      = balance      or 3

    # ── Construir DataFrame con dummies ──────────────────────
    row = {
        "edad": edad, "salario": salario, "años_empresa": años,
        "departamento": depto, "satisfaccion": satisfaccion,
        "horas_trabajadas": horas, "promociones": promociones,
        "viajes_negocio": viajes, "distancia_hogar": distancia,
        "balance_vida": balance,
        "abandono": 0,   # columna requerida por preprocess (se descarta)
    }
    df_input = pd.DataFrame([row])
    df_input = preprocess(df_input)
    features  = get_feature_columns()

    # Garantiza que todas las columnas existan
    for col in features:
        if col not in df_input.columns:
            df_input[col] = 0

    X    = df_input[features]
    prob = _pipeline.predict_proba(X)[0][1]
    pred = int(prob >= 0.5)

    # ── Determinar nivel de riesgo ────────────────────────────
    if prob < 0.4:
        alerta_color = "#CAFFBF"
        alerta_icon  = "🟢"
        alerta_texto = "Riesgo bajo de abandono"
        recom = [
            "Mantener el nivel actual de satisfacción y beneficios.",
            "Continuar con el plan de desarrollo profesional.",
            "Reconocer logros para fortalecer el vínculo.",
        ]
    elif prob < 0.65:
        alerta_color = "#FDFFB6"
        alerta_icon  = "🟡"
        alerta_texto = "Riesgo moderado de abandono"
        recom = [
            "Programar una conversación 1:1 con el empleado.",
            "Revisar opciones de flexibilidad o balance laboral.",
            "Evaluar si existe una promoción pendiente.",
        ]
    else:
        alerta_color = "#FFADAD"
        alerta_icon  = "🔴"
        alerta_texto = "Alto riesgo de abandono"
        recom = [
            "Revisar compensación y beneficios de manera urgente.",
            "Evaluar reducción de carga laboral o horas extra.",
            "Planificar plan de carrera o próxima promoción.",
            "Implementar encuesta de clima laboral inmediatamente.",
        ]

    # ── Tarjeta de resultado ──────────────────────────────────
    card_resultado = html.Div([
        html.Div([
            html.Span(alerta_icon, style={"fontSize": "2.5rem"}),
            html.Div([
                html.H4(alerta_texto, className="fw-bold mb-1"),
                html.H5(f"Probabilidad: {prob:.1%}",
                        className="text-muted mb-0"),
            ], className="ms-3"),
        ], className="d-flex align-items-center mb-3 p-3 rounded-3",
           style={"backgroundColor": alerta_color}),

        html.H6("💡 Recomendaciones:", className="fw-semibold mb-2"),
        html.Ul([html.Li(r) for r in recom], className="small"),
    ])

    # ── Gauge de probabilidad ─────────────────────────────────
    gauge_color = alerta_color
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(prob * 100, 1),
        number={"suffix": "%", "font": {"size": 36}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar":  {"color": gauge_color, "thickness": 0.3},
            "bgcolor": "#f8f9fa",
            "steps": [
                {"range": [0,  40],  "color": "#CAFFBF"},
                {"range": [40, 65],  "color": "#FDFFB6"},
                {"range": [65, 100], "color": "#FFADAD"},
            ],
            "threshold": {
                "line":      {"color": "#2d3436", "width": 3},
                "thickness": 0.8,
                "value":     50,
            },
        },
    ))
    fig_gauge.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#2d3436"),
        margin=dict(t=20, b=10, l=20, r=20),
        height=240,
    )

    return card_resultado, fig_gauge
