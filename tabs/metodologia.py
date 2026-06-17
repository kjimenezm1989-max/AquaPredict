# ============================================================
# tabs/metodologia.py
# Tab 2 – Metodología: descripción del dataset y del modelo
# ============================================================

import dash_bootstrap_components as dbc
from dash import html


CARD_STYLE = {
    "borderRadius": "16px",
    "border": "none",
    "boxShadow": "0 2px 12px rgba(0,0,0,0.07)",
}

PASTEL = {
    "green":  "#CAFFBF",
    "blue":   "#9BF6FF",
    "yellow": "#FDFFB6",
    "violet": "#BDB2FF",
}


def step_badge(number: str, color: str) -> html.Span:
    """Genera un badge circular numerado para el pipeline."""
    return html.Span(
        number,
        style={
            "backgroundColor": color,
            "borderRadius": "50%",
            "width": "32px",
            "height": "32px",
            "display": "inline-flex",
            "alignItems": "center",
            "justifyContent": "center",
            "fontWeight": "700",
            "marginRight": "10px",
            "fontSize": "0.9rem",
        }
    )


def layout() -> html.Div:
    """Retorna el layout del Tab de Metodología."""
    return html.Div([

        dbc.Row(dbc.Col(html.Div([
            html.H2("Metodología del Proyecto", className="fw-bold mb-2"),
            html.P("Cómo se generaron los datos, se prepararon las variables y se entrenó el modelo.",
                   className="lead text-muted"),
        ], className="py-4"))),

        # ── Bloque Dataset ────────────────────────────────────
        dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardHeader(html.H5("📦 Dataset Sintético", className="mb-0 fw-semibold")),
                dbc.CardBody([
                    html.P("Los datos fueron generados con NumPy con una función de probabilidad "
                           "de abandono multicausal, simulando condiciones realistas:"),
                    html.Ul([
                        html.Li("2.000 registros de empleados"),
                        html.Li("10 variables predictoras"),
                        html.Li("Variable objetivo binaria: abandono (0/1)"),
                        html.Li("Tasa de abandono ≈ 30–35% (desbalance moderado)"),
                        html.Li("Semilla fija para reproducibilidad (seed=42)"),
                    ]),
                    dbc.Table([
                        html.Thead(html.Tr([
                            html.Th("Variable"), html.Th("Tipo"), html.Th("Rango / Valores"),
                        ])),
                        html.Tbody([
                            html.Tr([html.Td("edad"),             html.Td("Numérica"),   html.Td("22–59 años")]),
                            html.Tr([html.Td("salario"),          html.Td("Numérica"),   html.Td("1.8M–12M COP")]),
                            html.Tr([html.Td("años_empresa"),     html.Td("Numérica"),   html.Td("0–19 años")]),
                            html.Tr([html.Td("departamento"),     html.Td("Categórica"), html.Td("6 áreas")]),
                            html.Tr([html.Td("satisfaccion"),     html.Td("Ordinal"),    html.Td("1–5")]),
                            html.Tr([html.Td("horas_trabajadas"), html.Td("Numérica"),   html.Td("35–69 h/sem")]),
                            html.Tr([html.Td("promociones"),      html.Td("Numérica"),   html.Td("0–4")]),
                            html.Tr([html.Td("viajes_negocio"),   html.Td("Ordinal"),    html.Td("0, 1, 2")]),
                            html.Tr([html.Td("distancia_hogar"),  html.Td("Numérica"),   html.Td("1–49 km")]),
                            html.Tr([html.Td("balance_vida"),     html.Td("Ordinal"),    html.Td("1–4")]),
                        ]),
                    ], bordered=True, hover=True, size="sm", className="mt-2"),
                ]),
            ], style=CARD_STYLE), md=7, className="mb-3"),

            # ── Bloque Modelo ─────────────────────────────────
            dbc.Col(dbc.Card([
                dbc.CardHeader(html.H5("🤖 Modelo: Regresión Logística",
                                       className="mb-0 fw-semibold")),
                dbc.CardBody([
                    html.P("Se eligió Regresión Logística por su interpretabilidad y "
                           "rendimiento sólido en problemas de clasificación binaria."),

                    html.Div([
                        html.Div([step_badge("1", "#CAFFBF"),
                                  html.Span("Generación del dataset (2.000 registros)")],
                                 className="d-flex align-items-center mb-2"),
                        html.Div([step_badge("2", "#9BF6FF"),
                                  html.Span("One-hot encoding de 'departamento'")],
                                 className="d-flex align-items-center mb-2"),
                        html.Div([step_badge("3", "#FDFFB6"),
                                  html.Span("División 80/20 estratificada (train/test)")],
                                 className="d-flex align-items-center mb-2"),
                        html.Div([step_badge("4", "#FFD6A5"),
                                  html.Span("StandardScaler para normalización")],
                                 className="d-flex align-items-center mb-2"),
                        html.Div([step_badge("5", "#BDB2FF"),
                                  html.Span("LogisticRegression con class_weight='balanced'")],
                                 className="d-flex align-items-center mb-2"),
                        html.Div([step_badge("6", "#FFC6FF"),
                                  html.Span("Serialización del pipeline completo en model.pkl")],
                                 className="d-flex align-items-center mb-2"),
                    ]),

                    dbc.Alert([
                        html.Strong("💡 ¿Por qué class_weight='balanced'? "),
                        "Corrige el desbalance de clases penalizando más los errores "
                        "sobre la clase minoritaria (abandono=1).",
                    ], color="info", className="mt-3 mb-0 small"),
                ]),
            ], style=CARD_STYLE), md=5, className="mb-3"),
        ]),

        # ── Pipeline Visual ───────────────────────────────────
        dbc.Row(dbc.Col(dbc.Card([
            dbc.CardHeader(html.H5("🔄 Pipeline de Scikit-Learn", className="mb-0 fw-semibold")),
            dbc.CardBody(
                html.Div([
                    html.Span("Raw Data", style={"background": "#CAFFBF",
                              "padding": "6px 14px", "borderRadius": "20px", "fontWeight": "600"}),
                    html.Span("  →  ", className="text-muted"),
                    html.Span("Preprocessing", style={"background": "#9BF6FF",
                              "padding": "6px 14px", "borderRadius": "20px", "fontWeight": "600"}),
                    html.Span("  →  ", className="text-muted"),
                    html.Span("StandardScaler", style={"background": "#FDFFB6",
                              "padding": "6px 14px", "borderRadius": "20px", "fontWeight": "600"}),
                    html.Span("  →  ", className="text-muted"),
                    html.Span("LogisticRegression", style={"background": "#BDB2FF",
                              "padding": "6px 14px", "borderRadius": "20px", "fontWeight": "600"}),
                    html.Span("  →  ", className="text-muted"),
                    html.Span("Predicción", style={"background": "#FFC6FF",
                              "padding": "6px 14px", "borderRadius": "20px", "fontWeight": "600"}),
                ], className="d-flex flex-wrap align-items-center gap-1 py-2"),
            ),
        ], style=CARD_STYLE))),

    ], className="px-2 py-3")
