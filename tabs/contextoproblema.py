# ============================================================
# tabs/contextoproblema.py
# Tab 1 – Contexto del Problema: Impacto empresarial del
# fenómeno de rotación laboral (Employee Attrition)
# ============================================================

import dash_bootstrap_components as dbc
from dash import html


# ── Paleta de colores pastel ──────────────────────────────────
PASTEL = {
    "red":    "#FFADAD",
    "orange": "#FFD6A5",
    "yellow": "#FDFFB6",
    "green":  "#CAFFBF",
    "blue":   "#9BF6FF",
    "indigo": "#A0C4FF",
    "violet": "#BDB2FF",
    "pink":   "#FFC6FF",
}

CARD_STYLE = {
    "borderRadius": "16px",
    "border": "none",
    "boxShadow": "0 2px 12px rgba(0,0,0,0.07)",
}


def kpi_card(value: str, label: str, color: str, icon: str) -> dbc.Col:
    """Genera una tarjeta KPI compacta con ícono y cifra destacada."""
    return dbc.Col(
        dbc.Card(
            dbc.CardBody([
                html.Div(icon, style={"fontSize": "2rem", "marginBottom": "6px"}),
                html.H3(value, className="fw-bold mb-1", style={"color": "#2d3436"}),
                html.P(label, className="text-muted mb-0", style={"fontSize": "0.85rem"}),
            ], className="text-center py-3"),
            style={**CARD_STYLE, "backgroundColor": color},
        ),
        xs=12, sm=6, md=3, className="mb-3",
    )


def layout() -> html.Div:
    """Retorna el layout completo del Tab de Contexto del Problema."""
    return html.Div([

        # ── Encabezado ────────────────────────────────────────
        dbc.Row(dbc.Col(html.Div([
            html.H2("¿Por qué importa la rotación laboral?",
                    className="fw-bold mb-2", style={"color": "#2d3436"}),
            html.P(
                "La rotación no deseada de talento es uno de los desafíos más costosos "
                "para las organizaciones modernas. Anticiparla con datos permite actuar "
                "antes de que ocurra.",
                className="lead text-muted",
            ),
        ], className="py-4")), className="mb-2"),

        # ── KPIs ─────────────────────────────────────────────
        dbc.Row([
            kpi_card("33%",  "Costo promedio de reemplazar un empleado (% del salario anual)",
                     PASTEL["red"],    "💸"),
            kpi_card("~60d", "Tiempo promedio para cubrir una vacante crítica",
                     PASTEL["orange"], "📅"),
            kpi_card("18%",  "Tasa promedio de rotación anual en Latinoamérica",
                     PASTEL["indigo"], "📊"),
            kpi_card("×4",   "Más probable que abandone un empleado insatisfecho",
                     PASTEL["pink"],   "⚠️"),
        ], className="mb-4"),

        # ── Impacto Empresarial ───────────────────────────────
        dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardHeader(html.H5("🏢 Impacto Organizacional", className="mb-0 fw-semibold")),
                dbc.CardBody(html.Ul([
                    html.Li("Pérdida de conocimiento institucional y relaciones con clientes."),
                    html.Li("Reducción en productividad del equipo durante la transición."),
                    html.Li("Costos de reclutamiento, selección y onboarding."),
                    html.Li("Deterioro del clima laboral por vacantes prolongadas."),
                    html.Li("Impacto negativo en la marca empleadora."),
                ], className="mb-0"), className="pt-3"),
            ], style=CARD_STYLE), md=6, className="mb-3"),

            dbc.Col(dbc.Card([
                dbc.CardHeader(html.H5("🎯 ¿Qué resuelve este dashboard?",
                                       className="mb-0 fw-semibold")),
                dbc.CardBody(html.Ul([
                    html.Li("Identificar los factores que más contribuyen al abandono."),
                    html.Li("Visualizar patrones de riesgo por departamento y perfil."),
                    html.Li("Evaluar la capacidad predictiva del modelo de ML."),
                    html.Li("Predecir en tiempo real si un empleado tiene riesgo de irse."),
                    html.Li("Habilitar decisiones preventivas basadas en evidencia."),
                ], className="mb-0"), className="pt-3"),
            ], style=CARD_STYLE), md=6, className="mb-3"),
        ]),

        # ── Variables clave ───────────────────────────────────
        dbc.Row(dbc.Col(dbc.Card([
            dbc.CardHeader(html.H5("🔑 Variables clave del análisis",
                                   className="mb-0 fw-semibold")),
            dbc.CardBody(dbc.Row([
                dbc.Col([
                    html.P("👤 Demográficas", className="fw-semibold text-secondary mb-1"),
                    html.P("Edad, años en la empresa, distancia al hogar", className="small"),
                ], md=3),
                dbc.Col([
                    html.P("💰 Económicas", className="fw-semibold text-secondary mb-1"),
                    html.P("Salario mensual, número de promociones recibidas", className="small"),
                ], md=3),
                dbc.Col([
                    html.P("😊 Bienestar", className="fw-semibold text-secondary mb-1"),
                    html.P("Satisfacción, balance vida-trabajo, viajes de negocio", className="small"),
                ], md=3),
                dbc.Col([
                    html.P("⏱️ Carga laboral", className="fw-semibold text-secondary mb-1"),
                    html.P("Horas trabajadas por semana, departamento", className="small"),
                ], md=3),
            ])),
        ], style=CARD_STYLE))),

    ], className="px-2 py-3")
