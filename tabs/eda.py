# ============================================================
# tabs/eda.py
# Tab 3 – Análisis Exploratorio de Datos (EDA)
# Incluye: dona de abandono, barplot, histogramas, correlación
# ============================================================

import sys
import os
import dash_bootstrap_components as dbc
from dash import html, dcc
import plotly.graph_objects as go
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data.generate_data import generate_employee_data


# ── Carga única del dataset al importar el módulo ────────────
_df = generate_employee_data(n_samples=1000)

# ── Estilos comunes ──────────────────────────────────────────
CARD_STYLE = {
    "borderRadius": "16px",
    "border": "none",
    "boxShadow": "0 2px 12px rgba(0,0,0,0.07)",
}
PASTEL_SEQ = [
    "#A0C4FF", "#BDB2FF", "#FFC6FF", "#FFADAD",
    "#FFD6A5", "#FDFFB6", "#CAFFBF", "#9BF6FF",
]
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#2d3436"),
    margin=dict(t=40, b=30, l=30, r=20),
)


def _dona_abandono() -> go.Figure:
    """Gráfico de dona mostrando proporción abandono vs permanece."""
    counts = _df["abandono"].value_counts()
    labels = ["Permanece", "Abandona"]
    values = [counts.get(0, 0), counts.get(1, 0)]
    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        hole=0.55,
        marker_colors=["#CAFFBF", "#FFADAD"],
        textinfo="label+percent",
        textfont_size=13,
    ))
    fig.update_layout(title="Distribución de Abandono", **PLOTLY_LAYOUT,
                      showlegend=False)
    return fig


def _bar_departamento() -> go.Figure:
    """Barplot horizontal con tasa de abandono por departamento."""
    tasa = (
        _df.groupby("departamento")["abandono"]
        .mean()
        .reset_index()
        .rename(columns={"abandono": "tasa_abandono"})
        .sort_values("tasa_abandono", ascending=True)
    )
    fig = go.Figure(go.Bar(
        x=tasa["tasa_abandono"],
        y=tasa["departamento"],
        orientation="h",
        marker_color=PASTEL_SEQ[:len(tasa)],
        text=[f"{v:.1%}" for v in tasa["tasa_abandono"]],
        textposition="outside",
    ))
    fig.update_layout(
        title="Tasa de Abandono por Departamento",
        xaxis=dict(tickformat=".0%", showgrid=True, gridcolor="#f0f0f0"),
        yaxis=dict(showgrid=False),
        **PLOTLY_LAYOUT,
    )
    return fig


def _histogramas() -> go.Figure:
    """Histograma superpuesto de satisfacción por grupo."""
    col = "satisfaccion"
    fig = go.Figure()
    for val, label, color in [(0, "Permanece", "#A0C4FF"), (1, "Abandona", "#FFADAD")]:
        subset = _df[_df["abandono"] == val][col]
        fig.add_trace(go.Histogram(
            x=subset, name=label,
            marker_color=color, opacity=0.75,
            xbins=dict(size=1),
        ))
    fig.update_layout(
        title="Distribución de Satisfacción por Grupo",
        barmode="overlay",
        xaxis_title="Satisfacción (1–5)",
        yaxis_title="Empleados",
        legend=dict(orientation="h", y=1.1),
        **PLOTLY_LAYOUT,
    )
    return fig


def _horas_boxplot() -> go.Figure:
    """Boxplot de horas trabajadas por grupo de abandono."""
    fig = go.Figure()
    for val, label, color in [(0, "Permanece", "#CAFFBF"), (1, "Abandona", "#FFADAD")]:
        subset = _df[_df["abandono"] == val]["horas_trabajadas"]
        fig.add_trace(go.Box(
            y=subset, name=label,
            marker_color=color,
            boxmean=True,
        ))
    fig.update_layout(
        title="Horas Trabajadas por Semana",
        yaxis_title="Horas / semana",
        **PLOTLY_LAYOUT,
    )
    return fig


def _correlacion() -> go.Figure:
    """Mapa de calor de correlaciones entre variables numéricas."""
    num_cols = [
        "edad", "salario", "años_empresa", "satisfaccion",
        "horas_trabajadas", "promociones", "distancia_hogar",
        "balance_vida", "abandono",
    ]
    corr = _df[num_cols].corr()
    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.index,
        colorscale=[
            [0,   "#FFADAD"],
            [0.5, "#FFFFFF"],
            [1,   "#A0C4FF"],
        ],
        zmin=-1, zmax=1,
        text=np.round(corr.values, 2),
        texttemplate="%{text}",
        textfont_size=10,
    ))
    fig.update_layout(
        title="Mapa de Correlación",
        **PLOTLY_LAYOUT,
        height=420,
    )
    return fig


def layout() -> html.Div:
    """Retorna el layout del Tab de EDA."""
    return html.Div([

        dbc.Row(dbc.Col(html.Div([
            html.H2("Análisis Exploratorio de Datos", className="fw-bold mb-2"),
            html.P("Distribuciones, patrones y correlaciones del dataset de empleados.",
                   className="lead text-muted"),
        ], className="py-4"))),

        # ── Fila 1: Dona + Barplot ────────────────────────────
        dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardBody(dcc.Graph(figure=_dona_abandono(),
                                      config={"displayModeBar": False}))
            ], style=CARD_STYLE), md=5, className="mb-3"),

            dbc.Col(dbc.Card([
                dbc.CardBody(dcc.Graph(figure=_bar_departamento(),
                                      config={"displayModeBar": False}))
            ], style=CARD_STYLE), md=7, className="mb-3"),
        ]),

        # ── Fila 2: Histograma + Boxplot ──────────────────────
        dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardBody(dcc.Graph(figure=_histogramas(),
                                      config={"displayModeBar": False}))
            ], style=CARD_STYLE), md=6, className="mb-3"),

            dbc.Col(dbc.Card([
                dbc.CardBody(dcc.Graph(figure=_horas_boxplot(),
                                      config={"displayModeBar": False}))
            ], style=CARD_STYLE), md=6, className="mb-3"),
        ]),

        # ── Fila 3: Correlación ───────────────────────────────
        dbc.Row(dbc.Col(dbc.Card([
            dbc.CardBody(dcc.Graph(figure=_correlacion(),
                                   config={"displayModeBar": False}))
        ], style=CARD_STYLE), className="mb-3")),

        # ── Estadísticos descriptivos ─────────────────────────
        dbc.Row(dbc.Col(dbc.Card([
            dbc.CardHeader(html.H5("📋 Estadísticos Descriptivos",
                                   className="mb-0 fw-semibold")),
            dbc.CardBody(
                dbc.Table.from_dataframe(
                    _df.describe().round(2).reset_index().rename(columns={"index": ""}),
                    striped=True, hover=True, bordered=False, size="sm",
                )
            ),
        ], style=CARD_STYLE))),

    ], className="px-2 py-3")
