# ============================================================
# tabs/metricasmodelo.py
# Tab 4 – Métricas del Modelo
# Accuracy, Precision, Recall, F1, ROC AUC, Matriz de Confusión
# ============================================================

import os
import sys
import joblib
import numpy as np
import dash_bootstrap_components as dbc
from dash import html, dcc
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix,
)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data.generate_data import generate_employee_data
from model.train_model import preprocess, get_feature_columns


# ── Carga modelo y calcula métricas al importar el módulo ────
def _load_and_evaluate():
    """Carga el modelo entrenado y evalúa sobre el test set."""
    model_path = os.path.join(os.path.dirname(__file__), "..", "model", "model.pkl")
    pipeline = joblib.load(model_path)

    df = generate_employee_data(n_samples=2000)
    df = preprocess(df)
    features = get_feature_columns()

    X = df[features]
    y = df["abandono"]
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    y_pred  = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall":    recall_score(y_test, y_pred),
        "f1":        f1_score(y_test, y_pred),
        "roc_auc":   roc_auc_score(y_test, y_proba),
    }
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    return metrics, fpr, tpr, cm


_metrics, _fpr, _tpr, _cm = _load_and_evaluate()

# ── Estilos ──────────────────────────────────────────────────
CARD_STYLE = {
    "borderRadius": "16px",
    "border": "none",
    "boxShadow": "0 2px 12px rgba(0,0,0,0.07)",
}
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#2d3436"),
    margin=dict(t=40, b=30, l=40, r=20),
)


def _metric_card(label: str, value: float, color: str, description: str) -> dbc.Col:
    """Tarjeta individual para cada métrica del modelo."""
    return dbc.Col(dbc.Card([
        dbc.CardBody([
            html.H3(f"{value:.1%}", className="fw-bold mb-1", style={"color": "#2d3436"}),
            html.P(label, className="fw-semibold mb-1"),
            html.P(description, className="text-muted mb-0", style={"fontSize": "0.78rem"}),
        ], className="text-center py-3"),
    ], style={**CARD_STYLE, "backgroundColor": color}),
    xs=12, sm=6, md=4, lg=2, className="mb-3")


def _roc_curve_fig() -> go.Figure:
    """Curva ROC con área bajo la curva."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=_fpr, y=_tpr, mode="lines",
        name=f"ROC (AUC={_metrics['roc_auc']:.3f})",
        line=dict(color="#A0C4FF", width=3),
        fill="tozeroy", fillcolor="rgba(160,196,255,0.15)",
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines", name="Línea base",
        line=dict(color="#cccccc", dash="dash"),
    ))
    fig.update_layout(
        title="Curva ROC",
        xaxis_title="Tasa de Falsos Positivos",
        yaxis_title="Tasa de Verdaderos Positivos",
        legend=dict(x=0.6, y=0.15),
        **PLOTLY_LAYOUT,
    )
    return fig


def _confusion_matrix_fig() -> go.Figure:
    """Heatmap de la matriz de confusión."""
    labels  = ["Permanece", "Abandona"]
    z_text  = [[str(v) for v in row] for row in _cm]
    fig = go.Figure(go.Heatmap(
        z=_cm, x=labels, y=labels,
        colorscale=[[0, "#FFFFFF"], [1, "#A0C4FF"]],
        text=z_text,
        texttemplate="<b>%{text}</b>",
        textfont_size=16,
        showscale=False,
    ))
    fig.update_layout(
        title="Matriz de Confusión",
        xaxis_title="Predicho",
        yaxis_title="Real",
        **PLOTLY_LAYOUT,
        height=340,
    )
    return fig


def layout() -> html.Div:
    """Retorna el layout del Tab de Métricas del Modelo."""
    return html.Div([

        dbc.Row(dbc.Col(html.Div([
            html.H2("Métricas del Modelo", className="fw-bold mb-2"),
            html.P(
                "Evaluación del modelo de Regresión Logística sobre el conjunto de prueba (20%).",
                className="lead text-muted",
            ),
        ], className="py-4"))),

        # ── KPIs de Métricas ──────────────────────────────────
        dbc.Row([
            _metric_card("Accuracy",  _metrics["accuracy"],
                         "#CAFFBF", "Proporción de predicciones correctas"),
            _metric_card("Precision", _metrics["precision"],
                         "#9BF6FF", "De los predichos positivos, ¿cuántos son reales?"),
            _metric_card("Recall",    _metrics["recall"],
                         "#FFD6A5", "De los reales positivos, ¿cuántos detectamos?"),
            _metric_card("F1-Score",  _metrics["f1"],
                         "#BDB2FF", "Media armónica de precision y recall"),
            _metric_card("ROC-AUC",   _metrics["roc_auc"],
                         "#FFC6FF", "Capacidad discriminatoria del modelo"),
        ], className="justify-content-center mb-4"),

        # ── ROC + Matriz de confusión ─────────────────────────
        dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardBody(dcc.Graph(figure=_roc_curve_fig(),
                                      config={"displayModeBar": False}))
            ], style=CARD_STYLE), md=7, className="mb-3"),

            dbc.Col(dbc.Card([
                dbc.CardBody(dcc.Graph(figure=_confusion_matrix_fig(),
                                      config={"displayModeBar": False}))
            ], style=CARD_STYLE), md=5, className="mb-3"),
        ]),

        # ── Interpretación ────────────────────────────────────
        dbc.Row(dbc.Col(dbc.Card([
            dbc.CardHeader(html.H5("📖 Interpretación de Métricas",
                                   className="mb-0 fw-semibold")),
            dbc.CardBody(dbc.Row([
                dbc.Col([
                    html.P("🟢 Recall alto", className="fw-semibold mb-1"),
                    html.P(
                        "En rotación laboral, minimizar los falsos negativos "
                        "(empleados que se van y no detectamos) es la prioridad. "
                        "Un recall alto indica que el modelo captura bien ese riesgo.",
                        className="small text-muted",
                    ),
                ], md=4),
                dbc.Col([
                    html.P("🔵 Precision y F1", className="fw-semibold mb-1"),
                    html.P(
                        "Un F1 equilibrado indica que el modelo no sobre-alerta. "
                        "La precision controla los recursos invertidos en retenciones "
                        "innecesarias de empleados que no iban a irse.",
                        className="small text-muted",
                    ),
                ], md=4),
                dbc.Col([
                    html.P("🟣 ROC-AUC", className="fw-semibold mb-1"),
                    html.P(
                        "Mide la capacidad de ordenar correctamente los empleados "
                        "de menor a mayor riesgo, independientemente del umbral "
                        "de decisión elegido.",
                        className="small text-muted",
                    ),
                ], md=4),
            ])),
        ], style=CARD_STYLE))),

    ], className="px-2 py-3")
