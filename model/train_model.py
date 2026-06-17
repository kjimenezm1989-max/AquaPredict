# ============================================================
# train_model.py
# Entrena un modelo de Regresión Logística para predecir
# rotación laboral y lo guarda como model.pkl
# ============================================================

import os
import sys
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score

# Asegura que se puede importar desde la raíz del proyecto
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data.generate_data import generate_employee_data


def get_feature_columns() -> list:
    """Retorna la lista de columnas usadas como features."""
    return [
        "edad", "salario", "años_empresa", "satisfaccion",
        "horas_trabajadas", "promociones", "viajes_negocio",
        "distancia_hogar", "balance_vida",
        # Variables dummy del departamento
        "dept_Finanzas", "dept_Marketing", "dept_Operaciones",
        "dept_RRHH", "dept_TI", "dept_Ventas",
    ]


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica one-hot encoding al departamento y devuelve
    el dataframe listo para entrenamiento.
    """
    dummies = pd.get_dummies(df["departamento"], prefix="dept")
    df = pd.concat([df.drop(columns=["departamento"]), dummies], axis=1)

    # Garantiza que todas las columnas de dummies existan
    for dept in ["Finanzas", "Marketing", "Operaciones", "RRHH", "TI", "Ventas"]:
        col = f"dept_{dept}"
        if col not in df.columns:
            df[col] = 0

    return df


def train_and_save():
    """Genera datos, entrena el modelo y lo serializa en model.pkl."""

    print("📊 Generando datos sintéticos...")
    df = generate_employee_data(n_samples=2000)

    df = preprocess(df)
    features = get_feature_columns()

    X = df[features]
    y = df["abandono"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Pipeline: escalado + regresión logística
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(max_iter=1000, random_state=42,
                                      class_weight="balanced")),
    ])

    print("🤖 Entrenando Regresión Logística...")
    pipeline.fit(X_train, y_train)

    y_pred  = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    print("\n📈 Resultados en conjunto de prueba:")
    print(classification_report(y_test, y_pred, target_names=["Permanece", "Abandona"]))
    print(f"   ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")

    # Guarda el pipeline completo
    model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
    joblib.dump(pipeline, model_path)
    print(f"\n✅ Modelo guardado en: {model_path}")


if __name__ == "__main__":
    train_and_save()
