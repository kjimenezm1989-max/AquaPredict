# ============================================================
# generate_data.py
# Genera un dataset sintético de rotación laboral (attrition)
# ============================================================

import numpy as np
import pandas as pd
import os


def generate_employee_data(n_samples: int = 1000, seed: int = 42) -> pd.DataFrame:
    """
    Genera datos sintéticos de empleados con variables relevantes
    para predecir rotación laboral.

    Args:
        n_samples: Número de registros a generar.
        seed: Semilla para reproducibilidad.

    Returns:
        pd.DataFrame con las variables del dataset.
    """
    rng = np.random.default_rng(seed)

    departamentos = ["Ventas", "TI", "RRHH", "Finanzas", "Operaciones", "Marketing"]

    edad             = rng.integers(22, 60, n_samples)
    salario          = rng.integers(1_800_000, 12_000_000, n_samples)  # COP
    años_empresa     = rng.integers(0, 20, n_samples)
    departamento     = rng.choice(departamentos, n_samples)
    satisfaccion     = rng.integers(1, 6, n_samples)           # Escala 1-5
    horas_trabajadas = rng.integers(35, 70, n_samples)         # Horas semanales
    promociones      = rng.integers(0, 5, n_samples)
    viajes_negocio   = rng.choice([0, 1, 2], n_samples, p=[0.5, 0.35, 0.15])
    distancia_hogar  = rng.integers(1, 50, n_samples)          # km
    balance_vida     = rng.integers(1, 5, n_samples)           # 1=malo, 4=excelente

    # Función de probabilidad de abandono basada en variables
    prob_abandono = (
        0.3  * (satisfaccion < 3).astype(float)
        + 0.2  * (horas_trabajadas > 55).astype(float)
        + 0.15 * (salario < 3_000_000).astype(float)
        + 0.1  * (años_empresa < 2).astype(float)
        + 0.1  * (balance_vida < 2).astype(float)
        + 0.1  * (viajes_negocio == 2).astype(float)
        + rng.uniform(0, 0.1, n_samples)
    )
    prob_abandono = np.clip(prob_abandono, 0, 1)
    abandono = (rng.uniform(0, 1, n_samples) < prob_abandono).astype(int)

    df = pd.DataFrame({
        "edad":             edad,
        "salario":          salario,
        "años_empresa":     años_empresa,
        "departamento":     departamento,
        "satisfaccion":     satisfaccion,
        "horas_trabajadas": horas_trabajadas,
        "promociones":      promociones,
        "viajes_negocio":   viajes_negocio,
        "distancia_hogar":  distancia_hogar,
        "balance_vida":     balance_vida,
        "abandono":         abandono,
    })

    return df


if __name__ == "__main__":
    df = generate_employee_data()
    output_path = os.path.join(os.path.dirname(__file__), "employees.csv")
    df.to_csv(output_path, index=False)
    print(f"✅ Dataset generado: {output_path}")
    print(f"   Registros: {len(df)} | Tasa de abandono: {df['abandono'].mean():.1%}")
