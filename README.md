# AquaPredict - Sistema de Calidad del Agua

Proyecto enfocado en la predicción de Demanda Bioquímica de Oxígeno (BOD) mediante técnicas de Machine Learning (XGBoost) y preprocesamiento MICE.

## Estructura del Proyecto
- `app.py`: Dashboard interactivo desarrollado en Dash.
- `train.csv` / `test.csv`: Datasets de entrenamiento y prueba.
- `mejor_modelo_xgb_pipeline.joblib`: Modelo entrenado.

## Instalación
1. Clonar el repositorio.
2. Instalar dependencias: `pip install -r requirements.txt`
3. Ejecutar: `python app.py`

## Sustentación Técnica
El modelo fue seleccionado tras comparar regresiones lineales (Baseline) con modelos de ensamble. **XGBoost** fue el ganador por su capacidad de capturar relaciones no lineales en variables fisico-químicas.