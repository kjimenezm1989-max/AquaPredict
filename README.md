# 🏢 Employee Attrition Dashboard

Dashboard analítico de rotación laboral construido con Dash, Plotly y Scikit-Learn.

---

## 🗂️ Estructura del Proyecto

```
employee_attrition/
├── app.py                     ← Archivo principal
├── requirements.txt           ← Dependencias
├── README.md
├── data/
│   ├── __init__.py
│   └── generate_data.py       ← Genera dataset sintético
├── model/
│   ├── __init__.py
│   ├── train_model.py         ← Entrena y guarda model.pkl
│   └── model.pkl              ← Generado al ejecutar train_model.py
└── tabs/
    ├── __init__.py
    ├── contextoproblema.py    ← Tab 1: Contexto empresarial
    ├── metodologia.py         ← Tab 2: Metodología
    ├── eda.py                 ← Tab 3: Análisis exploratorio
    ├── metricasmodelo.py      ← Tab 4: Métricas del modelo
    └── prediccionmodelo.py    ← Tab 5: Predicción interactiva
```

---

## 🚀 Instrucciones de Ejecución

### 1. Requisitos previos
- Python 3.10 o 3.11 (recomendado)

### 2. Crear entorno virtual

```bash
# macOS / Linux
python3.11 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Generar el dataset sintético

```bash
python data/generate_data.py
```

### 5. Entrenar el modelo

```bash
python model/train_model.py
```

> Esto crea el archivo `model/model.pkl`. **Este paso es obligatorio antes de ejecutar el dashboard.**

### 6. Ejecutar el dashboard

```bash
python app.py
```

Abre tu navegador en: **http://localhost:8050**

---

## 🌐 Despliegue en producción

```bash
pip install gunicorn
gunicorn app:server -b 0.0.0.0:8050
```

---

## 📊 Pestañas del Dashboard

| Tab | Contenido |
|-----|-----------|
| 📋 Contexto | Impacto empresarial de la rotación laboral |
| 🔬 Metodología | Dataset, variables y pipeline del modelo |
| 📊 EDA | Dona, barplot, histogramas, correlación |
| 📈 Métricas | Accuracy, F1, ROC-AUC, matriz de confusión |
| 🔮 Predicción | Formulario interactivo con predicción en tiempo real |
