# � H2O Analytics - Control de Calidad del Agua

Dashboard interactivo de análisis de datos de calidad del agua usando Ciencia de Datos, IterativeImputer y Machine Learning.

---

## 🚀 Ejecutar en GitHub Codespaces (La forma más fácil)

### Opción 1: Desde el repositorio
1. Ve a [github.com/kjimenezm1989-max/AquaPredict](https://github.com/kjimenezm1989-max/AquaPredict)
2. Click en **Code** (botón verde)
3. Click en **Codespaces** → **Create codespace on main**
4. Espera ~2 minutos a que se configure
5. En la terminal, ejecuta:
   ```bash
   python app.py
   ```
6. Haz clic en el enlace "Open in Browser"

**¡Listo! Tu app está en línea sin instalar nada localmente.**

---

## 💻 Ejecutar en tu computadora

### Requisitos
- Python 3.10+
- Git

### Pasos

```bash
# 1. Clonar
git clone https://github.com/kjimenezm1989-max/AquaPredict.git
cd AquaPredict

# 2. Entorno virtual
python -m venv venv

# Activar (Windows)
venv\Scripts\activate
# Activar (macOS/Linux)
source venv/bin/activate

# 3. Instalar
pip install -r requirements.txt

# 4. Ejecutar
python app.py
```

Abre en navegador: **http://localhost:8050**

---

## 📊 Características

- 📋 **Contexto**: Análisis del problema de calidad del agua
- 🔬 **Metodología**: Técnicas de imputación de datos (IterativeImputer)
- 📊 **EDA**: Análisis exploratorio detallado
- 📈 **Métricas**: Modelos antes y después de imputación
- 🔮 **Predicción**: Predictor interactivo en tiempo real

---

## 🛠️ Stack

- **Frontend**: Dash + Plotly + Bootstrap
- **Backend**: Python 3.11
- **ML**: Scikit-learn
- **Data**: Pandas, NumPy

---

## 📋 Estructura

```
AquaPredict/
├── app.py
├── requirements.txt
├── Procfile
├── .devcontainer/
├── data/
├── model/
└── tabs/
```
