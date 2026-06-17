# Usar una imagen oficial de Python ligera
FROM python:3.11-slim

# Establecer el directorio de trabajo en el contenedor
WORKDIR /code

# Copiar el archivo de requerimientos e instalar dependencias
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copiar todo el código del proyecto al contenedor
COPY . /code/

# Forzar a que la app escuche en el puerto que Render asigna dinámicamente o por defecto
EXPOSE 7860

# Comando directo que evita que Gunicorn falle al parsear el atributo de Dash
CMD ["gunicorn", "--workers", "1", "--bind", "0.0.0.0:7860", "app:server"]