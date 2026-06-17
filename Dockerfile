# Usar una imagen oficial de Python ligera
FROM python:3.11-slim

# Establecer el directorio de trabajo en el contenedor
WORKDIR /code

# Copiar el archivo de requerimientos e instalar dependencias
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copiar todo el código del proyecto al contenedor
COPY . .

# Exponer el puerto estándar que usa Hugging Face Spaces (7860)
EXPOSE 7860

# Comando para arrancar la app usando Gunicorn en el puerto correcto
CMD ["gunicorn", "-b", "0.0.0.0:7860", "app:app.server"]