# Usar una imagen oficial de Python ligera
FROM python:3.11-slim

# Establecer el directorio de trabajo en el contenedor
WORKDIR /code

# Copiar el archivo de requerimientos e instalar dependencias
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copiar todo el código del proyecto al contenedor
COPY . /code/

# Exponer el puerto por defecto (Render usa el 10000 o el que asigne la variable PORT)
EXPOSE 10000

# Comando optimizado para arrancar Gunicorn en Render buscando la variable PORT interna
CMD ["gunicorn", "-b", "0.0.0.0:7860", "app:server"]