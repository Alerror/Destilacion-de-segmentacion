# Usamos una versión ligera de Python 3.10
FROM python:3.10-slim

# Evita que Python genere archivos basura .pyc y fuerza a que los prints salgan de inmediato
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Instalamos librerías del sistema necesarias para que OpenCV funcione en Linux
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Creamos la carpeta de trabajo dentro del contenedor
WORKDIR /app

# Copiamos primero el archivo de requisitos e instalamos (esto optimiza el tiempo de construcción)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiamos todo el resto de tu código (FastAPI, el procesador_datos.py, etc.)
COPY . .

# Exponemos el puerto donde vivirá FastAPI
EXPOSE 8000

# El comando que arranca el servidor cuando el contenedor se enciende
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]