import cv2
import json
import numpy as np
import time
from ultralytics import YOLO

# ==========================================
# 1. Configuración
# ==========================================
ruta_modelo = 'C:/Users/benja/OneDrive - mail.pucv.cl/Laboratorio/Agregacion/Codigos/Segmentacion/Modelos/best (2).pt'
ruta_video = 'C:/Users/benja/Downloads/experimento_base_1080p.mp4'
ruta_json_salida = 'C:/Users/benja/OneDrive - mail.pucv.cl/Laboratorio/Agregacion/Codigos/Segmentacion/Datos de salida/test_coordenadas.json'

print("Cargando modelo...")
modelo = YOLO(ruta_modelo)

cap = cv2.VideoCapture(ruta_video)
if not cap.isOpened():
    raise ValueError("Error al abrir el video.")

# ==========================================
# 2. Estructura de Datos
# ==========================================
# Aquí guardaremos toda la información del video
datos_video = {}

print("Iniciando extracción de coordenadas (Testeando los primeros 50 frames)...")
inicio = time.time()
frame_actual = 0

# Limitamos a 50 frames para esta prueba, ya que el JSON crecerá MUY rápido
while cap.isOpened() and frame_actual < 50:
    ret, frame = cap.read()
    if not ret:
        break
        
    frame_actual += 1
    
    # Inferencia
    resultados = modelo.predict(source=frame, imgsz=1024, device="cuda:0", verbose=False)
    
    # Estructura base para este frame
    datos_frame = {
        "celula": {},
        "rulcell": {},
        "rulo": {},
        "solapa": {}
    }
    
    # Contadores para asignar el ID (1, 2, 3...) a cada objeto de la clase
    contadores = {"celula": 0, "rulcell": 0, "rulo": 0, "solapa": 0}
    
    if resultados[0].masks is not None:
        contornos = resultados[0].masks.xy
        clases_detectadas = resultados[0].boxes.cls.cpu().numpy()
        nombres_clases = resultados[0].names
        
        for contorno, id_clase in zip(contornos, clases_detectadas):
            if len(contorno) > 0:
                nombre = nombres_clases[int(id_clase)].lower()
                
                # Clasificamos según tus nombres
                if "rulcell" in nombre: clave = "rulcell"
                elif "rulo" in nombre: clave = "rulo"
                elif "celula" in nombre or "célula" in nombre: clave = "celula"
                elif "solapa" in nombre: clave = "solapa"
                else: continue
                
                contadores[clave] += 1
                id_objeto = str(contadores[clave])
                
                # contorno es un arreglo numpy de (N, 2). 
                # Extraemos la columna 0 (X) y la 1 (Y), y las convertimos a enteros
                coordenadas_x = [int(p[0]) for p in contorno]
                coordenadas_y = [int(p[1]) for p in contorno]
                
                # Guardamos en la estructura que pediste
                datos_frame[clave][id_objeto] = {
                    "x": coordenadas_x,
                    "y": coordenadas_y
                }
    
    # Agregamos el frame al diccionario principal
    nombre_frame = f"frame_{frame_actual}"
    datos_video[nombre_frame] = datos_frame

cap.release()

# ==========================================
# 3. Exportar a JSON
# ==========================================
with open(ruta_json_salida, 'w') as archivo:
    json.dump(datos_video, archivo, indent=2)

print(f"¡Prueba finalizada en {time.time() - inicio:.2f} segundos!")
print(f"JSON exportado a: {ruta_json_salida}")
