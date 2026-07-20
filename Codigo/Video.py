import cv2
import numpy as np
import time
import glob
import os
from ultralytics import YOLO

# ==========================================
# 1. Configuración de Rutas y Parámetros
# ==========================================
ruta_modelo = 'C:/Users/benja/OneDrive - mail.pucv.cl/Laboratorio/Agregacion/Codigos/Segmentacion/Modelos/best (2).pt'

# Cambia esto a la carpeta donde tienes la secuencia de imágenes de tu experimento
ruta_imagenes = 'H:/Alerror/Experimentos/Datos 1-15-2026/60%/20260115/*.png' 
ruta_video_salida = 'C:/Users/benja/Downloads/agregacion_1080p.mp4'

fps_video = 10  # Ajusta los cuadros por segundo según la velocidad real de tu experimento

# ==========================================
# 2. Inicialización
# ==========================================
print("Cargando modelo en la GPU...")
modelo_nativo = YOLO(ruta_modelo)

# Leer y ordenar cronológicamente las imágenes
lista_imagenes = sorted(glob.glob(ruta_imagenes))
total_frames = len(lista_imagenes)

if total_frames == 0:
    raise ValueError("No se encontraron imágenes en la ruta especificada.")

print(f"Se detectaron {total_frames} imágenes para procesar.")

# Configurar el escritor de video (Codec MP4, Resolución 1920x1080)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(ruta_video_salida, fourcc, fps_video, (1920, 1080))

# ==========================================
# 3. Bucle de Procesamiento en Tiempo Real
# ==========================================
inicio_total = time.time()

for i, ruta_img in enumerate(lista_imagenes):
    imagen = cv2.imread(ruta_img)
    if imagen is None:
        continue
        
    imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    imagen_gris_rgb = cv2.cvtColor(imagen_gris, cv2.COLOR_GRAY2RGB)
    
    # -- Recorte Central a 1920x1080 --
    alto_original, ancho_original = imagen_gris_rgb.shape[:2]
    centro_y, centro_x = alto_original // 2, ancho_original // 2
    
    y_inicio = centro_y - 540
    y_fin = centro_y + 540
    x_inicio = centro_x - 960
    x_fin = centro_x + 960
    
    imagen_recortada = imagen_gris_rgb[y_inicio:y_fin, x_inicio:x_fin]
    
    # -- Inferencia Nativa YOLO --
    resultados = modelo_nativo.predict(
        source=imagen_recortada,
        imgsz=1024, 
        device="cuda:0",
        verbose=False
    )
    
    # -- Dibujo Eficiente (Capa de Tinte y Bordes) --
    frame_final = imagen_recortada.copy()
    capa_tinte = imagen_recortada.copy()
    
    if resultados[0].masks is not None:
        contornos = resultados[0].masks.xy
        clases_detectadas = resultados[0].boxes.cls.cpu().numpy()
        nombres_clases = resultados[0].names
        
        for contorno, id_clase in zip(contornos, clases_detectadas):
            if len(contorno) > 0:
                pts = np.array(contorno, dtype=np.int32)
                nombre_clase = nombres_clases[int(id_clase)].lower()
                
                # Asignación estricta de colores BGR
                if "rulcell" in nombre_clase:
                    color = (0, 255, 255)     # Amarillo
                elif "rulo" in nombre_clase:
                    color = (0, 165, 255)     # Naranjo
                elif "celula" in nombre_clase or "célula" in nombre_clase:
                    color = (255, 255, 0)     # Cian
                elif "solapa" in nombre_clase:
                    color = (255, 0, 255)     # Magenta
                else:
                    color = (0, 255, 0)       # Verde por defecto
                
                cv2.fillPoly(capa_tinte, [pts], color=color)
                cv2.polylines(frame_final, [pts], isClosed=True, color=color, thickness=2)

        # Fusionar la transparencia
        opacidad_tinte = 0.35
        cv2.addWeighted(capa_tinte, opacidad_tinte, frame_final, 1 - opacidad_tinte, 0, frame_final)
    
    # Escribir el frame procesado en el video
    out.write(frame_final)
    
    # Progreso en consola para no pensar que Spyder se congeló
    if (i + 1) % 10 == 0 or (i + 1) == total_frames:
        tiempo_actual = time.time() - inicio_total
        print(f"Procesando: {i + 1}/{total_frames} frames... ({tiempo_actual:.1f}s transcurridos)")

# ==========================================
# 4. Cierre y Guardado
# ==========================================
out.release()
print(f"¡Video generado con éxito! Guardado en: {ruta_video_salida}")