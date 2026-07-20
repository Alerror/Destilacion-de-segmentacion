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
ruta_video_entrada = 'C:/Users/benja/Downloads/experimento_base_1080p.mp4'
ruta_video_salida = 'C:/Users/benja/Downloads/comparacion_IA.mp4'

# ==========================================
# 2. Inicialización
# ==========================================
print("Cargando modelo en la GPU...")
modelo_nativo = YOLO(ruta_modelo)

cap = cv2.VideoCapture(ruta_video_entrada)
if not cap.isOpened():
    raise ValueError("No se pudo abrir el video base.")

fps_video = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Video de salida lado a lado (3840 x 1080)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(ruta_video_salida, fourcc, fps_video, (3840, 1080))

print("Iniciando inferencia en el video...")

# ==========================================
# 3. Bucle de Inferencia sobre el Video
# ==========================================
contador = 0
while cap.isOpened():
    ret, frame_original = cap.read()
    if not ret:
        break  # Fin del video
        
    contador += 1
    
    # Inferencia (El frame ya es 1920x1080, no hay que recortar)
    resultados = modelo_nativo.predict(source=frame_original, imgsz=1024, device="cuda:0", verbose=False)
    
    frame_procesado = frame_original.copy()
    capa_tinte = frame_original.copy()
    
    if resultados[0].masks is not None:
        contornos = resultados[0].masks.xy
        clases_detectadas = resultados[0].boxes.cls.cpu().numpy()
        nombres_clases = resultados[0].names
        
        for contorno, id_clase in zip(contornos, clases_detectadas):
            if len(contorno) > 0:
                pts = np.array(contorno, dtype=np.int32)
                nombre_clase = nombres_clases[int(id_clase)].lower()
                
                # Colores exactos
                if "rulcell" in nombre_clase:
                    color = (0, 255, 255)     # Amarillo
                elif "rulo" in nombre_clase:
                    color = (0, 165, 255)     # Naranjo
                elif "celula" in nombre_clase or "célula" in nombre_clase:
                    color = (255, 255, 0)     # Cian
                elif "solapa" in nombre_clase:
                    color = (255, 0, 255)     # Magenta
                else:
                    color = (0, 255, 0)       # Verde
                
                cv2.fillPoly(capa_tinte, [pts], color=color)
                cv2.polylines(frame_procesado, [pts], isClosed=True, color=color, thickness=2)

        opacidad_tinte = 0.35
        cv2.addWeighted(capa_tinte, opacidad_tinte, frame_procesado, 1 - opacidad_tinte, 0, frame_procesado)
    
    # Títulos
    cv2.putText(frame_original, "Video Base 1080p", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)
    cv2.putText(frame_procesado, "Deteccion YOLO", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)

    # Unir lado a lado
    frame_doble = cv2.hconcat([frame_original, frame_procesado])
    out.write(frame_doble)
    
    # Previsualización
    vista_previa = cv2.resize(frame_doble, (1920, 540)) 
    cv2.imshow("Inferencia en Video", vista_previa)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Cancelado por el usuario.")
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"¡Análisis completado! Video final guardado en: {ruta_video_salida}")