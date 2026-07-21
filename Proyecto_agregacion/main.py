from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
import cv2
import numpy as np
import tempfile
import os
import json

app = FastAPI(title="Motor de Inferencia Morfológica de Agregación")

print("Inicializando motor ONNX y cargando best.onnx...")
modelo = YOLO("best (2).onnx", task="segment") 

@app.post("/analizar_video/")
async def analizar_video(file: UploadFile = File(...)):
    """
    Recibe un video, salta cada 4 segundos exactos, lo procesa con ONNX 
    y devuelve la geometría completa estructurada en JSON.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(await file.read())
        video_path = temp_video.name

    cap = cv2.VideoCapture(video_path)
    
    # 1. MATEMÁTICA DEL TIEMPO: Calcular cuántos frames hay en 4 segundos
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps):
        fps = 10 # Respaldo por si el video no tiene metadatos legibles
        
    frames_por_intervalo = int(fps * 4) 
    
    datos_video = {}
    segundo_actual = 0
    frame_objetivo = 0

    while cap.isOpened():
        # 2. SALTO TEMPORAL: Ir directo al frame deseado sin leer los intermedios
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_objetivo)
        ret, frame = cap.read()
        
        if not ret:
            break # El video terminó
            
        nombre_tiempo = f"segundo_{segundo_actual}"
        
        # 3. INFERENCIA
        resultados = modelo.predict(source=frame, imgsz=1024, device="0", verbose=False)
        
        datos_frame = {"celula": {}, "rulcell": {}, "rulo": {}, "solapa": {}}
        contadores = {"celula": 0, "rulcell": 0, "rulo": 0, "solapa": 0}

        if resultados[0].masks is not None:
            contornos = resultados[0].masks.xy
            clases = resultados[0].boxes.cls.cpu().numpy()
            nombres = resultados[0].names

            for contorno, id_clase in zip(contornos, clases):
                if len(contorno) > 0:
                    nombre = nombres[int(id_clase)].lower()
                    
                    if "rulcell" in nombre: clave = "rulcell"
                    elif "rulo" in nombre: clave = "rulo"
                    elif "celula" in nombre or "célula" in nombre: clave = "celula"
                    elif "solapa" in nombre: clave = "solapa"
                    else: continue

                    contadores[clave] += 1
                    id_objeto = str(contadores[clave])
                    
                    datos_frame[clave][id_objeto] = {
                        "x": [int(p[0]) for p in contorno],
                        "y": [int(p[1]) for p in contorno]
                    }
        
        datos_video[nombre_tiempo] = datos_frame
        
        # 4. PREPARAR EL SIGUIENTE CICLO
        segundo_actual += 4
        frame_objetivo += frames_por_intervalo

    cap.release()
    os.remove(video_path)

    # Guardar el diccionario gigante directamente en un archivo físico
    with open("resultados_geometria.json", "w") as archivo_json:
        json.dump(datos_video, archivo_json)

    # Retornar un mensaje ligero para que el navegador no colapse
    return {
        "status": "completado", 
        "intervalo_segundos": 4, 
        "mensaje": "Análisis finalizado exitosamente. Archivo guardado como resultados_geometria.json"
    }