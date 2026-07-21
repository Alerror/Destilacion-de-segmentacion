from fastapi import FastAPI, File, UploadFile, BackgroundTasks 
from fastapi.responses import FileResponse
from ultralytics import YOLO
import cv2
import numpy as np
import tempfile
import os
import json
import zipfile 
import shutil 
import uuid 

from Postprocesado import procesar_json_a_db, graficar_resultados

app = FastAPI(title="Motor de Inferencia Morfológica de Agregación")

print("Inicializando motor pt y cargando best (2).pt...")
modelo = YOLO("best (2).pt", task="segment") 

@app.post("/analizar_video/")
# <-- CORRECCIÓN 2: Se agregó background_tasks como parámetro de la función
async def analizar_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    Recibe un video, salta cada 4 segundos exactos, lo procesa con YOLO pt 
    y devuelve la geometría completa estructurada en JSON, DB y Gráfico.
    """
    nombre_muestra = file.filename
    
    # 1. Crear un entorno aislado y único en /tmp
    id_tarea = str(uuid.uuid4())[:8] # Ej: 'a1b2c3d4'
    dir_trabajo = f"/tmp/analisis_{id_tarea}"
    os.makedirs(dir_trabajo, exist_ok=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(await file.read())
        video_path = temp_video.name

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps):
        fps = 10 
        
    frames_por_intervalo = int(fps * 4) 
    
    datos_video = {}
    segundo_actual = 0
    frame_objetivo = 0

    while cap.isOpened():
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_objetivo)
        ret, frame = cap.read()
        
        if not ret:
            break 
            
        nombre_tiempo = f"segundo_{segundo_actual}"
        
        # Inferencia con la GPU habilitada
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
                    
                    if "rull_cel" in nombre: 
                        clave = "rulcell"
                    elif "rulos" in nombre: 
                        clave = "rulo"
                    elif "celulas" in nombre: 
                        clave = "celula"
                    elif "solapa" in nombre: 
                        clave = "solapa"
                    else: 
                        continue

                    contadores[clave] += 1
                    id_objeto = str(contadores[clave])
                    
                    datos_frame[clave][id_objeto] = {
                        "x": [int(p[0]) for p in contorno],
                        "y": [int(p[1]) for p in contorno]
                    }
        
        datos_video[nombre_tiempo] = datos_frame
        segundo_actual += 4
        frame_objetivo += frames_por_intervalo

    cap.release()
    os.remove(video_path)

    # ---------------------------------------------------------
    # EL PUENTE: Usar las rutas temporales aisladas
    # ---------------------------------------------------------
    
    # <-- CORRECCIÓN 3: Pegamos el dir_trabajo al inicio de cada ruta
    ruta_db = f"{dir_trabajo}/resultados_agregacion.db" 
    ruta_png = f"{dir_trabajo}/grafico_{nombre_muestra}.png" 
    ruta_json = f"{dir_trabajo}/{nombre_muestra}.json"
    ruta_zip = f"/tmp/{nombre_muestra}_resultados.zip"

    # <-- CORRECCIÓN 4: Pasamos las nuevas rutas a las funciones de Postprocesado
    procesar_json_a_db(datos_video, nombre_muestra, ruta_db)
    graficar_resultados(nombre_muestra, ruta_db, ruta_png)

    # 6. Guardar JSON y armar el ZIP usando las rutas dinámicas
    with open(ruta_json, "w") as f:
        json.dump(datos_video, f, indent=4)

    with zipfile.ZipFile(ruta_zip, 'w') as zipf:
        if os.path.exists(ruta_json):
            zipf.write(ruta_json, arcname=f"{nombre_muestra}.json")
        if os.path.exists(ruta_db):
            zipf.write(ruta_db, arcname=f"{nombre_muestra}.db")
        if os.path.exists(ruta_png):
            zipf.write(ruta_png, arcname=f"{nombre_muestra}_grafico.png")

    # 7. La Barredora: Se ejecuta sola DESPUÉS de que el usuario reciba su archivo
    def limpiar_entorno():
        shutil.rmtree(dir_trabajo, ignore_errors=True)
        if os.path.exists(ruta_zip):
            os.remove(ruta_zip)

    background_tasks.add_task(limpiar_entorno)

    # 8. Usar FileResponse para disparar la descarga automática
    return FileResponse(
        path=ruta_zip,
        media_type="application/zip",
        filename=f"resultados_{nombre_muestra}.zip"
    )