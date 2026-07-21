import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
import sqlite3

def tintar_video_timelapse(ruta_video, ruta_json, ruta_salida):
    print("Cargando datos de geometría...")
    with open(ruta_json, 'r') as archivo:
        datos_geometria = json.load(archivo)

    cap = cv2.VideoCapture(ruta_video)
    if not cap.isOpened():
        print("Error: No se pudo abrir el video original.")
        return

    # Propiedades del video original
    ancho = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_original = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Configurar el video de salida (Time-lapse a 5 FPS para que se vea pausado y analizable)
    fps_salida = 5 
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(ruta_salida, fourcc, fps_salida, (ancho, alto))

    # Diccionario de colores (Formato BGR de OpenCV)
    colores = {
        "celula": (255, 0, 0),    # Azul (Células sueltas)
        "rulcell": (0, 255, 0),   # Verde (Células dentro de un rouleaux)
        "rulo": (255, 255, 0),    # Celeste (Contorno del rouleaux completo)
        "solapa": (0, 0, 255)     # Rojo (Áreas de solapa / Watershed)
    }

    segundo_actual = 0
    intervalo = 4

    print("Generando reconstrucción visual...")
    
    while True:
        llave_tiempo = f"segundo_{segundo_actual}"
        
        # Si el JSON ya no tiene más segundos, terminamos
        if llave_tiempo not in datos_geometria:
            break

        # Calcular qué frame exacto corresponde a este segundo
        frame_objetivo = segundo_actual * fps_original
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_objetivo)
        ret, frame = cap.read()
        
        if not ret:
            break

        overlay = frame.copy()
        datos_frame = datos_geometria[llave_tiempo]

        # Iterar sobre las 4 clases de tu diccionario
        for clase, objetos in datos_frame.items():
            color = colores.get(clase, (255, 255, 255)) # Blanco por defecto
            
            # Iterar sobre cada objeto detectado de esa clase ("1", "2", "3"...)
            for id_objeto, coords in objetos.items():
                x_lista = coords["x"]
                y_lista = coords["y"]
                
                # Unir las listas X e Y en pares de coordenadas (x, y) para OpenCV
                puntos = np.array(list(zip(x_lista, y_lista)), dtype=np.int32)
                puntos = puntos.reshape((-1, 1, 2))
                
                # Tintar el interior del polígono
                cv2.fillPoly(overlay, [puntos], color=color)
                
                # Opcional: Dibujar el borde ligeramente más marcado
                cv2.polylines(overlay, [puntos], isClosed=True, color=color, thickness=2)

        # Fusionar el color con el frame original (Opacidad al 40%)
        alfa = 0.4
        frame_final = cv2.addWeighted(overlay, alfa, frame, 1 - alfa, 0)
        
        # Agregar una etiqueta de tiempo en la esquina para saber qué segundo es
        cv2.putText(frame_final, f"Tiempo: {segundo_actual}s", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Guardar el frame procesado
        out.write(frame_final)
        print(f"Frame del segundo {segundo_actual} procesado.")
        
        # Avanzar al siguiente bloque
        segundo_actual += intervalo

    cap.release()
    out.release()
    print(f"\n¡Listo! Time-Lapse generado en: {ruta_salida}")
    
def procesar_json_a_db(ruta_json, nombre_muestra):
    print(f"Cargando JSON para la muestra: {nombre_muestra}...")
    
    # 1. Leer el JSON original
    with open(ruta_json, 'r') as archivo:
        datos_geometria = json.load(archivo)

    # 2. Conectar a SQLite
    conexion = sqlite3.connect("resultados_agregacion.db")
    cursor = conexion.cursor()

    # Crear una única tabla resumen (si no existe)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS resumen_experimento (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            muestra TEXT,
            segundo INTEGER,
            celulas INTEGER,
            rulos INTEGER,
            rull_cel INTEGER,
            solapas INTEGER,
            area_total_rulos REAL
        )
    ''')
    conexion.commit()

    print("Procesando fotogramas e insertando en la base de datos...")

    # 3. Masticar los datos segundo a segundo
    for llave_tiempo, datos_frame in datos_geometria.items():
        # Extraer el número del segundo (ej: "segundo_4" -> 4)
        segundo_actual = int(llave_tiempo.split('_')[1])
        
        # Contadores en cero para este frame
        conteos = {"celula": 0, "rulo": 0, "rulcell": 0, "solapa": 0}
        area_total_rulos = 0.0

        for clase, objetos in datos_frame.items():
            if clase in conteos:
                # El número de elementos es simplemente cuántos objetos hay en el diccionario de esa clase
                conteos[clase] = len(objetos)
            
            # Si la clase es rulo, calculamos el área sumada de todos los rulos en este frame
            if clase == "rulo":
                for id_objeto, coords in objetos.items():
                    x_lista = coords["x"]
                    y_lista = coords["y"]
                    
                    # Reconstruir el contorno para OpenCV
                    puntos = np.array(list(zip(x_lista, y_lista)), dtype=np.float32)
                    
                    # Sumar al área total del frame
                    area_total_rulos += cv2.contourArea(puntos)

        # 4. Insertar la fila limpia y resumida en SQLite
        cursor.execute('''
            INSERT INTO resumen_experimento 
            (muestra, segundo, celulas, rulos, rull_cel, solapas, area_total_rulos)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (nombre_muestra, segundo_actual, conteos["celula"], conteos["rulo"], 
              conteos["rulcell"], conteos["solapa"], area_total_rulos))

    conexion.commit()
    conexion.close()
    print(f"¡Listo! Base de datos actualizada con los resultados de {nombre_muestra}.")

# --- Ejecución ---
if __name__ == "__main__":
    VIDEO_ORIGINAL = "C:/Users/benja/OneDrive - mail.pucv.cl/Laboratorio/Agregacion/Codigos/Segmentacion/Video/experimento_base_1080p.mp4"       # Reemplaza con tu archivo
    ARCHIVO_JSON = "C:/Users/benja/OneDrive - mail.pucv.cl/Laboratorio/Agregacion/Codigos/Segmentacion/Proyecto_agregacion/resultados_geometria.json"
    VIDEO_SALIDA = "C:/Users/benja/OneDrive - mail.pucv.cl/Laboratorio/Agregacion/Codigos/Segmentacion/Video/video_anotado_rouleaux.mp4"

    tintar_video_timelapse(VIDEO_ORIGINAL, ARCHIVO_JSON, VIDEO_SALIDA)
    MUESTRA = "plasma_base"
    
    procesar_json_a_db(ARCHIVO_JSON, MUESTRA)