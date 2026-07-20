import cv2
import time
import numpy as np
from ultralytics import YOLO

# 1. Rutas (Verifica que sean las tuyas)
ruta_modelo = 'C:/Users/benja/OneDrive - mail.pucv.cl/Laboratorio/Agregacion/Codigos/Segmentacion/Modelos/best (2).pt'
imagen_prueba = 'H:/Alerror/Experimentos/Datos 1-15-2026/60%/20260115/IMG-1300.png'

# 2. Cargar modelo y preparar la imagen
print("Cargando modelo...")
modelo_nativo = YOLO(ruta_modelo)

imagen = cv2.imread(imagen_prueba)
imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
imagen_gris_rgb = cv2.cvtColor(imagen_gris, cv2.COLOR_GRAY2RGB)

# 3. Hacer el recorte (Cropping) al centro de 1920x1080
alto_original, ancho_original = imagen_gris_rgb.shape[:2]
centro_y, centro_x = alto_original // 2, ancho_original // 2

# Calculamos los límites: mitad de 1080 es 540, mitad de 1920 es 960
y_inicio = centro_y - 540
y_fin = centro_y + 540
x_inicio = centro_x - 960
x_fin = centro_x + 960

imagen_recortada = imagen_gris_rgb[y_inicio:y_fin, x_inicio:x_fin]

print(f"Tamaño original: {ancho_original}x{alto_original}")
print(f"Tamaño del recorte: {imagen_recortada.shape[1]}x{imagen_recortada.shape[0]}")

# 4. Inferencia YOLO nativa sobre el recorte
print("Aplicando inferencia YOLO nativa al recorte...")
inicio = time.time()

# imgsz=1024 ya no aplastará tanto la imagen porque partimos de 1920x1080
resultados = modelo_nativo.predict(
    source=imagen_recortada,
    imgsz=1024, 
    device="cuda:0",
    verbose=False
)

fin = time.time()
print(f"⏱️ Tiempo YOLO nativo en 1080p: {fin - inicio:.3f} segundos")

# 5. Dibujo con bordes y tinte según clase (Eficiente en RAM)
img_ploteada = imagen_recortada.copy()
capa_tinte = imagen_recortada.copy()

if resultados[0].masks is not None:
    contornos = resultados[0].masks.xy
    clases_detectadas = resultados[0].boxes.cls.cpu().numpy()
    nombres_clases = resultados[0].names
    
    for contorno, id_clase in zip(contornos, clases_detectadas):
        if len(contorno) > 0:
            pts = np.array(contorno, dtype=np.int32)
            nombre_clase = nombres_clases[int(id_clase)].lower()
            
            # Definir color en formato BGR de OpenCV (Azul, Verde, Rojo)
            if "rulcell" in nombre_clase:
                color = (0, 255, 255)     # Amarillo
            elif "rulo" in nombre_clase:
                color = (0, 165, 255)     # Naranjo
            elif "celula" in nombre_clase or "célula" in nombre_clase:
                color = (255, 255, 0)     # Cian
            elif "solapa" in nombre_clase:
                color = (255, 0, 255)     # Magenta
            else:
                color = (0, 255, 0)       # Verde por defecto si hay algo más
            
            # 1. Rellenar el polígono en la capa de tinte con el color de la clase
            cv2.fillPoly(capa_tinte, [pts], color=color)
            
            # 2. Dibujar el borde sólido
            cv2.polylines(img_ploteada, [pts], isClosed=True, color=color, thickness=2)

    # 3. Fusionar las capas
    opacidad_tinte = 0.35  # Ajusta esto si quieres más o menos transparencia
    cv2.addWeighted(capa_tinte, opacidad_tinte, img_ploteada, 1 - opacidad_tinte, 0, img_ploteada)

ruta_exportada = "C:/Users/benja/Downloads/prueba_recorte_colores.png"
cv2.imwrite(ruta_exportada, img_ploteada)
print(f"¡Imagen a color exportada en: {ruta_exportada}!")